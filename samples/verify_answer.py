"""Verify answer.py: arg parsing, the prompt/loop, the distance gate, and the
Ollama HTTP call — all with chromadb / the model / urlopen stubbed."""
import importlib.util
import io
import json
import sys
import types
import urllib.error
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

ROOT = Path(r"E:\AI\AI_Agent\workplace\hk-law-rag")

# --- stub chromadb ---
chroma = types.ModuleType("chromadb")


class _Col:
    def count(self):
        return 84

    def query(self, query_embeddings=None, n_results=5):
        n = n_results
        row = [{"source": "cap57e.htm", "qno": 3, "topic": "休息日",
                "question": f"Q{i}"} for i in range(n)]
        return {"metadatas": [row],
                "documents": [[f"A{i}" for i in range(n)]],
                "distances": [[0.3, 0.4, 0.5, 0.6, 0.7][:n]]}


class _Client:
    def __init__(self, path=None):
        pass

    def get_collection(self, name):
        return _Col()


chroma.PersistentClient = _Client
sys.modules["chromadb"] = chroma

# --- stub sentence_transformers ---
st = types.ModuleType("sentence_transformers")


class _Arr(list):
    def tolist(self):
        return list(self)


class _Model:
    def __init__(self, *a, **k):
        pass

    def encode(self, texts, **kw):
        return _Arr([[0.0] * 4 for _ in texts])


st.SentenceTransformer = _Model
sys.modules["sentence_transformers"] = st

spec = importlib.util.spec_from_file_location("ans", ROOT / "answer.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

# --- parse_args ---
assert mod.parse_args(["x"]) == ("qwen3.5:9b", 5, False, False, [])
assert mod.parse_args(["x", "--model", "m", "--top-k", "3", "--debug", "q"]) \
       == ("m", 3, True, False, ["q"])
assert mod.parse_args(["x", "--no-rag", "q"]) == ("qwen3.5:9b", 5, False, True, ["q"])
print("PASS parse_args")

# --- prompt contains the constraint and the numbered articles ---
p = mod.build_prompt("休息日有薪嗎？", [{"topic": "休息日", "question": "Q1", "answer": "A1"}])
assert "只可根據" in p and "[1]" in p and "休息日有薪嗎？" in p
print("PASS build_prompt")

# --- call_ollama parses the response ---
class _Resp:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def read(self):
        return json.dumps({"response": "  有薪。  "}).encode()


with mock.patch("urllib.request.urlopen", return_value=_Resp()):
    assert mod.call_ollama("m", "p") == "有薪。"
print("PASS call_ollama parsing")

# --- loop exits on empty input (no hang), and answers via argv first ---
def run(argv, inputs=None):
    sys.argv = argv
    buf = io.StringIO()
    with mock.patch("urllib.request.urlopen", return_value=_Resp()):
        if inputs is None:
            with redirect_stdout(buf):
                mod.main()
        else:
            with mock.patch("builtins.input", side_effect=inputs), redirect_stdout(buf):
                mod.main()
    return buf.getvalue()

out = run(["answer.py", "休息日有薪嗎"], inputs=[""])
assert "有薪。" in out and "問題> " not in out.split("有薪。")[0][-20:], out
print("PASS: argv query answered, then empty input exits")

out = run(["answer.py"], inputs=EOFError)
assert out.count("有薪。") == 0
print("PASS: no args + EOF exits cleanly")

# --- distance gate fires without calling the model ---
mod._ORIG = mod.MAX_DIST
mod.MAX_DIST = 0.05          # force every retrieval to look unrelated
with mock.patch("urllib.request.urlopen", side_effect=AssertionError("model called!")):
    out = run(["answer.py", "無關的問題"])
assert "沒有相關規定" in out, out
print("PASS: distance gate refuses without calling the model")

# --- HTTP 404 from Ollama must name the likely cause, not "cannot connect" ---
mod.MAX_DIST = 2.0
err = urllib.error.HTTPError("http://x", 404, "Not Found", {}, None)
sys.argv = ["answer.py", "休息日有薪嗎"]
buf = io.StringIO()
with mock.patch("urllib.request.urlopen", side_effect=err), redirect_stdout(buf):
    mod.main()
out = buf.getvalue()
assert "HTTP 404" in out and "ollama list" in out, out
assert "連不上" not in out, out
print("PASS: HTTP 404 reported as a refused request (not a connection failure)")

# --- connection failure (service down) is a different message ---
buf = io.StringIO()
with mock.patch("urllib.request.urlopen",
                side_effect=urllib.error.URLError("Connection refused")), \
        redirect_stdout(buf):
    mod.main()
out = buf.getvalue()
assert "連不上" in out, out
print("PASS: connection failure reported separately")

print("\nANSWER.PY VERIFIED")
