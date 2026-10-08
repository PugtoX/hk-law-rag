"""Verify app.py's request handling with stubs, so it runs anywhere.

Covers: a normal ask (answer + hits + distances), the distance gate refusing
without calling the model, the generation-disabled path, the OpenAI-compatible
backend shape, and /api/health.
"""
import importlib.util
import json
import sys
import types
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------- chromadb
chroma = types.ModuleType("chromadb")


class _Col:
    DISTS = [0.31, 0.42, 0.53, 0.64, 0.75]

    def count(self):
        return 84

    def query(self, query_embeddings=None, n_results=5):
        n = n_results
        return {
            "documents": [[f"A{i}" for i in range(n)]],
            "metadatas": [[{"source": "cap57e.htm", "qno": 3, "topic": "休息日",
                            "question": f"Q{i}"} for i in range(n)]],
            "distances": [self.DISTS[:n]],
        }


class _Client:
    def __init__(self, path=None):
        pass

    def get_collection(self, name):
        return _Col()


chroma.PersistentClient = _Client
sys.modules["chromadb"] = chroma

# ------------------------------------------------- sentence_transformers
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

# ---------------------------------------------------------------- fastapi
fastapi = types.ModuleType("fastapi")


class _App:
    def __init__(self, **kw):
        pass

    def add_middleware(self, *a, **k):
        pass

    def get(self, path):
        return lambda f: f

    def post(self, path):
        return lambda f: f


fastapi.FastAPI = _App
mw = types.ModuleType("fastapi.middleware")
cors = types.ModuleType("fastapi.middleware.cors")
cors.CORSMiddleware = object
mw.cors = cors
resp = types.ModuleType("fastapi.responses")


class _FileResponse:
    def __init__(self, path):
        self.path = path


resp.FileResponse = _FileResponse
sys.modules.update({"fastapi": fastapi, "fastapi.middleware": mw,
                    "fastapi.middleware.cors": cors, "fastapi.responses": resp})

# --------------------------------------------------------------- pydantic
pydantic = types.ModuleType("pydantic")


class _BaseModel:
    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


pydantic.BaseModel = _BaseModel
sys.modules["pydantic"] = pydantic

# ------------------------------------------------------------- load app
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location("app_mod", ROOT / "app.py")
mod = importlib.util.module_from_spec(spec)
sys.modules["app_mod"] = mod
spec.loader.exec_module(mod)


class _Resp:
    def __init__(self, payload):
        self._payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def read(self):
        return json.dumps(self._payload).encode()


# --- 1) normal ask: ollama backend ---
with mock.patch("urllib.request.urlopen",
                return_value=_Resp({"response": "休息日有薪與否由雙方協議決定 [1]"})):
    res = mod.ask(mod.Ask(question="休息日有薪水嗎？"))
assert res["answer"] == "休息日有薪與否由雙方協議決定 [1]", res
assert len(res["hits"]) == 5 and res["hits"][0]["distance"] == 0.31, res["hits"]
assert res["refused"] is False
print("PASS: normal ask -> answer + 5 hits with distances")

# --- 2) distance gate: refuse without calling the model ---
_Col.DISTS = [1.31, 1.42, 1.53]
with mock.patch("urllib.request.urlopen", side_effect=AssertionError("model called!")):
    res = mod.ask(mod.Ask(question="今天天气怎么样？", top_k=3))
assert res["refused"] is True and res["answer"] is None, res
assert "未呼叫模型" in res["reason"], res
print("PASS: distance gate refuses without calling the model")

# --- 3) generate=False returns hits only ---
_Col.DISTS = [0.31, 0.42]
with mock.patch("urllib.request.urlopen", side_effect=AssertionError("model called!")):
    res = mod.ask(mod.Ask(question="休息日有薪水嗎？", top_k=2, generate=False))
assert res["answer"] is None and len(res["hits"]) == 2, res
print("PASS: generate=False returns retrieval only")

# --- 4) OpenAI-compatible backend (Groq etc.) ---
mod.PROVIDER = "openai"
mod.OPENAI_KEY = "test-key"
with mock.patch("urllib.request.urlopen",
                return_value=_Resp({"choices": [{"message": {"content": "遠端答案 [2]"}}]})):
    res = mod.ask(mod.Ask(question="休息日有薪水嗎？"))
assert res["answer"] == "遠端答案 [2]", res
print("PASS: openai-compatible backend parses choices[0].message.content")

# --- 5) missing key surfaces as an error, not a crash ---
mod.OPENAI_KEY = ""
res = mod.ask(mod.Ask(question="休息日有薪水嗎？"))
assert "OPENAI_KEY" in res.get("error", ""), res
print("PASS: missing OPENAI_KEY reported in error field")

# --- 6) health ---
h = mod.health()
assert h["pairs"] == 84 and h["embed_model"] == "BAAI/bge-m3", h
print("PASS: /api/health reports pairs and models")

# --- 7) index page is served from web/index.html ---
page = mod.index()
assert Path(page.path).name == "index.html", page.path
assert Path(page.path).parent.name == "web", page.path
print("PASS: index page path")

print("\nAPP.PY VERIFIED")
