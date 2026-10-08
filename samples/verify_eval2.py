"""Verify eval_retrieval.py both modes + arg parsing, with a mock store where
the hit pattern is known."""
import importlib.util
import io
import sys
import types
from contextlib import redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

QUESTIONS = ["Q0", "Q1", "Q2", "Q3"]
ANSWERS = ["A0", "A1", "A2", "A3"]
# query i retrieves: questions [Qi?] -> Q0 hit only for i=0 ; docs -> A0 only for i=0
RET_Q = [["Q0"], ["Q2"], ["Q2"], ["Q0"]]
RET_D = [["A0"], ["A2"], ["A2"], ["A0"]]

chroma = types.ModuleType("chromadb")


class _Col:
    def count(self):
        return len(QUESTIONS)

    def get(self, include=None):
        return {"documents": list(ANSWERS),
                "metadatas": [{"question": q, "topic": "T"} for q in QUESTIONS]}

    def query(self, query_embeddings=None, n_results=1):
        return {"metadatas": [[{"question": q, "topic": "T"} for q in row]
                              for row in RET_Q],
                "documents": list(RET_D),
                "distances": [[0.5] * len(row) for row in RET_Q]}


class _Client:
    def __init__(self, path=None):
        pass

    def get_collection(self, name):
        return _Col()


chroma.PersistentClient = _Client
sys.modules["chromadb"] = chroma

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

spec = importlib.util.spec_from_file_location("ev", ROOT / "eval_retrieval.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def run(argv):
    sys.argv = argv
    buf = io.StringIO()
    with redirect_stdout(buf):
        mod.main()
    return buf.getvalue()


# --by-question: Q0 appears in RET_Q[0] and RET_Q[3]; Q1 never -> 2/4
out = run(["eval_retrieval.py", "1"])
assert "recall@1 (by question) = 2/4 = 50.0%" in out, out
print("PASS by-question: 2/4 = 50.0%")

# --by-answer: gold answer A_i must appear in RET_D[i]; A0 appears for i=0 and i=3
# -> 2/4
out = run(["eval_retrieval.py", "--by-answer", "1"])
assert "recall@1 (by answer) = 2/4 = 50.0%" in out, out
print("PASS by-answer: 2/4 = 50.0%")

# default arg parsing
assert mod.parse_args(["x"]) == ("question", 5)
assert mod.parse_args(["x", "--by-answer", "10"]) == ("answer", 10)
print("PASS arg parsing")

print("\nBOTH EVAL MODES VERIFIED")
