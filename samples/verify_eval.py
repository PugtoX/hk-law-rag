"""Verify eval_retrieval.py's recall math against a known hit/miss pattern."""
import importlib.util
import io
import sys
import types
from contextlib import redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

QUESTIONS = ["Q0", "Q1", "Q2", "Q3"]
RETRIEVED = [["Q0"], ["Q2"], ["Q2"], ["Q0"]]   # hits at i=0,2 -> 2/4

chroma = types.ModuleType("chromadb")


class _Col:
    def count(self):
        return len(QUESTIONS)

    def get(self, include=None):
        return {"documents": [f"DOC{i}" for i in range(len(QUESTIONS))],
                "metadatas": [{"question": q, "topic": "T"} for q in QUESTIONS]}

    def query(self, query_embeddings=None, n_results=1):
        return {"metadatas": [[{"question": q, "topic": "T"} for q in hits]
                              for hits in RETRIEVED],
                "documents": [[f"DOC::{q}" for q in hits] for hits in RETRIEVED],
                "distances": [[0.5] * len(hits) for hits in RETRIEVED]}


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

sys.argv = ["eval_retrieval.py", "1"]
buf = io.StringIO()
with redirect_stdout(buf):
    mod.main()
out = buf.getvalue()
print(out)

checks = {
    "recall line": "recall@1 (by question) = 2/4 = 50.0%" in out,
    "Q1 miss": "MISS: Q1" in out,
    "Q3 miss": "MISS: Q3" in out,
    "no false miss on Q0": "MISS: Q0" not in out,
}
for name, ok in checks.items():
    print(f"  {'PASS' if ok else 'FAIL'}: {name}")
assert all(checks.values()), f"failures: {[k for k, v in checks.items() if not v]}"
print("\nEVAL MATH VERIFIED")
