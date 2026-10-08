"""Exercises search.py's arg/input loop with chromadb + model stubbed.
Verifies: each argv arg becomes its own query, argv then falls through to
interactive prompting, empty input exits. Catches infinite-loop regressions."""
import importlib.util
import sys
import types
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
RESULTS = []

# ---- stub chromadb ----
chroma = types.ModuleType("chromadb")


class _Col:
    def count(self):
        return 3

    def query(self, query_embeddings=None, n_results=1):
        return {"documents": [["DOC"]], "metadatas": [[{"topic": "T", "question": "Q"}]],
                "distances": [[0.5]]}


class _Client:
    def __init__(self, path=None):
        pass

    def get_collection(self, name):
        return _Col()


chroma.PersistentClient = _Client
sys.modules["chromadb"] = chroma

# ---- stub sentence_transformers ----
st = types.ModuleType("sentence_transformers")


class _Arr(list):
    def tolist(self):
        return list(self)


class _Model:
    def __init__(self, *a, **k):
        pass

    def encode(self, texts, **kw):
        RESULTS.append(list(texts))
        return _Arr([[0.0] * 4 for _ in texts])


st.SentenceTransformer = _Model
sys.modules["sentence_transformers"] = st

# ---- load the real module ----
spec = importlib.util.spec_from_file_location("search_mod", ROOT / "search.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

# case 1: three argv queries, each must be its own encode call
RESULTS.clear()
sys.argv = ["search.py", "試用期", "拖欠工資", "年假"]
mod.main()
assert RESULTS == [["試用期"], ["拖欠工資"], ["年假"]], RESULTS
print("case 1 OK: each argv arg queried separately ->", RESULTS)

# case 2: argv then interactive prompt; empty input must exit (no hang)
RESULTS.clear()
sys.argv = ["search.py", "病假有薪嗎"]
with mock.patch("builtins.input", side_effect=["年假", ""]):
    mod.main()
assert RESULTS == [["病假有薪嗎"], ["年假"]], RESULTS
print("case 2 OK: argv then interactive, empty exits ->", RESULTS)

# case 3: no args, EOF must exit cleanly
RESULTS.clear()
sys.argv = ["search.py"]
with mock.patch("builtins.input", side_effect=EOFError):
    mod.main()
assert RESULTS == [], RESULTS
print("case 3 OK: no args + EOF exits ->", RESULTS)

print("\nALL SEARCH.SPY LOOP TESTS PASSED")
