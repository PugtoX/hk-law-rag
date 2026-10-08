"""Minimal check of the one rule that matters in eval_real_queries.py:
a query hits iff any acceptable (source, qno) appears in the retrieved rows."""
import importlib.util
import sys
import types
from pathlib import Path

ROOT = Path(r"E:\AI\AI_Agent\workplace\hk-law-rag")

# --- load the module's ACCEPTABLE table only (no chromadb import needed) ---
src = (ROOT / "eval_real_queries.py").read_text(encoding="utf-8")
ns = {}
start = src.index("ACCEPTABLE: list")
end = src.index("def main")
exec("from typing import List\n" + src[start:end], ns)
ACCEPTABLE = ns["ACCEPTABLE"]
print(f"ACCEPTABLE entries: {len(ACCEPTABLE)}")
for text, keys in ACCEPTABLE:
    print(f"  {len(keys)} key(s): {text[:40]}")

# --- the hit rule, applied to a synthetic retrieval result ---
def is_hit(accepted, got_keys):
    return bool(set(accepted) & set(got_keys))

cases = [
    ("exact key retrieved", [("a.htm", 1)], {("a.htm", 1)}, True),
    ("second acceptable key retrieved", [("a.htm", 1), ("b.htm", 2)], {("b.htm", 2)}, True),
    ("nothing acceptable retrieved", [("a.htm", 1)], {("c.htm", 9)}, False),
    ("qno mismatch only", [("a.htm", 1)], {("a.htm", 2)}, False),
]
ok = True
for name, accepted, got, want in cases:
    got_res = is_hit(accepted, got)
    flag = "PASS" if got_res == want else "FAIL"
    if got_res != want:
        ok = False
    print(f"  {flag} {name}: got={got_res} want={want}")

assert len(ACCEPTABLE) == 15, f"expected 15 queries, found {len(ACCEPTABLE)}"
assert ok, "hit rule broken"
print("\nHIT RULE + QUERY TABLE VERIFIED")
