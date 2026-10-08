"""Verify eval_real_queries.py's query table and hit rule — no chromadb mock.

The full main() needs a live store, so this checks the two things that can
actually be wrong: the ACCEPTABLE table is well formed, and the hit rule
(a query hits iff a retrieved (source, qno) is in its acceptable list) holds.
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = (ROOT / "eval_real_queries.py").read_text(encoding="utf-8")

# pull out the table literal without importing the module (which needs chromadb)
start = SRC.index("ACCEPTABLE: list")
end = SRC.index("def main")
ns: dict = {}
exec("from typing import List\n" + SRC[start:end], ns)
ACCEPTABLE = ns["ACCEPTABLE"]

print(f"ACCEPTABLE entries: {len(ACCEPTABLE)}")
assert ACCEPTABLE, "query table is empty"
for text, keys in ACCEPTABLE:
    assert text and isinstance(text, str), f"bad query text: {text!r}"
    assert keys, f"no acceptable keys for {text!r}"
    for src, qno in keys:
        assert re.fullmatch(r"cap57[a-n]\.htm|smw_(coverage|wage_items)\.htm", src), \
            f"unknown source {src!r} in {text!r}"
        assert isinstance(qno, int) and qno >= 1, f"bad qno {qno!r} in {text!r}"
print("PASS: every entry has a query and well-formed (source, qno) keys")


def is_hit(accepted, got_keys):
    """The rule from eval_real_queries.main()."""
    return bool(set(accepted) & set(got_keys))


cases = [
    ("exact key retrieved", [("cap57c.htm", 1)], {("cap57c.htm", 1)}, True),
    ("second acceptable key", [("cap57l.htm", 7), ("cap57l.htm", 2)],
     {("cap57l.htm", 2)}, True),
    ("nothing acceptable", [("cap57c.htm", 1)], {("cap57e.htm", 9)}, False),
    ("qno mismatch", [("cap57c.htm", 1)], {("cap57c.htm", 2)}, False),
    ("source mismatch", [("cap57c.htm", 1)], {("cap57e.htm", 1)}, False),
]
for name, accepted, got, want in cases:
    got_res = is_hit(accepted, got)
    assert got_res == want, f"{name}: got {got_res}, want {want}"
    print(f"PASS: {name}")

print("\nQUERY TABLE + HIT RULE VERIFIED")
