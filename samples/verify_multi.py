"""Verify the shipped build_index.py against 6 real pages
(cap57b/i/l/n + the two minimum-wage pages). Stubs the heavy deps."""
import importlib.util
import shutil
import sys
from pathlib import Path
from unittest import mock

for name in ("chromadb", "sentence_transformers"):
    sys.modules.setdefault(name, mock.MagicMock())

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("bi", ROOT / "build_index.py")
bi = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bi)

tmp = ROOT / "_verify_data"
shutil.rmtree(tmp, ignore_errors=True)
tmp.mkdir()
names = ["cap57b.htm", "cap57i.htm", "cap57l.htm", "cap57n.htm",
         "smw_coverage.htm", "smw_wage_items.htm"]
for n in names:
    shutil.copy(ROOT / "samples" / n, tmp / n)
bi.DATA = tmp

items = bi.collect()

# expected pairs per page, counted from the anchor probe
EXPECT = {"cap57b.htm": 8, "cap57i.htm": 6, "cap57l.htm": 9,
          "cap57n.htm": 14, "smw_coverage.htm": 3, "smw_wage_items.htm": 3}
got = {}
for it in items:
    got[it["source"]] = got.get(it["source"], 0) + 1

print("\n=== assertions ===")
ok = True
for name, want in EXPECT.items():
    have = got.get(name, 0)
    flag = "PASS" if have == want else "FAIL"
    if have != want:
        ok = False
    print(f"  {flag} {name:<20} pairs={have} (expected {want})")

topics = {}
for it in items:
    topics.setdefault(it["source"], set()).add(it["topic"])
print("\ntopics detected:")
for k in sorted(topics):
    print(f"  {k:<22} {sorted(topics[k])}")

assert ok, "pair count mismatch"
assert all(len(v) == 1 for v in topics.values()), "a page got multiple topics"
print(f"\nTOTAL {len(items)} pairs — ALL ASSERTIONS PASSED")

# crude sanity on a known question/answer pair
sample = [it for it in items if it["source"] == "cap57n.htm"][0]
print("\nsample from 侍產假:", sample["question"][:50], "->", sample["answer"][:70])
shutil.rmtree(tmp, ignore_errors=True)
