"""Download all 16 pages and report parsed pairs vs expected per page.
Locates the single missing pair (83 indexed vs 84 markers)."""
import importlib.util
import re
import sys
import urllib.request
from pathlib import Path
from unittest import mock

for n in ("chromadb", "sentence_transformers"):
    sys.modules.setdefault(n, mock.MagicMock())

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("bi", ROOT / "build_index.py")
bi = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bi)

UA = {"User-Agent": "Mozilla/5.0 (hk-law-rag study project)"}
BASE = "https://www.labour.gov.hk/tc/faq/"
targets = [f"cap57{c}_whole.htm" for c in "abcdefghijklmn"] + [
    "smw_coverage.htm", "smw_wage_items.htm"]

out = []
total = 0
for name in targets:
    req = urllib.request.Request(BASE + name, headers=UA)
    html = urllib.request.urlopen(req, timeout=25).read().decode("utf-8", "replace")
    anchors = len(re.findall(r'<a[^>]+(?:name|id)="Q\d+"', html))
    marks = len(set(re.findall(r"問\s*(\d+)\s*[.、:：]", html)))
    pairs = bi.parse_pairs(html)
    total += len(pairs)
    flag = "OK " if len(pairs) == marks else "LOSS"
    out.append(f"{flag} {name:<22} expected={marks:<3} parsed={len(pairs):<3} "
               f"anchors={anchors}")
    if len(pairs) != marks:
        got_q = {q for q, _ in pairs}
        # which marker numbers are missing?
        nums = sorted(int(x) for x in set(re.findall(r"問\s*(\d+)\s*[.、:：]", html)))
        out.append(f"     markers: {nums}")
        out.append(f"     parsed questions:")
        for q, a in pairs:
            out.append(f"       Q: {q[:60]!r}")
out.append(f"\nTOTAL parsed={total} (expected 84)")

(ROOT / "samples" / "verify_all16_out.txt").write_text("\n".join(out), encoding="utf-8")
print(f"total parsed = {total}")
print("written samples/verify_all16_out.txt")
