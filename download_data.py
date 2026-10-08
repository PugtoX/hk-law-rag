#!/usr/bin/env python3
"""
Step 1: fetch the Labour Department FAQ pages (public HK gov data).

Covers two chapters:
  * Employment Ordinance (Cap.57)    -> cap57a_whole.htm .. cap57n_whole.htm
  * Minimum Wage Ordinance (Cap.608) -> smw_coverage.htm, smw_wage_items.htm

Run from the project root:   python download_data.py
"""
import re
import urllib.request
from pathlib import Path

DATA = Path(__file__).resolve().parent / "data"
DATA.mkdir(exist_ok=True)

BASE = "https://www.labour.gov.hk/tc/faq/"
UA = {"User-Agent": "Mozilla/5.0 (hk-law-rag study project)"}

TARGETS = [f"cap57{c}_whole.htm" for c in "abcdefghijklmn"] + [
    "smw_coverage.htm",
    "smw_wage_items.htm",
]

saved, skipped = [], []
for name in TARGETS:
    try:
        req = urllib.request.Request(BASE + name, headers=UA)
        raw = urllib.request.urlopen(req, timeout=30).read()
    except Exception as e:
        skipped.append((name, f"{type(e).__name__}"))
        continue
    if len(raw) < 2000:                      # error / not-found page
        skipped.append((name, f"too small ({len(raw)} B)"))
        continue
    html = raw.decode("utf-8", errors="replace")
    pairs = len(re.findall(r'<a[^>]+(?:name|id)="Q\d+"', html))
    out = name.replace("_whole.htm", ".htm")
    (DATA / out).write_bytes(raw)
    saved.append((out, len(raw), pairs))

print(f"saved {len(saved)} pages into {DATA}")
total = 0
for name, size, pairs in saved:
    print(f"  {name:<20} {size:>7} B   ~{pairs} Q&A")
    total += pairs
if skipped:
    print("skipped:")
    for name, why in skipped:
        print(f"  {name}: {why}")
print(f"\napprox total Q&A: {total}")
print("[next] python build_index.py")
