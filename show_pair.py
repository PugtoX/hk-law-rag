#!/usr/bin/env python3
"""
Look up one Q&A pair by source and question number, to verify that an
'eval_real_queries.py' answer key really is the answer you meant.

Usage:  python show_pair.py cap57l.htm 7
        python show_pair.py cap57h.htm 5
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"


def read_any(p: Path) -> str:
    raw = p.read_bytes()
    for enc in ("utf-8-sig", "utf-8", "utf-16", "big5", "gb18030"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def main() -> None:
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    source, n = sys.argv[1], sys.argv[2]

    # import the parser from build_index.py so this stays in sync with it
    import importlib.util
    spec = importlib.util.spec_from_file_location("bi", ROOT / "build_index.py")
    bi = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bi)

    p = DATA / source
    if not p.exists():
        sys.exit(f"[!] {p} not found (data/ must hold the downloaded pages)")

    pairs = bi.parse_pairs(read_any(p))
    print(f"{source}: {len(pairs)} pairs parsed")
    hits = 0
    for i, (q, a) in enumerate(pairs, 1):
        # qno here is the parse order; match on the question text instead
        if q.strip() == n.strip() or str(i) == n:
            print(f"\n問{i}")
            print(f"  Q: {q}")
            print(f"  A: {a[:600]}")
            hits += 1
    if not hits:
        print(f"\n[!] no pair #{n} in {source}. Showing all questions:")
        for i, (q, _) in enumerate(pairs, 1):
            print(f"  {i:>2}. {q[:70]}")


if __name__ == "__main__":
    main()
