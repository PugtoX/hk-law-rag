#!/usr/bin/env python3
"""
Real-user-style evaluation: colloquial queries, as a worker would actually ask.

Why this exists: eval_retrieval.py queries the corpus with the corpus's own
questions, so it scores ~100% by construction — a regression test, not a
benchmark. Here the queries are paraphrases, which is the honest measure.

A query may have SEVERAL acceptable answers (the same rule is often covered on
more than one page), so each entry carries a list of (source, 問n) and any of
them counts as a hit.

Edit ACCEPTABLE to add/repair keys. Verify a key with:
    python show_pair.py <source> <問n>
Usage:  python eval_real_queries.py
"""
import sys
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent
INDEX = ROOT / "chroma"
COLLECTION = "cap57"
EMBED_MODEL = "BAAI/bge-m3"

# (colloquial query, [(source, 問n), ...])
ACCEPTABLE: list[tuple[str, list[tuple[str, int]]]] = [
    ("公司给佣金算工资吗？", [("cap57c.htm", 1)]),
    ("休息日有薪水吗？", [("cap57e.htm", 3)]),
    ("法定假期要不要给钱？", [("cap57f.htm", 2)]),
    ("请假看病有钱吗？", [("cap57g.htm", 1)]),
    ("做满一年有多少天年假？", [("cap57i.htm", 1)]),
    ("老板突然把我辞退，要提前多久通知？", [("cap57d.htm", 1)]),
    ("被裁员能拿到什么钱？", [("cap57l.htm", 1), ("cap57l.htm", 2)]),
    ("太太生孩子，我有几天假？", [("cap57n.htm", 1)]),
    ("试用期被辞退有赔偿吗？", [("cap57k.htm", 4), ("cap57k.htm", 3),
                                ("cap57k.htm", 2)]),
    ("工作日加班有加班费吗？", [("cap57c.htm", 2)]),           # known gap
    ("离职时年假没休完怎么办？", [("cap57i.htm", 2), ("cap57i.htm", 6),
                                  ("cap57d.htm", 3)]),
    ("最低工资多少钱一小时？", [("smw_coverage.htm", 1)]),
    ("公司拖我工资怎么办？", [("cap57c.htm", 3)]),
    ("做满五年被裁，遣散费和长期服务金能同时拿吗？",
     [("cap57l.htm", 7), ("cap57l.htm", 2)]),
    ("怀孕了老板能开除我吗？", [("cap57h.htm", 5), ("cap57h.htm", 6),
                                ("cap57k.htm", 3)]),
]


def main() -> None:
    k = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 5

    model = SentenceTransformer(EMBED_MODEL)
    col = chromadb.PersistentClient(path=str(INDEX)).get_collection(COLLECTION)
    print(f"[i] {col.count()} pairs indexed, {len(ACCEPTABLE)} colloquial queries, top-{k}\n")

    texts = [q for q, _ in ACCEPTABLE]
    embs = model.encode(texts, normalize_embeddings=True).tolist()
    res = col.query(query_embeddings=embs, n_results=min(k, col.count()))

    hits, misses, bad_keys = 0, [], []
    for i, (text, accepted) in enumerate(ACCEPTABLE):
        metas = res["metadatas"][i]
        got = [(m["source"], int(m["qno"]), m["question"]) for m in metas]
        got_keys = {(s, n) for s, n, _ in got}
        if not (set(accepted) & got_keys):
            misses.append((text, accepted, got))
        else:
            hits += 1

    # verify the acceptable keys themselves exist in the corpus
    stored = col.get(include=["metadatas"])["metadatas"]
    stored_keys = {(m["source"], int(m["qno"])) for m in stored}
    for text, accepted in ACCEPTABLE:
        for key in accepted:
            if key not in stored_keys:
                bad_keys.append((text, key))

    print(f"recall@{k} (colloquial) = {hits}/{len(ACCEPTABLE)} = {hits / len(ACCEPTABLE):.1%}")
    if bad_keys:
        print(f"\n[!] {len(bad_keys)} acceptable keys do NOT exist in the corpus "
              f"(check with show_pair.py):")
        for text, key in bad_keys:
            print(f"    {key[0]} 問{key[1]}   <- {text}")
    if misses:
        print(f"\n--- {len(misses)} misses (no acceptable answer retrieved) ---")
        for text, accepted, got in misses:
            print(f"\nQ: {text}")
            print(f"  acceptable: {accepted}")
            for g in got[:3]:
                print(f"    got: {g[0]} 問{g[1]}  {g[2][:50]}")


if __name__ == "__main__":
    main()
