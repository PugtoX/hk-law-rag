#!/usr/bin/env python3
"""
Step 3: search the Cap.57 FAQ index. Retrieval only — no LLM yet.

Goal this week: prove the right Q&A pair comes back before a model speaks.

Usage:  python search.py "病假有薪嗎"
        python search.py "試用期" "拖欠工資" "年假"     # each arg = one query
        python search.py                                # interactive loop
"""
import sys
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent
INDEX = ROOT / "chroma"
COLLECTION = "cap57"
EMBED_MODEL = "BAAI/bge-m3"
TOP_K = 5


def main() -> None:
    model = SentenceTransformer(EMBED_MODEL)
    col = chromadb.PersistentClient(path=str(INDEX)).get_collection(COLLECTION)
    print(f"[i] {col.count()} Q&A pairs indexed\n")

    # argv queries first; once they run out we keep prompting (None = prompt)
    pending = sys.argv[1:] if len(sys.argv) > 1 else [None]

    while pending:
        q = pending.pop(0)
        if q is None:
            try:
                q = input("query> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not q:
                break
        else:
            q = q.strip()
            if not q:
                continue

        emb = model.encode([q], normalize_embeddings=True).tolist()
        res = col.query(query_embeddings=emb, n_results=min(TOP_K, col.count()))
        print(f"=== {q} ===")
        for rank, (doc, meta, dist) in enumerate(
            zip(res["documents"][0], res["metadatas"][0], res["distances"][0]), 1
        ):
            print(f"[{rank}] dist={dist:.4f}  [{meta['topic']}] {meta['question']}")
            print(f"     {doc[:260]}")
        print()

        if not pending:          # argv exhausted -> switch to interactive
            pending.append(None)


if __name__ == "__main__":
    main()
