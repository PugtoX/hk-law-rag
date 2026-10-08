#!/usr/bin/env python3
"""
Step 4: measure retrieval quality — recall@k over the indexed Q&A pairs.

Two yardsticks (both print, so the gap is visible):

  --by-question  (default, strict lower bound)
      gold = the stored question text; hit only if that exact question comes back
      in the top-k. Punishes paraphrases: the corpus often holds the SAME rule
      phrased differently, which is a correct retrieval but scores as a miss.

  --by-answer    (looser, closer to user intent)
      gold = the answer text of the query's own pair; hit if the retrieved
      documents include that answer. Same rule worded differently -> still a hit.

Neither is a real-user benchmark: the queries are the corpus's own questions, so
both numbers run optimistic. They are for regression testing — rerun after
changing chunking / embedding / top-k to see whether quality moved.

Usage:  python eval_retrieval.py                 # by question, top-5
        python eval_retrieval.py --by-answer     # by answer, top-5
        python eval_retrieval.py 10              # by question, top-10
"""
import sys
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent
INDEX = ROOT / "chroma"
COLLECTION = "cap57"
EMBED_MODEL = "BAAI/bge-m3"


def parse_args(argv: list[str]) -> tuple[str, int]:
    mode, k = "question", 5
    for a in argv[1:]:
        if a == "--by-answer":
            mode = "answer"
        elif a == "--by-question":
            mode = "question"
        elif a.isdigit():
            k = int(a)
        else:
            sys.exit(f"[!] unknown argument: {a}")
    return mode, k


def main() -> None:
    mode, k = parse_args(sys.argv)

    model = SentenceTransformer(EMBED_MODEL)
    col = chromadb.PersistentClient(path=str(INDEX)).get_collection(COLLECTION)
    got = col.get(include=["metadatas", "documents"])
    questions = [m["question"] for m in got["metadatas"]]
    gold_answers = got["documents"]           # the answer stored for each question
    print(f"[i] {len(questions)} pairs, judging by {mode}, top-{k}\n")

    embs = model.encode(questions, batch_size=8, show_progress_bar=True,
                        normalize_embeddings=True)
    res = col.query(query_embeddings=embs.tolist(), n_results=min(k, col.count()))

    hits, misses = 0, []
    for i, gold in enumerate(questions):
        got_questions = [m["question"] for m in res["metadatas"][i]]
        if mode == "question":
            hit = gold in got_questions
        else:
            hit = gold_answers[i] in res["documents"][i]
        if hit:
            hits += 1
        else:
            misses.append((i, gold, got_questions))

    print(f"\nrecall@{k} (by {mode}) = {hits}/{len(questions)} = {hits / len(questions):.1%}")
    if misses:
        print(f"\n--- {len(misses)} misses ---")
        for i, gold, retrieved in misses[:5]:
            print(f"\nMISS: {gold}")
            if mode == "answer":
                # the gold answer is what we required to be retrieved: print its
                # opening so it can be compared against what actually came back
                print(f"  gold answer: {gold_answers[i][:110]}")
            for r in retrieved[:3]:
                print(f"   got: {r}")


if __name__ == "__main__":
    main()
