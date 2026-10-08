"""A/B test: does embedding 'question + answer' beat embedding 'answer' alone?

Short answers like '這條款由僱主和僱員雙方協定。' carry almost no signal for
bge-m3, so the pair is unretrievable. Prepending the question should fix it.
Runs on the sample pages (no network, no real index touched)."""
import importlib.util
import sys
import tempfile
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

# locate the project root from this file (this script lives in samples/), so it
# works on Linux/WSL as well as Windows
SAMPLES = Path(__file__).resolve().parent
ROOT = SAMPLES.parent
if not (ROOT / "build_index.py").exists():
    sys.exit(f"[!] build_index.py not found next to {SAMPLES}.\n"
             f"    This script must live in <project>/samples/.\n"
             f"    Expected: {ROOT / 'build_index.py'}")

spec = importlib.util.spec_from_file_location("bi", ROOT / "build_index.py")
bi = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bi)

# collect pairs: prefer the full dataset in data/, fall back to samples/
src = ROOT / "data"
pages = sorted(p for p in src.glob("*.htm*") if p.is_file())
if not pages:
    src = SAMPLES
    pages = sorted(p for p in src.glob("*.htm") if p.is_file())
items = []
for p in pages:
    for q, a in bi.parse_pairs(bi.read_any(p)):
        items.append((q, a))
print(f"source: {src}")
print(f"pairs available: {len(items)}")
print("shortest answers:")
for q, a in sorted(items, key=lambda x: len(x[1]))[:5]:
    print(f"  len={len(a):<4} Q={q[:34]!r} A={a[:34]!r}")

model = SentenceTransformer("BAAI/bge-m3")
questions = [q for q, _ in items]
answers = [a for _, a in items]
if not items:
    sys.exit("[!] no pairs parsed from samples/*.htm")

# how big is the short-answer problem?
short = [a for a in answers if len(a) < 40]
print(f"answers shorter than 40 chars: {len(short)}/{len(answers)}")

with tempfile.TemporaryDirectory() as tmp:
    client = chromadb.PersistentClient(path=tmp)

    def build(name, docs):
        col = client.create_collection(name)
        embs = model.encode(docs, batch_size=8, normalize_embeddings=True)
        col.add(ids=[f"i{i}" for i in range(len(docs))],
                embeddings=embs.tolist(), documents=docs,
                metadatas=[{"question": q} for q in questions])
        return col

    col_a = build("answers_only", answers)
    col_q = build("with_question", [f"{q}\n{a}" for q, a in items])

    q_embs = model.encode(questions, batch_size=8, normalize_embeddings=True)
    k = 5
    for name, col in (("answer only", col_a), ("question+answer", col_q)):
        res = col.query(query_embeddings=q_embs.tolist(), n_results=k)
        hits = sum(1 for i, q in enumerate(questions)
                   if q in [m["question"] for m in res["metadatas"][i]])
        print(f"{name:<16} recall@{k} = {hits}/{len(questions)} = {hits / len(questions):.1%}")

    # which pairs does 'answer only' miss that 'question+answer' gets?
    res_a = col_a.query(query_embeddings=q_embs.tolist(), n_results=k)
    res_q = col_q.query(query_embeddings=q_embs.tolist(), n_results=k)
    print("\nrecovered by prepending the question:")
    recovered = 0
    for i, q in enumerate(questions):
        in_a = q in [m["question"] for m in res_a["metadatas"][i]]
        in_q = q in [m["question"] for m in res_q["metadatas"][i]]
        if in_q and not in_a:
            recovered += 1
            print(f"  RECOVERED: {q[:52]}")
            print(f"             A={answers[i][:52]!r}")
    if not recovered:
        print("  (none)")

    # the three pairs that missed in the real 84-pair index: can either strategy
    # retrieve them, and at what rank?
    print("\n--- the three real-index misses, probed directly ---")
    probes = [
        "佣金、勤工獎、交通津貼是否屬於工資的一部份？",
        "休息日是否規定有薪？",
        "法定假日是否規定有薪抑或無薪？",
    ]
    for text in probes:
        exact = [i for i, q in enumerate(questions) if q.strip() == text.strip()]
        if not exact:
            print(f"\n[not in this dataset] {text}")
            continue
        i = exact[0]
        short = len(answers[i]) < 40
        for name, col, res in (("answer only", col_a, res_a),
                               ("question+answer", col_q, res_q)):
            rank = next((r + 1 for r, m in enumerate(res["metadatas"][i])
                         if m["question"] == questions[i]), None)
            print(f"{name:<16} rank={rank if rank else 'MISS':<5} "
                  f"ans_len={len(answers[i])} short={short}  {text[:34]}")
