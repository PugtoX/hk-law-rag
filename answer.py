#!/usr/bin/env python3
"""
Step 5: generation layer — turn retrieved articles into a cited answer.

Two independent guards against hallucination (a wrong answer here is a wrong
legal answer, so this cannot be left to the model's good behaviour):

  1. Distance gate (hard, in code): if even the best retrieved pair is further
     than MAX_DIST, refuse without calling the model at all.
  2. Prompt constraint (soft): the model is told to answer ONLY from the supplied
     articles and to cite [1][2]; if they do not cover the question it must say so.

Retrieved articles are numbered in the prompt so the answer can cite them.

Usage:  python answer.py "休息日有薪水嗎？"
        python answer.py                      # interactive
        python answer.py --model qwen2.5:3b "年假有幾天？"
        python answer.py --top-k 3 "試用期被辭退有賠償嗎？"
        python answer.py --debug "..."        # print distances and the raw prompt
"""
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent
INDEX = ROOT / "chroma"
COLLECTION = "cap57"
EMBED_MODEL = "BAAI/bge-m3"

OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "qwen3.5:9b"       # check with: ollama list
TIMEOUT = 300
# If the closest pair is further than this, treat the corpus as not covering the
# question and refuse WITHOUT calling the model. Tuned from observed distances:
# a well-covered question sits at ~0.54; a related-but-uncovered one at ~0.70.
# So 0.95 blocks only queries that have essentially nothing to do with the corpus.
# Run `--debug` on an unrelated question to check where the gap actually is.
MAX_DIST = 0.95

SYSTEM_PROMPT = (
    "你是香港《僱傭條例》的問答助手。規則：\n"
    "1. 只可根據「參考條文」回答，不可使用你自己的知識。\n"
    "2. 每個結論後面用 [1]、[2] 標明依據的條文編號。\n"
    "3. 若參考條文沒有涵蓋問題，直接回答「資料庫中沒有相關規定」，不要猜測。\n"
    "4. 用繁體中文回答，簡潔、口語，不要照抄整段條文。\n"
    "5. 若條文提到條件（例如受僱年期），必須一併說明。"
)


def build_prompt(question: str, hits: list[dict]) -> str:
    blocks = []
    for i, h in enumerate(hits, 1):
        blocks.append(f"[{i}] （{h['topic']}）問：{h['question']}\n答：{h['answer']}")
    articles = "\n\n".join(blocks)
    return (f"{SYSTEM_PROMPT}\n\n"
            f"參考條文：\n{articles}\n\n"
            f"問題：{question}\n\n回答：")


def call_ollama(model: str, prompt: str) -> str:
    # think=false is required for reasoning models (qwen3.5): with thinking on,
    # the answer can land in the `thinking` field and leave `response` empty, and
    # the model is also more prone to answering from memory (which produced a
    # mainland-China labour answer here). Harmless for models that don't reason.
    payload = json.dumps({"model": model, "prompt": prompt,
                          "stream": False, "think": False}).encode()
    req = urllib.request.Request(OLLAMA_URL, data=payload,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        return json.loads(r.read())["response"].strip()


def parse_args(argv: list[str]) -> tuple[str, int, bool, bool, list[str]]:
    model, top_k, debug, no_rag, queries = DEFAULT_MODEL, 5, False, False, []
    i = 1
    while i < len(argv):
        a = argv[i]
        if a == "--model" and i + 1 < len(argv):
            model = argv[i + 1]; i += 2
        elif a == "--top-k" and i + 1 < len(argv):
            top_k = int(argv[i + 1]); i += 2
        elif a == "--debug":
            debug = True; i += 1
        elif a == "--no-rag":
            no_rag = True; i += 1
        else:
            queries.append(a); i += 1
    return model, top_k, debug, no_rag, queries


def main() -> None:
    model, top_k, debug, no_rag, queries = parse_args(sys.argv)

    embedder = SentenceTransformer(EMBED_MODEL)
    col = chromadb.PersistentClient(path=str(INDEX)).get_collection(COLLECTION)
    print(f"[i] {col.count()} pairs indexed | model={model} | top_k={top_k}"
          + ("  [NO-RAG baseline]" if no_rag else "") + "\n")

    # None in the queue means "prompt for the next question"; an empty input ends
    # the loop. Queries from argv are consumed first, then it goes interactive.
    pending: list[str | None] = list(queries) if queries else [None]
    while pending:
        q = pending.pop(0)
        if q is None:
            try:
                q = input("問題> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not q:
                break
        else:
            q = q.strip()
            if not q:
                continue
        pending.append(None)      # after every answer, go back to prompting

        emb = embedder.encode([q], normalize_embeddings=True).tolist()
        res = col.query(query_embeddings=emb, n_results=min(top_k, col.count()))
        hits, dists = [], res["distances"][0]
        for doc, meta in zip(res["documents"][0], res["metadatas"][0]):
            hits.append({"topic": meta["topic"], "question": meta["question"],
                         "answer": doc, "source": meta["source"], "qno": meta["qno"]})

        best = min(dists) if dists else 99.0
        if debug:
            print(f"[debug] best distance = {best:.4f} (gate = {MAX_DIST})")
            for h, d in zip(hits, dists):
                print(f"        {d:.4f}  [{h['topic']}] {h['question']}")

        # --- no-RAG baseline, for demonstrating why retrieval matters ---
        # The model answers from memory only. This is where jurisdiction
        # confusion shows up (e.g. it answers a Hong Kong question with
        # mainland-China labour law).
        if no_rag:
            print(f"\n=== {q} ===")
            try:
                bare = call_ollama(model, f"請用繁體中文回答：{q}")
            except Exception as e:
                print(f"[!] 生成失敗：{type(e).__name__}: {e}")
                break
            print(bare or "[!] 空回答")
            continue

        # --- guard 1: distance gate, decided in code, not by the model ---
        if best > MAX_DIST:
            print(f"\n=== {q} ===")
            print("資料庫中沒有相關規定（檢索相似度不足，未呼叫模型）。")
            print(f"[nearest was {best:.3f}] 試試換個問法，或補充相關資料。\n")
            continue

        prompt = build_prompt(q, hits)
        if debug:
            print(f"[debug] prompt chars = {len(prompt)}")

        print(f"\n=== {q} ===")
        try:
            answer = call_ollama(model, prompt)
        except urllib.error.HTTPError as e:
            body = ""
            try:
                body = e.read().decode("utf-8", "replace")[:300]
            except Exception:
                pass
            print(f"[!] Ollama 拒絕了請求：HTTP {e.code}")
            print(f"    回應：{body}")
            if e.code == 404:
                print(f"    多半是模型名不存在。先執行：ollama list")
                print(f"    若列表裡沒有 {model}，執行：ollama pull {model}")
            break
        except urllib.error.URLError as e:
            print(f"[!] 連不上 Ollama：{e.reason}")
            print("    確認服務在跑：systemctl status ollama")
            break
        except Exception as e:
            print(f"[!] 生成失敗：{type(e).__name__}: {e}")
            break
        if not answer:
            print("[!] 模型返回了空回答（可能是推理模型把內容放進了 thinking，"
                  "或達到長度上限）。試試 --model 換一個模型。")
        else:
            print(answer)

        if debug:
            print("\n--- 引用到的條文 ---")
            for i, h in enumerate(hits, 1):
                print(f"  [{i}] {h['source']} 問{h['qno']}  {h['question'][:46]}")


if __name__ == "__main__":
    main()
