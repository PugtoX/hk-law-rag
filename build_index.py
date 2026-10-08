#!/usr/bin/env python3
"""
Step 2: build the retrieval index from the Labour Department Cap.57 FAQ pages.

Pipeline: .htm -> split into Q&A pairs -> embed the ANSWER with bge-m3 -> Chroma.
The question text is kept as metadata, so search can match both question and answer.

Data source: https://www.labour.gov.hk/tc/faq/cap57X_whole.htm  (HK gov, public)
Verified structure: each page holds N pairs marked by <a name="Qn"> 問n. <answer> 答n.
"""
import re
import sys
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
INDEX = ROOT / "chroma"
COLLECTION = "cap57"
EMBED_MODEL = "BAAI/bge-m3"

# Fallback topic names, keyed by filename stem. Used when a page's second <h2>
# is missing (e.g. the two Minimum Wage pages, whose heading is the chapter name).
FALLBACK_TOPIC = {
    "cap57a": "受《僱傭條例》保障的僱員", "cap57b": "僱傭合約", "cap57c": "工資",
    "cap57d": "終止僱傭合約", "cap57e": "休息日", "cap57f": "法定假日",
    "cap57g": "病假", "cap57h": "產假", "cap57i": "年假", "cap57j": "年終酬金",
    "cap57k": "僱傭保障", "cap57l": "遣散費及長期服務金",
    "cap57m": "強制性公積金計劃與僱員福利", "cap57n": "侍產假",
    "smw_coverage": "最低工資：適用範圍",
    "smw_wage_items": "最低工資：工資項目",
}


def extract_topic(html: str, stem: str) -> str:
    """The page's 2nd <h2> is the section title (h2[0] is the chapter name)."""
    heads = [to_text(h) for h in re.findall(r"<h2[^>]*>(.*?)</h2>", html, re.S | re.I)]
    heads = [h for h in heads if h]
    if len(heads) >= 2:
        return heads[1]
    return FALLBACK_TOPIC.get(stem, stem)


def read_any(p: Path) -> str:
    raw = p.read_bytes()
    for enc in ("utf-8-sig", "utf-8", "utf-16", "big5", "gb18030"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def to_text(s: str) -> str:
    s = re.sub(r"(?is)<(script|style).*?</\1>", " ", s)
    s = re.sub(r"</(p|div|li|tr|h\d)>", "\n", s, flags=re.I)
    s = re.sub(r"<br\s*/?>", "\n", s, flags=re.I)
    s = re.sub(r"<[^>]+>", "", s)
    s = (s.replace("&nbsp;", " ").replace("&amp;", "&")
          .replace("&lt;", "<").replace("&gt;", ">").replace("&quot;", '"')
          .replace("&#39;", "'"))
    s = re.sub(r"[ \t\u3000]+", " ", s)
    s = re.sub(r"\n\s*\n+", "\n", s)
    s = re.sub(r"\s*回到問題\s*$", "", s)   # strip the page's back-link
    return s.strip()


# A row/cell is a pure marker when it holds only '問 n .' / '答 n :' — in that
# case the real text lives in the adjacent cell (markers and text swap cells
# between pages, and the two Minimum Wage pages use '問 1 :' instead of '問1.').
# Some markers carry a roman suffix ('答2(I)'), so the suffix is optional.
MARK_RE = re.compile(r"^[問答]\s*\d+\s*(?:\([IVXivx]+\))?\s*[.、:：]?$")
Q_RE = re.compile(r"問\s*(\d+)\s*(?:\([IVXivx]+\))?\s*[.、:：]?")
A_RE = re.compile(r"答\s*(\d+)\s*(?:\([IVXivx]+\))?\s*[.、:：]?")


def row_text(cells: list[str], mark_re) -> str:
    """The row's content text, with its own marker stripped."""
    head = " ".join(c for c in cells if c and not MARK_RE.match(c.strip()))
    return mark_re.sub("", head).strip()


def parse_pairs(html: str) -> list[tuple[str, str]]:
    """Return [(question, answer)] pairing each 答n row with the nearest
    preceding 問n row.

    Why answer-anchored: on several pages the 問1..問n sequence appears TWICE
    (a table of contents near the top, then the real Q&A), so any "remember the
    last question" logic gets clobbered by the TOC repeat (this cost cap57k one
    pair). Walking from the 答 row upwards to the matching 問 row is robust to
    that, and also handles 問7 sitting 30 rows above 答7.

    Layouts verified against real pages:
      * '<td>問1.</td><td>甚麼是僱傭合約？</td>'   -> marker and text in separate cells
      * '<td>問1. 甚麼是僱傭合約？</td>'           -> marker and text in one cell
      * '<td>問 1 :</td><td>法定最低工資…</td>'    -> colon form (Cap.608)
      * question and answer in separate <tr>s      -> matched by number
    """
    rows = []
    for r in re.findall(r"<tr[^>]*>(.*?)</tr>", html, re.S | re.I):
        cells = [to_text(c) for c in
                 re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", r, re.S | re.I)]
        cells = [c.strip() for c in cells if c and c.strip()]
        if cells:
            rows.append(cells)

    # keyed by question number so a split answer (答2(I) + 答2(II)) merges into
    # one pair instead of producing a duplicate question
    found: dict[str, tuple[str, list[str]]] = {}
    for i, cells in enumerate(rows):
        joined = " ".join(cells)
        am = A_RE.search(joined)
        if not am:
            continue
        qno = am.group(1)
        a_text = row_text(cells, A_RE)
        if not a_text:
            continue
        for j in range(i - 1, max(-1, i - 60), -1):
            qm = Q_RE.search(" ".join(rows[j]))
            if qm and qm.group(1) == qno:
                q_text = row_text(rows[j], Q_RE)
                if q_text:
                    if qno in found:
                        found[qno][1].append(a_text)
                    else:
                        found[qno] = (q_text, [a_text])
                break

    return [(q, "\n".join(parts)) for q, parts in found.values()]


def collect() -> list[dict]:
    files = sorted(p for p in DATA.glob("*.htm*") if p.is_file())
    files += sorted(p for p in DATA.glob("*.txt") if p.is_file())
    if not files:
        sys.exit(f"[!] no data files under {DATA} — run download_data.py first")

    items, bad = [], []
    for p in files:
        html = read_any(p)
        pairs = parse_pairs(html)
        if not pairs:
            bad.append(p.name)
            continue
        topic = extract_topic(html, p.stem)
        for n, (q, a) in enumerate(pairs, 1):
            items.append({"topic": topic, "question": q, "answer": a,
                          "source": p.name, "qno": n})
        print(f"[i] {p.name:<20} topic={topic:<16} pairs={len(pairs)}")
    if bad:
        print(f"[!] parsed 0 pairs from: {', '.join(bad)}")
    if not items:
        sys.exit("[!] 0 Q&A pairs parsed — data format changed, re-probe the file")
    return items


def main() -> None:
    # Answers are sometimes a single short sentence ("是。" / "這條款由僱主和僱員
    # 雙方協定。"). Embedded alone they carry almost no signal, so the pair becomes
    # unretrievable — measured on this corpus: 81/84 (answer only) vs 84/84
    # (question + answer). Default is therefore question + answer; the documents
    # stored stay the plain answer, so search output and evaluation are unchanged.
    # Pass --answer-only for the old behaviour.
    answer_only = "--answer-only" in sys.argv

    items = collect()
    print(f"[i] total {len(items)} Q&A pairs")
    dup = len(items) - len({it["question"] for it in items})
    if dup:
        print(f"[i] note: {dup} duplicate questions (same topic repeats across pages)")
    short = [it for it in items if len(it["answer"]) < 40]
    print(f"[i] {len(short)} answers shorter than 40 chars "
          f"(these are the retrieval-weak pairs)")

    print(f"[i] loading {EMBED_MODEL} (first run downloads ~2GB, be patient)")
    model = SentenceTransformer(EMBED_MODEL)
    answers = [it["answer"] for it in items]
    to_embed = (answers if answer_only
                else [f'{it["question"]}\n{it["answer"]}' for it in items])
    print(f"[i] embedding strategy: {'answer only' if answer_only else 'question + answer'}")
    embs = model.encode(to_embed, batch_size=8, show_progress_bar=True,
                        normalize_embeddings=True)

    client = chromadb.PersistentClient(path=str(INDEX))
    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass
    col = client.create_collection(COLLECTION)
    col.add(
        ids=[f"qa-{i}" for i in range(len(items))],
        embeddings=embs.tolist(),
        documents=answers,          # documents stay the plain answer text
        metadatas=[{"topic": it["topic"], "question": it["question"],
                    "source": it["source"], "qno": it["qno"]} for it in items],
    )
    print(f"[OK] indexed {col.count()} Q&A pairs -> {INDEX}")
    print('[next] python eval_retrieval.py --by-answer   (expect 84/84)')


if __name__ == "__main__":
    main()
