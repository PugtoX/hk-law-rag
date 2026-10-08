#!/usr/bin/env python3
"""
Render demos/demo.png — a terminal-style snapshot of real output, for the README.

Run from the project root:   python demos/make_demo_image.py

Font note: CJK text must use a CJK font (Microsoft YaHei / JhengHei). Consolas
has no Chinese glyphs, so only commands and ASCII debug lines use it. The text
below is transcribed from actual runs; update it by hand if answers change.
"""
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "demos" / "demo.png"

W, PAD, LH = 1180, 26, 25
BG, FG, DIM = (13, 17, 23), (230, 237, 243), (139, 148, 158)
BLUE, GREEN, RED, YELLOW = (88, 166, 255), (63, 185, 80), (248, 81, 73), (210, 153, 34)

MONO = [r"C:\Windows\Fonts\consola.ttf",
        "/mnt/c/Windows/Fonts/consola.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"]
CJK = [r"C:\Windows\Fonts\msyh.ttc",      # Microsoft YaHei (SC + TC glyphs)
       "/mnt/c/Windows/Fonts/msyh.ttc",
       r"C:\Windows\Fonts\msjh.ttc",      # Microsoft JhengHei (TC)
       "/mnt/c/Windows/Fonts/msjh.ttc"]


def load(cands, size):
    for c in cands:
        try:
            f = ImageFont.truetype(c, size)
            print(f"[font] {size}px -> {Path(c).name} {f.getname()}")
            return f
        except Exception as e:
            print(f"[font] FAIL {c}: {type(e).__name__}")
    raise SystemExit(f"[!] no usable font for size {size}; candidates={cands}")


F_TITLE = load(CJK, 24)
F_CJK = load(CJK, 15)
F_CJK_SM = load(CJK, 13)
F_MONO = load(MONO, 14)

BLANK = object()
lines: list[tuple[str, object, object]] = []


def add(style, text, font=F_CJK):
    lines.append((style, text, font))


def blank():
    lines.append(("blank", "", None))


# --- content (transcribed from real runs) ---
add("title", "hk-law-rag  —  香港僱傭條例問答助手", F_TITLE)
add("dim", "RAG: 84 組問答對 · bge-m3 + Chroma · qwen3.5 (Ollama)", F_CJK_SM)
add("dim", "recall@5 = 100% 回歸測試 (84/84)   |   93.3% 口語問法 (14/15)", F_CJK_SM)
blank()

add("mono", '$ python answer.py --no-rag "休息日有薪水嗎？"          [ 裸模型 ]', F_CJK)
add("red", "這取決於您的休假類型……在台灣勞工法規（勞動基準法）的架構下：")
add("red", "特別休假（特休）—— 這天是有薪水的。")
add("dim", "   → 引用了台灣《勞基法》，對香港完全錯誤", F_CJK_SM)
blank()

add("mono", '$ python answer.py "休息日有薪水嗎？"         [ 經過檢索 ]', F_CJK)
add("green", "休息日是否有薪，由僱傭合約或勞資雙方協議決定，法例並沒有規定必須有薪 [1]、[2]。")
add("dim", "   [1] smw_wage_items.htm 問3     [2] cap57e.htm 問3", F_CJK_SM)
blank()

add("mono", '$ python answer.py "工作日加班有加班費嗎？"', F_CJK)
add("yellow", "資料庫中沒有關於「工作日」加班費的具體規定，只提及休息日和法定假日的相關安排。")
add("dim", "   → 拒答而非編造：說明「看到什麼、沒看到什麼」", F_CJK_SM)
blank()

add("mono", '$ python answer.py --debug "今天天氣怎麼樣？"', F_CJK)
add("mono", "   [debug] best distance = 1.2090 (gate = 0.95)", F_MONO)
add("red", "資料庫中沒有相關規定（檢索相似度不足，未呼叫模型）。")
add("dim", "   → 代碼層硬門禁：模型根本沒有被呼叫", F_CJK_SM)

# --- render ---
height = 52 + sum(12 if s == "blank" else LH for s, _, _ in lines) + PAD
img = Image.new("RGB", (W, height), BG)
d = ImageDraw.Draw(img)

d.rectangle([0, 0, W, 34], fill=(22, 27, 34))
d.ellipse([14, 12, 26, 24], fill=RED)
d.ellipse([34, 12, 46, 24], fill=YELLOW)
d.ellipse([54, 12, 66, 24], fill=GREEN)
d.text((W // 2 - 70, 10), "hk-law-rag — demo", font=F_MONO, fill=DIM)

COLORS = {"title": FG, "dim": DIM, "green": GREEN, "red": RED,
          "yellow": YELLOW, "mono": BLUE}

y = 56
for style, text, font in lines:
    if style == "blank":
        y += 12
        continue
    d.text((PAD, y), text, font=font, fill=COLORS.get(style, FG))
    y += LH

OUT.parent.mkdir(parents=True, exist_ok=True)
img.save(OUT)
print(f"wrote {OUT}  ({img.width}x{img.height})")
