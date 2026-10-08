#!/usr/bin/env bash
# One-shot setup + run. From the project root:
#     bash run.sh "休息日有薪水嗎？"
#     bash run.sh                         # interactive
set -euo pipefail
cd "$(dirname "$0")"

MODEL="${RAG_MODEL:-qwen3.5:9b}"

# 1) venv + deps
if [ ! -d .venv ]; then
  echo "[setup] creating venv"
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
python -c "import chromadb, sentence_transformers" 2>/dev/null \
  || { echo "[setup] installing requirements"; pip install -q -U pip; pip install -q -r requirements.txt; }

# 2) data + index
[ -d data ] && [ "$(ls -1 data/*.htm* 2>/dev/null | wc -l)" -gt 0 ] \
  || { echo "[setup] downloading corpus"; python download_data.py; }

if [ ! -d chroma ]; then
  echo "[setup] building index (first run also downloads bge-m3, ~2GB)"
  python build_index.py
fi

# 3) generation model
ollama list 2>/dev/null | grep -q "${MODEL%%:*}" \
  || { echo "[setup] pulling $MODEL"; ollama pull "$MODEL"; }

# 4) answer
exec python answer.py --model "$MODEL" "$@"
