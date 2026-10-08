#!/usr/bin/env bash
# One-shot setup + run. From the project root:
#     bash run.sh "休息日有薪水嗎？"      # ask one question
#     bash run.sh                         # interactive CLI
#     bash run.sh web                     # serve the web UI on :8000
#     PORT=9000 bash run.sh web
set -euo pipefail
cd "$(dirname "$0")"

MODE="${1:-cli}"
[ "$MODE" = "web" ] && shift || true

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

# 3) web mode needs the API deps but no Ollama check here
if [ "$MODE" = "web" ]; then
  PORT="${PORT:-8000}"
  echo "[run] serving on http://localhost:$PORT  (Ctrl-C to stop)"
  exec uvicorn app:app --host 0.0.0.0 --port "$PORT"
fi

# 4) CLI mode: make sure the local generation model is present, then ask
ollama list 2>/dev/null | grep -q "${MODEL%%:*}" \
  || { echo "[setup] pulling $MODEL"; ollama pull "$MODEL"; }

exec python answer.py --model "$MODEL" "$@"
