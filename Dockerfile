# Container for hk-law-rag — runs the FastAPI app + the single-page UI.
# Built for Hugging Face Spaces (Docker SDK): it listens on 7860.
#
#   docker build -t hk-law-rag .
#   docker run -p 7860:7860 -e LLM_PROVIDER=openai -e OPENAI_KEY=... hk-law-rag
#
# The corpus and the vector index are built INTO the image, so a cold start does
# not re-download anything. bge-m3 (~2GB) is baked into the layer as well.
FROM python:3.12-slim

RUN useradd -m -u 1000 app
WORKDIR /app
ENV HF_HOME=/home/app/.cache/huggingface \
    PYTHONUNBUFFERED=1 \
    TOKENIZERS_PARALLELISM=false

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=app:app . .
USER app

# Fetch the 16 FAQ pages and build the 84-pair index at build time.
RUN python download_data.py && python build_index.py

EXPOSE 7860
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
