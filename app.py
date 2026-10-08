#!/usr/bin/env python3
"""
Web API for hk-law-rag: retrieval + cited generation, plus a single-page UI.

Local:   uvicorn app:app --reload --port 8000
Deploy:  see README (Hugging Face Spaces / Docker)

The generation backend is chosen by environment, so the same code runs locally
on Ollama and on a free host with an OpenAI-compatible API (e.g. Groq):

    LLM_PROVIDER=ollama   (default)   -> local Ollama
    LLM_PROVIDER=openai               -> OPENAI_BASE + OPENAI_KEY + OPENAI_MODEL

If generation is unavailable the API still returns retrieval results, so the
page degrades to search-only instead of failing.
"""
import json
import os
import urllib.request
from pathlib import Path

import chromadb
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

import answer as rag

WEB = Path(__file__).resolve().parent / "web"

PROVIDER = os.environ.get("LLM_PROVIDER", "ollama").lower()
LOCAL_MODEL = os.environ.get("OLLAMA_MODEL", rag.DEFAULT_MODEL)
OPENAI_BASE = os.environ.get("OPENAI_BASE", "https://api.groq.com/openai/v1").rstrip("/")
OPENAI_KEY = os.environ.get("OPENAI_KEY", "")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "llama-3.3-70b-versatile")

app = FastAPI(title="hk-law-rag", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

_embedder = None
_col = None


def resources():
    """Load the embedding model and index once, on first use."""
    global _embedder, _col
    if _embedder is None:
        _embedder = SentenceTransformer(rag.EMBED_MODEL)
        _col = chromadb.PersistentClient(path=str(rag.INDEX)).get_collection(rag.COLLECTION)
    return _embedder, _col


class Ask(BaseModel):
    question: str
    top_k: int = 5
    generate: bool = True


def generate(prompt: str) -> str:
    """The single place that decides where generated text comes from."""
    if PROVIDER == "openai":
        if not OPENAI_KEY:
            raise RuntimeError("LLM_PROVIDER=openai but OPENAI_KEY is not set")
        body = json.dumps({
            "model": OPENAI_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
        }).encode()
        req = urllib.request.Request(
            f"{OPENAI_BASE}/chat/completions", data=body,
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {OPENAI_KEY}"})
        with urllib.request.urlopen(req, timeout=60) as r:
            data = json.loads(r.read())
        return data["choices"][0]["message"]["content"].strip()
    return rag.call_ollama(LOCAL_MODEL, prompt)


@app.get("/api/health")
def health():
    _, col = resources()
    return {
        "status": "ok",
        "pairs": col.count(),
        "embed_model": rag.EMBED_MODEL,
        "provider": PROVIDER,
        "gen_model": OPENAI_MODEL if PROVIDER == "openai" else LOCAL_MODEL,
        "gate": rag.MAX_DIST,
        "generation_ready": PROVIDER != "openai" or bool(OPENAI_KEY),
    }


@app.post("/api/ask")
def ask(req: Ask):
    question = req.question.strip()
    if not question:
        return {"error": "empty question"}

    embedder, col = resources()
    top_k = max(1, min(req.top_k, 10))
    hits, dists = rag.retrieve(embedder, col, question, top_k)
    best = min(dists) if dists else 99.0

    out = {
        "question": question,
        "best_distance": round(best, 4),
        "gate": rag.MAX_DIST,
        "hits": [dict(h, distance=round(d, 4)) for h, d in zip(hits, dists)],
        "answer": None,
        "refused": False,
    }

    # guard 1: the distance gate, decided in code — no model call
    if best > rag.MAX_DIST:
        out["refused"] = True
        out["reason"] = "檢索相似度不足，未呼叫模型"
        return out

    if not req.generate:
        return out

    try:
        out["answer"] = generate(rag.build_prompt(question, hits))
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"
    return out


@app.get("/")
def index():
    return FileResponse(WEB / "index.html")
