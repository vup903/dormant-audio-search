"""FastAPI app: natural-language search over the indexed corpus.

Serves the one-page UI, the search API, the eval report, and the original
audio files (StaticFiles handles HTTP range requests, which is what lets the
player seek straight to a result's timecode).
"""
from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from das.retrieval import Searcher

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "web"
EVAL_RESULTS = ROOT / "eval" / "results.json"

app = FastAPI(title="dormant-audio-search")
searcher: Searcher | None = None


class Query(BaseModel):
    query: str


@app.on_event("startup")
def load_searcher() -> None:
    global searcher
    searcher = Searcher()


@app.get("/")
def home():
    return FileResponse(WEB / "index.html")


@app.get("/eval")
def eval_page():
    return FileResponse(WEB / "eval.html")


@app.post("/api/search")
def search(q: Query):
    assert searcher is not None
    return {"query": q.query, "results": searcher.search(q.query)}


@app.get("/api/eval")
def eval_results():
    if not EVAL_RESULTS.exists():
        return JSONResponse({"error": "no eval results — run: python -m pytest eval/"}, status_code=404)
    return json.loads(EVAL_RESULTS.read_text(encoding="utf-8"))


app.mount("/audio", StaticFiles(directory=ROOT / "data" / "audio"), name="audio")
