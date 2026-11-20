"""FastAPI RESTful service for Perpres Hybrid Retrieval + QDA Reranking."""

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from engine import PerpresSearchEngine

STATIC_DIR = Path(__file__).resolve().parent / "static"
_engine: PerpresSearchEngine | None = None


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global _engine
    _engine = PerpresSearchEngine()
    yield
    _engine = None


app = FastAPI(
    title="Perpres Hybrid Retrieval & QDA API",
    description="Two-stage Indonesian Presidential Regulation search API (BM25 + Multilingual E5 + RRF + QDA).",
    version="1.0.0",
    lifespan=lifespan,
)


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Search query in Indonesian.")
    top_k: int = Field(10, ge=1, le=20, description="Number of results to return (1-20).")
    mode: Literal["hybrid_qda", "rrf"] = Field(
        "hybrid_qda", description="Ranking mode: 'hybrid_qda' (Stage 1 + QDA) or 'rrf' (Stage 1 only)."
    )


def _get_engine() -> PerpresSearchEngine:
    if _engine is None:
        raise HTTPException(status_code=503, detail="Search engine is not initialized.")
    return _engine


@app.get("/", include_in_schema=False)
def root() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health() -> dict:
    eng = _get_engine()
    return {
        "status": "ok",
        "device": str(eng.device),
        "model_name": eng.model_name,
        "num_documents": len(eng.doc_ids),
        "qda_loaded": eng.qda_loaded,
        "rerank_alpha": eng.rerank_alpha,
    }


@app.get("/search")
def search_get(
    q: str = Query(..., min_length=1, description="Search query in Indonesian."),
    top_k: int = Query(10, ge=1, le=20, description="Number of results to return (1-20)."),
    mode: Literal["hybrid_qda", "rrf"] = Query("hybrid_qda", description="Ranking mode."),
) -> dict:
    eng = _get_engine()
    return eng.search(query=q, top_k=top_k, mode=mode)


@app.post("/search")
def search_post(payload: SearchRequest) -> dict:
    eng = _get_engine()
    return eng.search(query=payload.query, top_k=payload.top_k, mode=payload.mode)
