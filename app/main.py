"""FastAPI entry point for DocMind."""

from __future__ import annotations

from dataclasses import asdict
from typing import Annotated

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.backend import kb
from app.generation import answer_question


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)
    mode: str = Field(default="classic", pattern="^(classic|agentic)$")
    max_iterations: int = Field(default=4, ge=1, le=10)


class EvalRequest(BaseModel):
    mode: str = Field(default="synthetic", pattern="^(beir|synthetic|both)$")
    max_samples: int = Field(default=50, ge=1, le=1000)
    include_qps: bool = False
    qps_concurrency: int = Field(default=1, ge=1, le=100)
    qps_warmup_queries: int = Field(default=0, ge=0, le=1000)


app = FastAPI(title="DocMind", version="0.2.0")
app.add_middleware(
    CORSMiddleware,
    # Vite may choose 5174+ when 5173 is busy; allow local development
    # ports while keeping production origins explicit.
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_origin_regex=r"https?://(localhost|127\.0\.0\.1):\d+",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, object]:
    return {"status": "ok", "documents": len(kb.documents), "chunks": kb.size}


@app.get("/documents")
def list_documents() -> dict[str, list[dict]]:
    return {"documents": list(kb.documents.values())}


@app.post("/ingest", status_code=201)
def ingest_document(file: Annotated[UploadFile, File(...)]) -> dict[str, object]:
    if not file.filename:
        raise HTTPException(status_code=400, detail="filename is required")
    try:
        content = file.file.read()
        result = kb.ingest(file.filename, content)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        # Avoid exposing filesystem, model, or provider internals to clients.
        raise HTTPException(status_code=500, detail=f"ingestion failed: {exc}") from exc
    return {
        "source": result.source,
        "modality": result.modality,
        "chunks": len(result.chunks),
        "assets": result.assets,
    }


@app.post("/query")
def query_documents(request: QueryRequest) -> dict[str, object]:
    if request.mode == "agentic":
        try:
            from app.agent import answer_agentically
            return asdict(answer_agentically(request.question, kb, request.max_iterations))
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"agent execution failed: {exc}") from exc
    try:
        retrieved = kb.retrieve(request.question, request.top_k)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"retrieval failed: {exc}") from exc
    if not retrieved:
        return {"answer": "I do not know.", "citations": [], "sources": [], "retrieved": []}
    try:
        generated = answer_question(request.question, retrieved)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        # Provider errors (invalid key, quota, timeout, etc.) should remain a
        # JSON response so the browser can display the real cause instead of
        # reporting the misleading generic "Failed to fetch" message.
        raise HTTPException(status_code=502, detail=f"generation failed: {exc}") from exc
    return {**generated, "sources": retrieved, "retrieved": len(retrieved)}


@app.post("/eval")
def evaluate_documents(request: EvalRequest) -> dict[str, object]:
    try:
        from backend.eval.pipeline import run_evaluation
        return run_evaluation(
            request.mode,
            request.max_samples,
            include_qps=request.include_qps,
            qps_concurrency=request.qps_concurrency,
            qps_warmup_queries=request.qps_warmup_queries,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"evaluation failed: {exc}") from exc


@app.get("/search")
def search_documents(
    q: Annotated[str, Query(min_length=1, max_length=2000)],
    top_k: Annotated[int, Query(ge=1, le=20)] = 5,
) -> dict[str, object]:
    """Inspect retrieval without calling the language model."""
    return {"results": kb.retrieve(q, top_k), "query": q}
