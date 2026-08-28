"""Application services used by the FastAPI API."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

# Chroma's optional telemetry is not needed for local evaluation and can emit
# noisy compatibility errors with newer posthog versions.
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
from threading import RLock
from typing import Any

from app.generation import answer_question
from app.indexing import BM25Index, VectorStore
from app.ingestion import IngestionResult, ingest_file
from app.retrieval import HybridRetriever


class KnowledgeBase:
    """Small process-local document store and indexing facade.

    BM25 is available immediately. Chroma is created on the first ingestion so
    health checks do not load models or create files. The store is intentionally
    process-local for this phase; persistent Chroma vectors survive restarts,
    while the sparse index is rebuilt from uploaded files at startup in a later
    phase.
    """

    def __init__(self, data_dir: str | Path = "data") -> None:
        self.data_dir = Path(data_dir)
        self.upload_dir = self.data_dir / "uploads"
        self.sparse = BM25Index()
        self.vector: VectorStore | None = None
        self.documents: dict[str, dict[str, Any]] = {}
        self._lock = RLock()

    @property
    def size(self) -> int:
        return self.sparse.size

    def ingest(self, filename: str, content: bytes) -> IngestionResult:
        safe_name = Path(filename).name
        if not safe_name or safe_name in {".", ".."}:
            raise ValueError("A valid filename is required")
        destination = self.upload_dir / safe_name
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)
        result = ingest_file(destination)
        with self._lock:
            if self.vector is None:
                self.vector = VectorStore(persist_directory=self.data_dir / "chroma")
            self.sparse.add_chunks(result.chunks)
            self.vector.add_chunks(result.chunks)
            self.documents[str(destination)] = {
                "source": str(destination),
                "filename": safe_name,
                "modality": result.modality,
                "chunks": len(result.chunks),
                "assets": len(result.assets),
            }
        return result

    def retrieve(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        with self._lock:
            if self.size == 0:
                return []
            if self.vector is None:
                return self.sparse.search(query, top_k=top_k)
            return HybridRetriever(self.sparse, self.vector, candidate_k=max(top_k, 10)).retrieve(
                query, top_k=top_k
            )


kb = KnowledgeBase()
