"""FastAPI entry point for DocMind."""

from fastapi import FastAPI

app = FastAPI(title="DocMind", version="0.1.0")


@app.get("/health")
def health() -> dict[str, str]:
    """Return service health information."""
    return {"status": "ok"}


@app.get("/documents")
def list_documents() -> dict[str, list]:
    """List indexed documents.

    The document store will be connected here during the indexing phase.
    """
    return {"documents": []}


@app.post("/ingest")
def ingest_document() -> dict[str, str]:
    """Ingest a document (placeholder for the ingestion pipeline)."""
    return {"status": "not_implemented"}


@app.post("/query")
def query_documents() -> dict[str, str]:
    """Query the knowledge base (placeholder for retrieval and generation)."""
    return {"status": "not_implemented"}
