"""ChromaDB vector-store integration."""

from app.ingestion.text import TextChunk


class VectorStore:
    """Small adapter around a ChromaDB collection."""

    def __init__(self, collection_name: str = "docmind") -> None:
        self.collection_name = collection_name
        self._collection = None

    def add_chunks(self, chunks: list[TextChunk]) -> None:
        """Embed and persist chunks in ChromaDB."""
        raise NotImplementedError("ChromaDB integration is not configured yet")

    def search(self, query: str, top_k: int = 10) -> list[dict]:
        """Return nearest vector matches."""
        raise NotImplementedError("ChromaDB integration is not configured yet")
