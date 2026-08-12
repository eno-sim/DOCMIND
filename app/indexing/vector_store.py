"""ChromaDB-backed dense vector indexing.

This module deliberately keeps ChromaDB behind a small application adapter.
The rest of DocMind works with ``TextChunk`` objects and result dictionaries,
not with Chroma's nested response format.  That gives the hybrid retriever a
stable interface and keeps database-specific details in one place.
"""

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from app.ingestion.text import TextChunk


class _SentenceTransformerEmbeddingFunction:
    """
    Wrapper for SentenceTransformerEmbeddingFunction that delays model loading until first use.
    Contains the __call__ method that converts a list of strings to a list of embedding vectors.
"""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._function: Any | None = None 

    def __call__(self, input: list[str]) -> list[list[float]]:
        # wrap the embedding function so it is only loaded when first called
        if self._function is None:
            try:
                from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
            except ImportError as exc:  # pragma: no cover - environment dependent
                raise RuntimeError(
                    "Install chromadb and sentence-transformers for vector indexing"
                ) from exc
            self._function = SentenceTransformerEmbeddingFunction(model_name=self.model_name)
        return self._function(input)


class VectorStore:
    """ChromaDB vector store.
    Parameters
    ----------
    collection_name:
        Chroma collection used for this index.
    persist_directory:
        If provided, use a persistent local Chroma database. If omitted, use
        an in-memory client, which is useful for tests and experiments.
    embedding_model:
        Sentence Transformers model name. ``all-MiniLM-L6-v2`` is a compact,
        general-purpose English baseline; it can be changed after evaluation.
    client / embedding_function:
        Dependency-injection hooks for tests and alternative embedding
        implementations. Supplying an embedding function avoids downloading
        model weights in unit tests.
    """

    def __init__(
        self,
        collection_name: str = "docmind",
        persist_directory: str | Path | None = None,
        embedding_model: str = "all-MiniLM-L6-v2",
        *,
        client: Any | None = None,
        embedding_function: Any | None = None,
    ) -> None:
        if not collection_name.strip():
            raise ValueError("collection_name must not be empty")
        if client is not None and persist_directory is not None:
            raise ValueError("Provide either client or persist_directory, not both")

        try:
            import chromadb
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError("Install chromadb to use VectorStore") from exc

        if client is not None:
            self._client = client
        elif persist_directory is not None:
            self._client = chromadb.PersistentClient(path=str(Path(persist_directory)))
        else:
            self._client = chromadb.EphemeralClient() 

        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self._embedding_function = embedding_function or _SentenceTransformerEmbeddingFunction(
            embedding_model
        )
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            embedding_function=self._embedding_function,
            metadata={"hnsw:space": "cosine"},
        )

    @property
    def size(self) -> int:
        """Return the number of vectors currently stored."""
        return int(self._collection.count())

    def add_chunks(self, chunks: Iterable[TextChunk]) -> None:
        """Embed and upsert chunks into ChromaDB.

        Chroma's ``upsert`` makes re-ingestion idempotent for stable chunk IDs.
        The original text is stored as the document and scalar metadata is
        stored separately for filtering and citations.
        """
        prepared = []
        for chunk in chunks:
            if not isinstance(chunk, TextChunk):
                raise TypeError("VectorStore accepts TextChunk instances")
            if not chunk.text.strip():
                continue
            metadata = {**chunk.metadata, "source": chunk.source}
            prepared.append((chunk.chunk_id, chunk.text, metadata))

        if not prepared:
            return
        self._collection.upsert(
            ids=[item[0] for item in prepared],
            documents=[item[1] for item in prepared],
            metadatas=[item[2] for item in prepared],
        ) # this is the way chunks are inserted into chromadb

    def search(
        self,
        query: str,
        top_k: int = 10,
        metadata_filter: dict[str, str | int | float | bool] | None = None,
    ) -> list[dict[str, Any]]:
        """Return nearest chunks with higher-is-better normalized scores.

        Chroma returns distances where lower means more similar.  The adapter
        preserves the raw ``distance`` and additionally exposes
        ``score = 1 / (1 + distance)`` so this result can be fused with BM25
        scores by the hybrid retriever.
        """
        if not isinstance(query, str):
            raise TypeError("query must be a string")
        if not query.strip():
            return []
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")
        if metadata_filter is not None and not isinstance(metadata_filter, dict):
            raise TypeError("metadata_filter must be a dictionary")
        if self.size == 0:
            return []

        kwargs: dict[str, Any] = {
            "query_texts": [query],
            "n_results": min(top_k, self.size),
            "include": ["documents", "metadatas", "distances"],
        }
        if metadata_filter:
            kwargs["where"] = metadata_filter
        
        # kwargs is passed to the chromadb collection query method, which returns a dictionary with keys "ids", "documents", "metadatas", and "distances". Each key maps to a list of lists, where the outer list corresponds to the queries (in this case, just one query), and the inner lists contain the results for that query.
        raw = self._collection.query(**kwargs)

        ids = raw.get("ids", [[]])[0]
        documents = raw.get("documents", [[]])[0]
        metadatas = raw.get("metadatas", [[]])[0]
        distances = raw.get("distances", [[]])[0]
        results: list[dict[str, Any]] = []
        for chunk_id, document, metadata, distance in zip(
            ids, documents, metadatas, distances, strict=False
        ):
            metadata = dict(metadata or {})
            numeric_distance = float(distance)
            results.append({
                "id": chunk_id,
                "chunk_id": chunk_id,
                "text": document,
                "source": str(metadata.get("source", "")),
                "metadata": metadata,
                "distance": numeric_distance,
                "score": 1.0 / (1.0 + numeric_distance),
            })
        return results

    def remove_chunks(self, chunk_ids: Iterable[str]) -> None:
        """Delete vectors by their stable chunk IDs."""
        ids = list(chunk_ids)
        if ids:
            self._collection.delete(ids=ids)

    def clear(self) -> None:
        """Delete and recreate the collection while preserving its name."""
        self._client.delete_collection(name=self.collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self._embedding_function,
            metadata={"hnsw:space": "cosine"},
        )

    def __len__(self) -> int:
        return self.size
