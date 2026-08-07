"""BM25+ sparse index for term-based retrieval.

The index keeps the original :class:`~app.ingestion.text.TextChunk` objects
for citation, while ``rank_bm25`` receives only normalized token lists.  This
separation is important: retrieval operates on a clean representation, but
answers must still quote and cite the original chunk text and metadata.
"""

from typing import Any, Iterable

from app.ingestion.text import TextChunk
from app.indexing.tokenizer import TokenizerConfig, tokenize


class BM25Index:
    """In-memory BM25Plus index.

    BM25+ adds a small ``delta`` term to the term-frequency component.  This
    reduces the tendency of regular BM25 to over-penalize relevant terms in
    longer chunks.  ``add_chunks`` is an upsert operation keyed by
    ``chunk_id``.  This makes
    re-ingesting a document idempotent instead of doubling its term frequency
    and unfairly changing retrieval scores.  The index is intentionally
    in-memory for this phase; persistence can be added after the scoring and
    filtering behavior is tested.
    """

    def __init__(
        self,
        tokenizer_config: TokenizerConfig | None = None,
        *,
        k1: float = 1.5,
        b: float = 0.75,
        delta: float = 1.0,
    ) -> None:
        if k1 < 0:
            raise ValueError("k1 must be non-negative")
        if not 0 <= b <= 1:
            raise ValueError("b must be between 0 and 1")
        if delta < 0:
            raise ValueError("delta must be non-negative")

        self.tokenizer_config = tokenizer_config or TokenizerConfig()
        self.k1 = k1
        self.b = b
        self.delta = delta
        self._chunks: dict[str, TextChunk] = {}
        self._tokenized_documents: list[list[str]] = []
        self._vocabulary: set[str] = set()
        self._bm25: Any | None = None
        self._dirty = False

    @property
    def chunks(self) -> list[TextChunk]:
        """Return indexed chunks in insertion order.

        A copy is returned so callers cannot mutate the index without forcing
        a rebuild.  Use ``add_chunks`` to replace a chunk safely.
        """
        return list(self._chunks.values())

    @property
    def size(self) -> int:
        """Number of unique chunks currently indexed."""
        return len(self._chunks)

    def add_chunks(self, chunks: Iterable[TextChunk]) -> None:
        """Add or replace chunks and rebuild lazily on the next search."""
        changed = False
        for chunk in chunks:
            if not isinstance(chunk, TextChunk):
                raise TypeError("BM25Index accepts TextChunk instances")
            if not chunk.text.strip():
                # Empty chunks cannot produce terms and only create confusing
                # zero-score results, so they are ignored at the index edge.
                continue
            if self._chunks.get(chunk.chunk_id) != chunk:
                self._chunks[chunk.chunk_id] = chunk
                changed = True
        self._dirty = self._dirty or changed

    def remove_chunks(self, chunk_ids: Iterable[str]) -> int:
        """Remove chunks by ID and return the number actually removed."""
        removed = 0
        for chunk_id in chunk_ids:
            if self._chunks.pop(chunk_id, None) is not None:
                removed += 1
        if removed:
            self._dirty = True
        return removed

    def clear(self) -> None:
        """Remove every chunk from the index."""
        if self._chunks:
            self._chunks.clear()
            self._tokenized_documents.clear()
            self._vocabulary.clear()
            self._bm25 = None
            self._dirty = False

    def _rebuild(self) -> None:
        """Create the rank_bm25 corpus from the current canonical chunks."""
        self._tokenized_documents = [
            tokenize(chunk.text, self.tokenizer_config)
            for chunk in self._chunks.values()
        ]
        self._vocabulary = {
            token for document in self._tokenized_documents for token in document
        }
        self._bm25 = None
        if self._tokenized_documents and self._vocabulary:
            try:
                from rank_bm25 import BM25Plus
            except ImportError as exc:  # pragma: no cover - environment dependent
                raise RuntimeError("Install rank-bm25 to use BM25+ retrieval") from exc
            self._bm25 = BM25Plus(
                self._tokenized_documents,
                k1=self.k1,
                b=self.b,
                delta=self.delta,
            )
        self._dirty = False

    def search(
        self,
        query: str,
        top_k: int = 10,
        metadata_filter: dict[str, str] | None = None,
    ) -> list[dict[str, Any]]:
        """Return the highest scoring chunks for a normalized term query.

        Results contain ``id``, ``chunk_id``, ``text``, ``source``, ``metadata``,
        and ``score`` so they can be consumed directly by the hybrid retriever.
        A metadata filter is applied after scoring, while all documents are
        scored, ensuring filtering does not change the BM25 corpus statistics.
        """
        if not isinstance(query, str):
            raise TypeError("query must be a string")
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")
        if metadata_filter is not None and not isinstance(metadata_filter, dict):
            raise TypeError("metadata_filter must be a dictionary")

        if self._dirty:
            self._rebuild()
        # BM25Plus assigns an IDF value to out-of-vocabulary terms.  Removing
        # them here prevents an unknown query from producing a baseline score
        # for every document.
        query_terms = [
            term for term in tokenize(query, self.tokenizer_config)
            if term in self._vocabulary
        ]
        if not query_terms or self._bm25 is None:
            return []

        scores = self._bm25.get_scores(query_terms)
        chunks = list(self._chunks.values())
        ranked_indexes = sorted(range(len(chunks)), key=lambda index: scores[index], reverse=True)
        results: list[dict[str, Any]] = []
        for index in ranked_indexes:
            # BM25+ deliberately gives every document a positive delta
            # baseline for an in-vocabulary term.  A document with no actual
            # query-term occurrence is not a useful retrieval result, so drop
            # that baseline-only case explicitly.
            if not any(term in self._tokenized_documents[index] for term in query_terms):
                continue
            chunk = chunks[index]
            if metadata_filter and any(chunk.metadata.get(key) != value for key, value in metadata_filter.items()):
                continue
            results.append({
                "id": chunk.chunk_id,
                "chunk_id": chunk.chunk_id,
                "text": chunk.text,
                "source": chunk.source,
                "metadata": dict(chunk.metadata),
                "score": float(scores[index]),
            })
            if len(results) >= top_k:
                break
        return results

    def __len__(self) -> int:
        return self.size
