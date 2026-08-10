"""Hybrid retrieval over BM25+ and dense vector search.

The two indexes have different strengths: BM25 is precise for names, dates,
and exact terminology, while embeddings can match paraphrases. Reciprocal
Rank Fusion (RRF) combines their *rank positions* rather than raw scores,
which is important because BM25 scores and vector distances are not
comparable quantities.
"""

from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any


def _result_id(result: dict[str, Any]) -> str:
    """Get the stable identity used to merge results from both indexes."""
    value = result.get("id", result.get("chunk_id"))
    if value is None or str(value) == "":
        raise ValueError("Every retrieval result must contain id or chunk_id")
    return str(value)


def reciprocal_rank_fusion(
    *ranked_lists: list[dict[str, Any]],
    k: int = 60,
    top_k: int = 10,
    list_names: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """Fuse ranked result lists using Reciprocal Rank Fusion.

    For a result at rank ``r``, RRF contributes ``1 / (k + r)``. A chunk
    appearing in both lists accumulates both contributions. Raw BM25 and
    vector scores are deliberately not combined because they use different
    scales.

    ``list_names`` is optional diagnostic information. When supplied, the
    fused result includes ``retrieval_sources`` and ``rrf_ranks`` so callers
    can inspect whether a chunk was found by BM25, vector search, or both.
    """
    if k <= 0:
        raise ValueError("k must be greater than zero")
    if top_k <= 0:
        raise ValueError("top_k must be greater than zero")
    if list_names is not None and len(list_names) != len(ranked_lists):
        raise ValueError("list_names must have one name per ranked list")

    scores: dict[str, float] = {}
    documents: dict[str, dict[str, Any]] = {}
    rank_details: dict[str, dict[str, int]] = {}

    for list_index, results in enumerate(ranked_lists):
        name = list_names[list_index] if list_names is not None else str(list_index)
        for rank, result in enumerate(results, start=1):
            if not isinstance(result, dict):
                raise TypeError("retrieval results must be dictionaries")
            key = _result_id(result)
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            # Both backends contain the same citation fields. Keep the first
            # copy as the canonical result and only add fusion diagnostics.
            documents.setdefault(key, dict(result))
            rank_details.setdefault(key, {})[name] = rank

    ranked_ids = sorted(
        scores,
        key=lambda key: (-scores[key], min(rank_details[key].values())),
    )
    fused: list[dict[str, Any]] = []
    for key in ranked_ids[:top_k]:
        result = documents[key] | {"rrf_score": scores[key]}
        if list_names is not None:
            result["retrieval_sources"] = sorted(rank_details[key])
            result["rrf_ranks"] = dict(rank_details[key])
        fused.append(result)
    return fused


class HybridRetriever:
    """Run sparse and dense retrieval and fuse their ranked candidates.

    Both backends must expose ``search(query, top_k, metadata_filter)``. The
    concrete BM25 and Chroma adapters already provide this interface, while
    duck typing keeps this class straightforward to test and extend.
    """

    def __init__(
        self,
        sparse_index: Any,
        vector_store: Any,
        *,
        rrf_k: int = 60,
        candidate_k: int = 20,
    ) -> None:
        if rrf_k <= 0:
            raise ValueError("rrf_k must be greater than zero")
        if candidate_k <= 0:
            raise ValueError("candidate_k must be greater than zero")
        if not hasattr(sparse_index, "search") or not hasattr(vector_store, "search"):
            raise TypeError("Both retrieval backends must provide a search method")

        self.sparse_index = sparse_index
        self.vector_store = vector_store
        self.rrf_k = rrf_k
        self.candidate_k = candidate_k

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Return RRF-ranked results from BM25+ and vector retrieval.

        The backend calls run concurrently because they are independent and
        vector search may perform model/database work while BM25 is scoring.
        ``candidate_k`` controls how many results each backend contributes to
        fusion; it is at least ``top_k`` so the requested output is possible.
        """
        if not isinstance(query, str):
            raise TypeError("query must be a string")
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")
        if metadata_filter is not None and not isinstance(metadata_filter, dict):
            raise TypeError("metadata_filter must be a dictionary")

        backend_top_k = max(top_k, self.candidate_k)
        with ThreadPoolExecutor(max_workers=2) as executor:
            sparse_future = executor.submit(
                self.sparse_index.search,
                query,
                backend_top_k,
                metadata_filter,
            )
            vector_future = executor.submit(
                self.vector_store.search,
                query,
                backend_top_k,
                metadata_filter,
            )
            sparse_results = sparse_future.result()
            vector_results = vector_future.result()

        return reciprocal_rank_fusion(
            sparse_results,
            vector_results,
            k=self.rrf_k,
            top_k=top_k,
            list_names=("bm25", "vector"),
        )
