"""Hybrid BM25 + vector retrieval with Reciprocal Rank Fusion."""


def reciprocal_rank_fusion(*ranked_lists: list[dict], k: int = 60, top_k: int = 10) -> list[dict]:
    """Fuse ranked result lists using RRF."""
    scores: dict[str, float] = {}
    documents: dict[str, dict] = {}
    for results in ranked_lists:
        for rank, result in enumerate(results, start=1):
            key = str(result.get("id", result.get("chunk_id", rank)))
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            documents[key] = result
    return [documents[key] | {"rrf_score": scores[key]} for key in sorted(scores, key=scores.get, reverse=True)[:top_k]]


class HybridRetriever:
    """Coordinate dense and sparse retrieval backends."""

    def retrieve(self, query: str, top_k: int = 10) -> list[dict]:
        raise NotImplementedError("Retrieval backends are not configured yet")
