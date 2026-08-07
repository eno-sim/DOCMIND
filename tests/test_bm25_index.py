import pytest

from app.ingestion.text import TextChunk
from app.indexing.bm25_index import BM25Index


pytest.importorskip("rank_bm25")


def chunk(chunk_id: str, text: str, source: str = "notes.md") -> TextChunk:
    return TextChunk(text=text, source=source, chunk_id=chunk_id, metadata={"file_type": "md"})


def test_bm25_returns_relevant_chunks_and_original_metadata():
    index = BM25Index()
    index.add_chunks([
        chunk("one", "The project deadline is Friday."),
        chunk("two", "The recipe uses flour and water."),
    ])

    results = index.search("project deadline", top_k=1)

    assert len(results) == 1
    assert results[0]["chunk_id"] == "one"
    assert results[0]["text"] == "The project deadline is Friday."
    # The fixture explicitly supplies file_type="md". This assertion checks
    # that BM25 preserves ingestion metadata for downstream citations/filters;
    # BM25 itself does not infer the file type.
    assert results[0]["metadata"]["file_type"] == "md"
    assert results[0]["score"] > 0


def test_reingesting_same_chunk_id_is_an_upsert():
    index = BM25Index()
    index.add_chunks([chunk("one", "old text")])
    index.add_chunks([chunk("one", "new project deadline")])

    assert index.size == 1
    assert index.search("project")[0]["text"] == "new project deadline"


def test_unknown_query_returns_no_results():
    index = BM25Index()
    index.add_chunks([chunk("one", "known term")])

    assert index.search("not-in-the-index") == []
