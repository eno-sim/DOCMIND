from uuid import uuid4

import pytest

from app.ingestion.text import TextChunk


pytest.importorskip("chromadb")

from app.indexing.vector_store import VectorStore  # noqa: E402


class FakeEmbeddingFunction:
    """Small deterministic embedding function for tests.

    Production uses Sentence Transformers. Tests should not download model
    weights, so this maps a few known terms to orthogonal dimensions.
    """

    vocabulary = {"project": 0, "deadline": 1, "recipe": 2, "flour": 3}

    def __call__(self, input: list[str]) -> list[list[float]]:
        embeddings = []
        for text in input:
            vector = [0.0] * len(self.vocabulary)
            for term in text.lower().split():
                term = term.strip(".,!?")
                if term in self.vocabulary:
                    vector[self.vocabulary[term]] += 1.0
            embeddings.append(vector)
        return embeddings


def make_store() -> VectorStore:
    import chromadb

    return VectorStore(
        # Chroma's in-memory clients can share a process-level system; use a
        # unique collection so tests cannot leak vectors into one another.
        collection_name=f"test-docmind-{uuid4().hex}",
        client=chromadb.EphemeralClient(),
        embedding_function=FakeEmbeddingFunction(),
    )


def make_chunk(chunk_id: str, text: str, file_type: str = "md") -> TextChunk:
    return TextChunk(
        text=text,
        source=f"{chunk_id}.{file_type}",
        chunk_id=chunk_id,
        metadata={"file_type": file_type},
    )


def test_vector_search_returns_citation_friendly_results():
    store = make_store()
    store.add_chunks([
        make_chunk("one", "The project deadline is Friday."),
        make_chunk("two", "The recipe uses flour and water.", "txt"),
    ])

    results = store.search("project deadline", top_k=1)

    assert len(results) == 1
    assert results[0]["id"] == "one"
    assert results[0]["chunk_id"] == "one"
    assert results[0]["text"] == "The project deadline is Friday."
    assert results[0]["source"] == "one.md"
    assert results[0]["metadata"]["file_type"] == "md"
    assert isinstance(results[0]["distance"], float)
    assert isinstance(results[0]["score"], float)


def test_reingesting_same_chunk_id_is_an_upsert():
    store = make_store()
    store.add_chunks([make_chunk("one", "The project deadline is Friday.")])
    store.add_chunks([make_chunk("one", "The recipe uses flour.", "txt")])

    assert store.size == 1
    results = store.search("recipe", top_k=1)
    assert results[0]["text"] == "The recipe uses flour."
    assert results[0]["source"] == "one.txt"


def test_metadata_filter_and_empty_query():
    store = make_store()
    store.add_chunks([
        make_chunk("one", "The project deadline is Friday."),
        make_chunk("two", "The recipe uses flour and water.", "txt"),
    ])

    results = store.search("project recipe", metadata_filter={"file_type": "txt"})

    assert results
    assert all(result["metadata"]["file_type"] == "txt" for result in results)
    assert store.search("   ") == []


def test_remove_and_clear():
    store = make_store()
    store.add_chunks([make_chunk("one", "The project deadline is Friday.")])

    store.remove_chunks(["one"])
    assert store.size == 0

    store.add_chunks([make_chunk("one", "The project deadline is Friday.")])
    store.clear()
    assert store.size == 0
    assert store.search("project") == []
