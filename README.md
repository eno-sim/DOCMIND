# DocMind

DocMind is a local, privacy-preserving knowledge-base search and Q&A system for documents, audio, and video.

## Planned architecture

```text
Files -> ingestion -> chunks + metadata -> ChromaDB + BM25
                                      \-> RRF hybrid retrieval -> Claude Q&A
```

## Current status

The ingestion component is implemented as the first development slice:

- Text, Markdown, and PDF extraction via `pypdf` (with page-aware metadata).
- Deterministic, whitespace-aware character chunking with overlap.
- Lazy Whisper transcription for audio, preserving segment timestamps.
- OpenCV fixed-interval keyframe extraction for video.
- Video transcription through the same Whisper path as standalone audio.
- A unified `app.ingestion.pipeline.ingest_file()` entry point that returns
  common `TextChunk` objects for downstream indexing.

The BM25 indexing component is now implemented. ChromaDB/vector indexing,
hybrid retrieval, generation, and evaluation remain the next phases.

## BM25 indexing design

`app.indexing.tokenizer` provides one shared tokenizer for both chunks and
queries. It applies Unicode normalization, optional accent folding, case
folding, punctuation splitting, and built-in English and Italian stopword
removal. Stopword removal is enabled by default but can be disabled through
`TokenizerConfig`; custom stopwords and the supported language selection are
configurable. Stemming is intentionally not applied because it can merge
unrelated terms and make citations/debugging less transparent.

`app.indexing.bm25_index.BM25Index` uses `rank_bm25.BM25Plus` with `k1=1.5`,
`b=0.75`, and `delta=1.0` defaults. BM25+ adds a delta term to reduce the
length bias against relevant terms in longer chunks. It keeps original
`TextChunk` objects for citations, while indexing only normalized tokens.
Chunks are upserted by their stable `chunk_id`, and the index rebuilds lazily
after changes. Out-of-vocabulary query terms are removed before scoring, and
BM25+'s baseline-only results are excluded, so a query with no actual term
matches returns no arbitrary zero-information chunks.

```python
from app.ingestion import ingest_file
from app.indexing import BM25Index

index = BM25Index()
index.add_chunks(ingest_file("notes/meeting.md").chunks)
results = index.search("project deadline", top_k=5)
```

## Ingestion design notes

### Why character chunks first?

A deterministic character window is a transparent baseline: it has no model
startup cost, behaves consistently across machines, and is easy to evaluate.
The default size is 800 characters with 120 characters of overlap. Chunks
prefer whitespace boundaries and retain source metadata. This choice can be
revisited after retrieval metrics are available rather than guessing that a
more complicated semantic splitter will perform better.

### Why preserve page and time metadata?

Citations must point to something a user can locate. PDF chunks include a
one-based `page`, while audio/video transcript chunks include
`start_seconds` and `end_seconds`. Video keyframes are saved only when a
`keyframe_dir` is explicitly supplied, avoiding unexpected writes from a
library call.

### Why lazy-load Whisper and OpenCV?

These dependencies are comparatively heavy and Whisper downloads model
weights. Imports and model loading therefore happen only when audio/video
ingestion is requested. The API can start and serve health checks without
allocating a transcription model.

### Example

```python
from app.ingestion.pipeline import ingest_file

result = ingest_file("notes/meeting.md")
for chunk in result.chunks:
    print(chunk.chunk_id, chunk.metadata, chunk.text[:80])
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --reload
```

Open http://localhost:8000/docs for the API documentation.

## Docker

```bash
docker compose up --build
```

Keep `.env` and the `data/` directory out of version control; `.env.example` documents the required configuration.
