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

The indexing components are now implemented:

- BM25+ sparse term retrieval with English and Italian tokenization.
- ChromaDB dense retrieval using a lazy-loaded Sentence Transformers model.
- A shared result shape suitable for the future hybrid retriever.

Hybrid retrieval and Claude-backed generation are implemented. The
`notebooks/full_pipeline_generation.ipynb` notebook demonstrates the complete
pipeline with a deterministic offline dense backend and a fake Claude client;
an optional cell enables the real API. Evaluation remains a future phase.

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
from app.indexing import BM25Index, VectorStore

chunks = ingest_file("notes/meeting.md").chunks

sparse_index = BM25Index()
sparse_index.add_chunks(chunks)
sparse_results = sparse_index.search("project deadline", top_k=5)

# Uses all-MiniLM-L6-v2 by default and persists locally when a directory is given.
dense_index = VectorStore(persist_directory="data/chroma")
dense_index.add_chunks(chunks)
dense_results = dense_index.search("project deadline", top_k=5)
```

### ChromaDB design

`VectorStore` uses ChromaDB with the local `all-MiniLM-L6-v2` Sentence
Transformers model as a compact embedding baseline. The model is loaded only
when the first add/search operation needs embeddings. Chroma distances are
converted into a higher-is-better `score = 1 / (1 + distance)` while the raw
`distance` is retained for diagnostics. Tests inject a deterministic embedding
function so they do not download model weights.

## Generation design notes

`app.generation.answer_question()` accepts the normalized result dictionaries
returned by `HybridRetriever`. It formats each chunk with its source and page,
chunk, or timestamp location, then sends only that context to Anthropic Claude.
The system prompt requires inline citations and an explicit "I do not know"
response when the retrieved evidence is insufficient. The Anthropic client is
created lazily and can be injected for tests or local demos:

```python
from app.generation import answer_question

answer = answer_question("What is the deadline?", retrieved_results)
print(answer["answer"], answer["citations"])
```

Set `ANTHROPIC_API_KEY` and optionally `ANTHROPIC_MODEL` before calling without
a client. The returned dictionary also includes the selected model and API
usage when the provider supplies it.

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

## Web application

The React/Vite client is in `frontend/`. Run the API and client separately:

```bash
# terminal 1
uvicorn app.main:app --reload

# terminal 2
cd frontend
npm install
npm run dev
```

Then open http://localhost:5173. The UI uploads supported text/PDF files,
shows indexed documents, and sends questions to the `/query` endpoint. The
API also exposes `/search` for inspecting retrieval without calling Claude.
Alternatively, `docker compose up --build` starts both services.

## Evaluation and QPS benchmark

DeepEval evaluates answer and retrieval quality using the synthetic goldens (or
BEIR), while the optional QPS benchmark measures DocMind's actual end-to-end
query path: retrieval plus answer generation. QPS is a throughput measurement,
not an LLM-judge metric; DeepEval's evaluation inputs are used as its workload.

```bash
# Quality metrics only
python cli.py eval --mode synthetic --max-samples 20

# Quality metrics plus full RAG throughput, after one warmup call per query
python cli.py eval --mode synthetic --max-samples 20 --qps \
  --qps-concurrency 4 --qps-warmup-queries 20
```

The JSON report's `qps` object includes completed and failed requests,
wall-clock `queries_per_second`, and mean/p50/p95/max request latency in
milliseconds. Failed requests remain in the completed count and are listed
(up to ten errors), so provider errors do not produce a misleading throughput
score. The same options are available through `POST /eval` as `include_qps`,
`qps_concurrency`, and `qps_warmup_queries`.

## Docker

```bash
docker compose up --build
```

Keep `.env` and the `data/` directory out of version control; `.env.example` documents the required configuration.
