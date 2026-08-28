"""Loaders for BEIR/Scifact and reusable DeepEval synthetic goldens."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any


def _case(input_text: str, expected: str, **metadata: Any) -> Any:
    try:
        from deepeval.test_case import LLMTestCase
    except ImportError as exc:
        raise RuntimeError("Install deepeval to load evaluation datasets") from exc
    return LLMTestCase(input=input_text, expected_output=expected, additional_metadata=metadata)


def load_beir_corpus() -> list[tuple[str, str]]:
    """Return Scifact document IDs and text for indexing by the pipeline."""
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Install datasets to load BEIR") from exc
    corpus_ds = load_dataset("BeIR/scifact", "corpus", split="corpus")
    return [(str(r.get("_id", r.get("id"))), f"{r.get('title', '')}\n{r.get('text', r.get('abstract', ''))}") for r in corpus_ds]


def load_beir_scifact(max_samples: int = 50) -> list[Any]:
    """Load Scifact and represent relevant abstracts as the expected output.

    Scifact is a retrieval benchmark, not a QA dataset; its relevant abstracts
    are therefore used as reference context. This makes it suitable for
    contextual precision/recall, while answer-quality scores should be treated
    as secondary for this loader.
    """
    if max_samples <= 0:
        raise ValueError("max_samples must be positive")
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Install datasets to load BEIR") from exc
    corpus_ds = load_dataset("BeIR/scifact", "corpus", split="corpus")
    queries_ds = load_dataset("BeIR/scifact", "queries", split="queries")
    # The HuggingFace mirror publishes qrels as a separate dataset.
    qrels_ds = load_dataset("BeIR/scifact-qrels", "default", split="test")
    corpus = {str(r.get("_id", r.get("id"))): f"{r.get('title', '')}\n{r.get('text', r.get('abstract', ''))}" for r in corpus_ds}
    qrels: dict[str, list[str]] = {}
    for row in qrels_ds:
        qid, did = str(row.get("query-id", row.get("query_id"))), str(row.get("corpus-id", row.get("corpus_id")))
        if int(row.get("score", row.get("relevance", 1))) > 0:
            qrels.setdefault(qid, []).append(did)
    cases = []
    for row in queries_ds:
        qid = str(row.get("_id", row.get("id")))
        relevant = qrels.get(qid, [])
        if not relevant:
            continue
        expected = "\n\n".join(corpus.get(doc_id, "") for doc_id in relevant).strip()
        cases.append(_case(str(row.get("text", row.get("query", ""))), expected, query_id=qid, relevant_document_ids=relevant, dataset="scifact"))
        if len(cases) >= max_samples:
            break
    return cases


def load_synthetic_goldens(document_paths: list[str | Path], max_goldens_per_document: int = 5, output_path: str | Path = "eval/synthetic_goldens.json") -> list[Any]:
    """Generate or reuse DeepEval goldens for already-ingested documents."""
    if max_goldens_per_document <= 0:
        raise ValueError("max_goldens_per_document must be positive")
    output = Path(output_path)
    if output.exists():
        data = json.loads(output.read_text(encoding="utf-8"))
    else:
        try:
            from deepeval.synthesizer import Synthesizer
        except ImportError as exc:
            raise RuntimeError("Install deepeval to generate synthetic goldens") from exc
        try:
            synthesizer = Synthesizer(max_goldens_per_context=max_goldens_per_document)
        except TypeError:
            synthesizer = Synthesizer()
        try:
            generated = synthesizer.generate_goldens_from_docs(
                document_paths=[str(p) for p in document_paths],
                include_expected_output=True,
            )
        except TypeError:
            generated = synthesizer.generate_goldens_from_docs(
                document_paths=[str(p) for p in document_paths],
            )
        data = [{"input": g.input, "expected_output": g.expected_output, "source": getattr(g, "source", None)} for g in generated]
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return [_case(row["input"], row.get("expected_output", ""), source=row.get("source"), dataset="synthetic") for row in data]
