"""End-to-end DocMind retrieval, generation, and DeepEval pipeline."""
from __future__ import annotations
import os
from pathlib import Path
from typing import Any

from app.backend import KnowledgeBase, kb as default_kb
from app.generation import answer_question
from .datasets import load_beir_corpus, load_beir_scifact, load_synthetic_goldens
from .evaluator import RAGEvaluator
from .throughput import run_qps_evaluation


def run_evaluation(
    mode: str = "synthetic",
    max_samples: int = 50,
    knowledge_base: KnowledgeBase | None = None,
    k: int | None = None,
    *,
    include_qps: bool = False,
    qps_concurrency: int = 1,
    qps_warmup_queries: int = 0,
) -> dict[str, Any]:
    if mode not in {"beir", "synthetic", "both"}:
        raise ValueError("mode must be beir, synthetic, or both")
    if max_samples <= 0:
        raise ValueError("max_samples must be positive")
    store = knowledge_base or default_kb
    top_k = k or int(os.getenv("EVAL_K", "5"))
    # CLI/Docker starts a fresh process, so rebuild the in-memory BM25 index
    # from the mounted documents before evaluating.
    paths = sorted(store.upload_dir.glob("*") if store.upload_dir.exists() else [])
    docs_dir = Path("docs")
    if docs_dir.exists():
        paths.extend(sorted(p for p in docs_dir.glob("*") if p.is_file()))
    for path in paths:
        if str(path) not in store.documents:
            store.ingest(path.name, path.read_bytes())
    cases = []
    if mode in {"beir", "both"}:
        # BEIR is only useful if its corpus is indexed in the same retriever
        # being measured. IDs remain in metadata for traceability.
        for document_id, text in load_beir_corpus():
            store.ingest(f"beir-scifact-{document_id}.txt", text.encode("utf-8"))
        cases.extend(load_beir_scifact(max_samples))
    if mode in {"synthetic", "both"}:
        if not paths:
            raise ValueError("Synthetic evaluation requires documents in data/uploads")
        cases.extend(load_synthetic_goldens(paths, max_goldens_per_document=5))
    selected_cases = cases[:max_samples]
    if include_qps:
        def execute_query(question: str) -> dict[str, Any]:
            retrieved = store.retrieve(question, top_k=top_k)
            if not retrieved:
                return {"answer": "I do not know.", "citations": []}
            return answer_question(question, retrieved)

        # This is an end-to-end application benchmark (retrieval + answer
        # generation), using the same DeepEval golden inputs as its workload.
        qps_report = run_qps_evaluation(
            [str(case.input) for case in selected_cases],
            execute_query,
            concurrency=qps_concurrency,
            warmup_queries=qps_warmup_queries,
        )
    else:
        qps_report = None

    evaluated = []
    for case in selected_cases:
        retrieved = store.retrieve(case.input, top_k=top_k)
        context = [str(item.get("text", "")) for item in retrieved if item.get("text")]
        generated = answer_question(case.input, retrieved)
        actual = generated.get("answer", "") if isinstance(generated, dict) else str(generated)
        # Construct a fresh case: DeepEval metrics must see the actual context.
        from deepeval.test_case import LLMTestCase
        evaluated.append(LLMTestCase(input=case.input, actual_output=str(actual), expected_output=case.expected_output or "", retrieval_context=context, additional_metadata=case.additional_metadata))
    threshold = float(os.getenv("EVAL_THRESHOLD", "0.7"))
    report = RAGEvaluator(threshold=threshold).run(evaluated)
    report.update({"mode": mode, "k": top_k})
    if qps_report is not None:
        report["qps"] = qps_report
    return report
