"""DeepEval adapter with a stable, JSON-serializable result shape."""
from __future__ import annotations
from typing import Any


class RAGEvaluator:
    def __init__(self, threshold: float = 0.7, include_bonus: bool = True) -> None:
        if not 0 <= threshold <= 1:
            raise ValueError("threshold must be between 0 and 1")
        self.threshold = threshold
        self.include_bonus = include_bonus

    def metrics(self) -> list[Any]:
        try:
            from deepeval.metrics import (
                AnswerRelevancyMetric, ContextualPrecisionMetric,
                ContextualRecallMetric, ContextualRelevancyMetric,
                FaithfulnessMetric,
            )
        except ImportError as exc:
            raise RuntimeError("Install deepeval to run evaluations") from exc
        classes = [ContextualPrecisionMetric, ContextualRecallMetric]
        if self.include_bonus:
            classes += [ContextualRelevancyMetric, FaithfulnessMetric, AnswerRelevancyMetric]
        import os
        model = os.getenv("DEEPEVAL_MODEL")
        kwargs = {"threshold": self.threshold, "include_reason": True}
        if model:
            kwargs["model"] = model
        return [cls(**kwargs) for cls in classes]

    def run(self, test_cases: list[Any]) -> dict[str, Any]:
        if not test_cases:
            return {"threshold": self.threshold, "samples": 0, "metrics": {}}
        try:
            from deepeval import evaluate
        except ImportError as exc:
            raise RuntimeError("Install deepeval to run evaluations") from exc
        metrics = self.metrics()
        # DeepEval's on-disk cache manager can be uninitialized on Windows
        # (especially after an interrupted run). Disable it for reproducible
        # CLI/API evaluations; DeepEval still returns the complete report.
        from deepeval.evaluate.configs import AsyncConfig, CacheConfig
        # DeepEval currently has a Windows bug where its global cache object
        # can remain None even when disk caching is disabled. Initialize it as
        # a defensive compatibility workaround and use the synchronous runner.
        import deepeval.test_run.cache as deeval_cache
        deeval_cache.global_test_run_cache_manager.disable_write_cache = True
        deeval_cache.global_test_run_cache_manager.cached_test_run = deeval_cache.CachedTestRun()
        evaluation = evaluate(
            test_cases=test_cases,
            metrics=metrics,
            cache_config=CacheConfig(write_cache=False, use_cache=False),
            async_config=AsyncConfig(run_async=False),
        )
        rows: dict[str, list[tuple[float, bool]]] = {m.__class__.__name__: [] for m in metrics}
        aliases = {}
        for name in rows:
            key = "".join(ch.lower() for ch in name if ch.isalnum())
            aliases[key] = name
            aliases[key.removesuffix("metric")] = name
        # DeepEval has changed its result wrapper between releases. Read the
        # public test_results/metrics attributes and tolerate either objects or dicts.
        results = getattr(evaluation, "test_results", None) or getattr(evaluation, "results", None) or []
        for result in results:
            items = (result.get("metrics", result.get("metrics_data", [])) if isinstance(result, dict) else (getattr(result, "metrics_data", None) or getattr(result, "metrics", None) or []))
            for item in items:
                raw_name = (item.get("metric", item.get("name", "")) if isinstance(item, dict) else getattr(item, "metric_name", None) or getattr(item, "name", None) or "")
                name = aliases.get("".join(ch.lower() for ch in str(raw_name) if ch.isalnum()))
                score = item.get("score") if isinstance(item, dict) else getattr(item, "score", None)
                passed = item.get("success") if isinstance(item, dict) else getattr(item, "success", None)
                if name and score is not None:
                    rows[name].append((float(score), bool(passed if passed is not None else float(score) >= self.threshold)))
        report: dict[str, Any] = {"threshold": self.threshold, "samples": len(test_cases), "metrics": {}}
        for metric in metrics:
            name = metric.__class__.__name__
            values = rows[name]
            score = sum(v[0] for v in values) / len(values) if values else None
            report["metrics"][name] = {"score": score, "passed": bool(score is not None and score >= self.threshold), "threshold": self.threshold, "evaluated": len(values)}
        return report
