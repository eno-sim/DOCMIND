"""Throughput benchmarking for DocMind query workloads.

DeepEval supplies the golden/query dataset used by the evaluation pipeline, but
it does not expose a QPS metric.  This module measures process-local, end-to-
end query throughput separately so it can be reported next to DeepEval's
quality metrics.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from statistics import mean
from time import perf_counter
from typing import Any, Callable, Sequence


def _percentile(values: list[float], percentile: float) -> float | None:
    """Return a nearest-rank percentile without requiring NumPy."""
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int((len(ordered) - 1) * percentile)))
    return ordered[index]


def run_qps_evaluation(
    queries: Sequence[str],
    query_fn: Callable[[str], Any],
    *,
    concurrency: int = 1,
    warmup_queries: int = 0,
) -> dict[str, Any]:
    """Measure completed end-to-end queries per second for ``query_fn``.

    Warmup calls are deliberately excluded from both the elapsed time and
    result count. Exceptions are counted as failed completed requests rather
    than aborting a benchmark, which makes transient provider failures visible
    in the report.
    """
    if concurrency <= 0:
        raise ValueError("qps_concurrency must be positive")
    if warmup_queries < 0:
        raise ValueError("qps_warmup_queries cannot be negative")

    workload = list(queries)
    if not workload:
        return {
            "queries": 0,
            "completed": 0,
            "failed": 0,
            "concurrency": concurrency,
            "warmup_queries": 0,
            "elapsed_seconds": 0.0,
            "queries_per_second": 0.0,
            "latency_ms": {"mean": None, "p50": None, "p95": None, "max": None},
            "errors": [],
        }

    # Cycle a supplied workload for warmup without changing the measured set.
    for index in range(warmup_queries):
        try:
            query_fn(workload[index % len(workload)])
        except Exception:
            # The measured run reports failures; a failed warmup should not
            # prevent a benchmark of a temporarily unavailable dependency.
            pass

    def invoke(query: str) -> tuple[float, str | None]:
        started = perf_counter()
        try:
            query_fn(query)
        except Exception as exc:
            return perf_counter() - started, f"{type(exc).__name__}: {exc}"
        return perf_counter() - started, None

    started = perf_counter()
    latencies: list[float] = []
    errors: list[str] = []
    with ThreadPoolExecutor(max_workers=concurrency, thread_name_prefix="docmind-qps") as executor:
        futures = [executor.submit(invoke, query) for query in workload]
        for future in as_completed(futures):
            latency, error = future.result()
            latencies.append(latency)
            if error is not None:
                errors.append(error)
    elapsed = perf_counter() - started
    # Latency describes all completed request attempts, including failed ones;
    # QPS is likewise completed requests / elapsed wall-clock time.
    latency_ms = [latency * 1000 for latency in latencies]
    return {
        "queries": len(workload),
        "completed": len(latencies),
        "failed": len(errors),
        "concurrency": concurrency,
        "warmup_queries": warmup_queries,
        "elapsed_seconds": elapsed,
        "queries_per_second": len(latencies) / elapsed if elapsed else 0.0,
        "latency_ms": {
            "mean": mean(latency_ms) if latency_ms else None,
            "p50": _percentile(latency_ms, 0.50),
            "p95": _percentile(latency_ms, 0.95),
            "max": max(latency_ms) if latency_ms else None,
        },
        # Keep this bounded so an outage cannot make the JSON report enormous.
        "errors": errors[:10],
    }
