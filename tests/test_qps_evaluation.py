from __future__ import annotations

from backend.eval.throughput import run_qps_evaluation


def test_qps_evaluation_reports_completed_queries_and_latency() -> None:
    seen: list[str] = []

    def query(query: str) -> None:
        seen.append(query)

    report = run_qps_evaluation(["one", "two", "three"], query, concurrency=2, warmup_queries=2)

    assert seen[:2] == ["one", "two"]
    assert report["queries"] == 3
    assert report["completed"] == 3
    assert report["failed"] == 0
    assert report["queries_per_second"] > 0
    assert report["latency_ms"]["p50"] is not None


def test_qps_evaluation_counts_failures_without_stopping() -> None:
    def query(query: str) -> None:
        if query == "bad":
            raise RuntimeError("provider unavailable")

    report = run_qps_evaluation(["good", "bad"], query)

    assert report["completed"] == 2
    assert report["failed"] == 1
    assert report["errors"] == ["RuntimeError: provider unavailable"]
