"""DocMind command line utilities."""
from __future__ import annotations
import argparse
import json


def main() -> None:
    # DeepEval uses Rich progress output; force UTF-8 for Windows consoles.
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(prog="docmind")
    sub = parser.add_subparsers(dest="command", required=True)
    evaluation = sub.add_parser("eval", help="run the RAG evaluation pipeline")
    evaluation.add_argument("--mode", choices=("beir", "synthetic", "both"), default="synthetic")
    evaluation.add_argument("--max-samples", type=int, default=50)
    evaluation.add_argument(
        "--qps", action="store_true",
        help="benchmark end-to-end queries/second using the selected evaluation inputs",
    )
    evaluation.add_argument("--qps-concurrency", type=int, default=1)
    evaluation.add_argument("--qps-warmup-queries", type=int, default=0)
    query = sub.add_parser("query", help="ask DocMind")
    query.add_argument("question")
    query.add_argument("--mode", choices=("classic", "agentic"), default="classic")
    query.add_argument("--top-k", type=int, default=5)
    query.add_argument("--max-iterations", type=int, default=4)
    args = parser.parse_args()
    if args.command == "eval":
        from backend.eval.pipeline import run_evaluation
        report = run_evaluation(
            args.mode,
            args.max_samples,
            include_qps=args.qps,
            qps_concurrency=args.qps_concurrency,
            qps_warmup_queries=args.qps_warmup_queries,
        )
        print(f"Mode: {report.get('mode')} | samples: {report.get('samples')} | k: {report.get('k')}")
        print(f"{'Metric':<30} {'Score':>8} {'Passed':>8} {'Threshold':>10}")
        print("-" * 62)
        for name, item in report.get("metrics", {}).items():
            score = "n/a" if item["score"] is None else f"{item['score']:.3f}"
            print(f"{name:<30} {score:>8} {str(item['passed']):>8} {item['threshold']:>10.2f}")
        if "qps" in report:
            qps = report["qps"]
            print(
                "\nQPS: "
                f"{qps['queries_per_second']:.2f} "
                f"({qps['completed']} completed, {qps['failed']} failed, "
                f"{qps['concurrency']} concurrent)"
            )
        print("\nJSON report:")
        print(json.dumps(report, indent=2))
    if args.command == "query":
        if args.mode == "agentic":
            from dataclasses import asdict
            from app.agent import answer_agentically
            from app.backend import kb
            print(json.dumps(asdict(answer_agentically(args.question, kb, args.max_iterations)), indent=2))
        else:
            from app.backend import kb
            from app.generation import answer_question
            sources = kb.retrieve(args.question, args.top_k)
            print(json.dumps(answer_question(args.question, sources) if sources else {"answer": "I do not know.", "citations": []}, indent=2))


if __name__ == "__main__":
    main()
