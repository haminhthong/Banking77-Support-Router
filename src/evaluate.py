"""CLI đánh giá Banking77 Support Router."""

from __future__ import annotations

import argparse
from typing import Any

from src.banking_router.evaluation.evaluator import evaluate_test_benchmark
from src.banking_router.utils import setup_logging


def evaluate_model(
    artifacts_dir: str = "artifacts",
    reports_dir: str = "reports/evaluation",
    raw_dir: str = "data/raw",
    ood_file: str = "data/evaluation/ood.jsonl",
) -> dict[str, Any]:
    setup_logging()
    return evaluate_test_benchmark(
        artifacts_dir=artifacts_dir,
        reports_dir=reports_dir,
        raw_dir=raw_dir,
        ood_file=ood_file,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Đánh giá Banking77 Support Router")
    parser.add_argument("--artifacts-dir", default="artifacts")
    parser.add_argument("--reports-dir", default="reports/evaluation")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--ood-file", default="data/evaluation/ood.jsonl")
    args = parser.parse_args()
    metrics = evaluate_model(
        artifacts_dir=args.artifacts_dir,
        reports_dir=args.reports_dir,
        raw_dir=args.raw_dir,
        ood_file=args.ood_file,
    )
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        elif not isinstance(value, (dict, list)):
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
