"""Mô-đun đánh giá toàn diện AI Customer Support Router trên tập kiểm thử độc lập (Official Test).

Facade tương thích ngược giao tiếp với package banking_router.evaluation.
"""

from __future__ import annotations

import argparse
from typing import Any
from src.banking_router.evaluation.evaluator import evaluate_test_benchmark
from src.banking_router.utils import setup_logging


def evaluate_model(
    models_dir: str = "models",
    reports_dir: str = "reports",
    raw_dir: str = "data/raw",
    ood_file: str = "data/evaluation/ood.jsonl",
) -> dict[str, Any]:
    """Hàm đánh giá tương thích ngược."""
    setup_logging()
    return evaluate_test_benchmark(
        models_dir=models_dir,
        reports_dir=reports_dir,
        raw_dir=raw_dir,
        ood_file=ood_file,
    )


def main() -> None:
    """Chạy toàn bộ đánh giá benchmark đa tầng trên tập Test chính thức."""
    parser = argparse.ArgumentParser(description="Đánh giá release candidate Banking77 Router")
    parser.add_argument(
        "--models-dir",
        default="models/releases/banking-router-v6",
        help="Thư mục release candidate cần đánh giá",
    )
    parser.add_argument(
        "--reports-dir",
        default="reports/v6-evaluation",
        help="Thư mục lưu báo cáo đánh giá",
    )
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--ood-file", default="data/evaluation/ood.jsonl")
    args = parser.parse_args()
    metrics = evaluate_model(
        models_dir=args.models_dir,
        reports_dir=args.reports_dir,
        raw_dir=args.raw_dir,
        ood_file=args.ood_file,
    )
    print("\n=== BÁO CÁO KẾT QUẢ ĐÁNH GIÁ CANONICAL TRÊN TẬP TEST ĐỘC LẬP ===")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        elif not isinstance(v, (dict, list)):
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
