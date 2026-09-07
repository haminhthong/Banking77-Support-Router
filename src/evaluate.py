"""Mô-đun đánh giá toàn diện AI Customer Support Router trên tập kiểm thử độc lập (Official Test).

Facade tương thích ngược giao tiếp với package banking_router.evaluation.
"""

from __future__ import annotations

from typing import Any
from src.banking_router.evaluation.evaluator import evaluate_test_benchmark
from src.banking_router.utils import setup_logging


def evaluate_model() -> dict[str, Any]:
    """Hàm đánh giá tương thích ngược."""
    setup_logging()
    return evaluate_test_benchmark()


def main() -> None:
    """Chạy toàn bộ đánh giá benchmark đa tầng trên tập Test chính thức."""
    metrics = evaluate_model()
    print("\n=== BÁO CÁO KẾT QUẢ ĐÁNH GIÁ CANONICAL TRÊN TẬP TEST ĐỘC LẬP ===")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        elif not isinstance(v, (dict, list)):
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
