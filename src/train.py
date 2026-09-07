"""Mô-đun huấn luyện pipeline phân loại ý định khách hàng và hiệu chỉnh xác suất (Probability Calibration).

Facade tương thích ngược giao tiếp với package banking_router.modeling.
"""

from __future__ import annotations

import argparse
import numpy as np

from src.banking_router.modeling.training import (
    calculate_ece,
    optimize_policy_thresholds,
    train_and_optimize,
)
from src.banking_router.utils import set_seed, setup_logging


def select_reject_threshold(
    confidence: np.ndarray,
    correct: np.ndarray,
    minimum_coverage: float = 0.80,
) -> float:
    """Hàm chọn ngưỡng tương thích ngược cho các unit test cũ."""
    if len(confidence) == 0:
        return 0.45

    candidates = np.linspace(0.2, 0.95, 151)
    for coverage_floor in (minimum_coverage, 0.50, 0.25):
        feasible = []
        for threshold in candidates:
            accepted = confidence >= threshold
            coverage = float(accepted.mean())
            if coverage >= coverage_floor and accepted.any():
                risk = float(1.0 - correct[accepted].mean())
                feasible.append((risk, -coverage, float(threshold)))
        if feasible:
            return min(feasible)[2]
    return 0.45


def main() -> None:
    """Chạy toàn diện quy trình huấn luyện, hiệu chỉnh và tối ưu hóa chính sách."""
    setup_logging()
    set_seed(42)

    parser = argparse.ArgumentParser(description="Train Banking77 Support Router")
    parser.add_argument(
        "--benchmark",
        choices=["official", "strict_decontaminated"],
        default="official",
        help="Chọn chuẩn đánh giá: official (chuẩn công bố) hoặc strict_decontaminated (khử trùng lặp tuyệt đối)",
    )
    args = parser.parse_args()

    train_and_optimize(seed=42, benchmark=args.benchmark)


if __name__ == "__main__":
    main()
