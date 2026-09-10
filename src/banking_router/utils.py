"""Tiện ích dùng chung: log, seed, ghi JSON và metric."""

from __future__ import annotations

import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import Any
import numpy as np

LOGGER = logging.getLogger("banking_router")


def setup_logging() -> None:
    """Thiết lập stdout UTF-8 và một định dạng log thống nhất."""
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except (OSError, ValueError):
            pass

    level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def set_seed(seed: int = 42) -> None:
    """Cố định seed để các lần chạy có thể tái lập."""
    random.seed(seed)
    np.random.seed(seed)


def save_json(path: str | Path, payload: dict[str, Any] | list[Any]) -> None:
    """Ghi dictionary hoặc list thành JSON UTF-8 có thụt dòng."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def calculate_ece(
    confidences: np.ndarray,
    predictions: np.ndarray,
    targets: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Tính sai số hiệu chuẩn kỳ vọng (Expected Calibration Error)."""
    if len(confidences) == 0:
        return 0.0
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    total_samples = len(confidences)
    correct_mask = (predictions == targets).astype(float)

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        in_bin = (confidences >= bin_lower) & (confidences <= bin_upper) if i == n_bins - 1 else (confidences >= bin_lower) & (confidences < bin_upper)
        bin_size = np.sum(in_bin)
        if bin_size > 0:
            bin_acc = np.mean(correct_mask[in_bin])
            bin_conf = np.mean(confidences[in_bin])
            ece += (bin_size / total_samples) * abs(bin_acc - bin_conf)
    return float(ece)


def calculate_entropy(probabilities: np.ndarray, eps: float = 1e-12) -> np.ndarray | float:
    """Tính entropy Shannon cho mảng xác suất 1D hoặc 2D."""
    p = np.clip(probabilities, eps, 1.0)
    if p.ndim == 1:
        return float(-np.sum(p * np.log(p)))
    return -np.sum(p * np.log(p), axis=1)
