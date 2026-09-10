"""Đánh giá hiệu chuẩn xác suất bằng ECE, Brier đa lớp và log-loss."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import log_loss

from .metrics import expected_calibration_error


def compute_multiclass_brier_score(
    probabilities: np.ndarray, targets: np.ndarray, classes: np.ndarray
) -> float:
    """Tính Brier đa lớp = mean(sum((p_c - y_c)^2))."""
    class_to_idx = {c: i for i, c in enumerate(classes)}
    n_samples = len(targets)
    n_classes = len(classes)
    y_one_hot = np.zeros((n_samples, n_classes), dtype=float)

    for i, target in enumerate(targets):
        if target in class_to_idx:
            y_one_hot[i, class_to_idx[target]] = 1.0

    brier = np.mean(np.sum((probabilities - y_one_hot) ** 2, axis=1))
    return float(brier)


def evaluate_calibration(
    confidences: np.ndarray,
    predictions: np.ndarray,
    targets: np.ndarray,
    probabilities: np.ndarray,
    classes: np.ndarray,
) -> dict[str, Any]:
    """Tính các metric hiệu chuẩn trên dữ liệu test."""
    ece = expected_calibration_error(confidences, predictions, targets)
    loss = float(log_loss(targets, probabilities, labels=classes))
    brier = compute_multiclass_brier_score(probabilities, targets, classes)

    return {
        "test_ece": round(ece, 4),
        "test_log_loss": round(loss, 4),
        "test_brier_score": round(brier, 4),
    }
