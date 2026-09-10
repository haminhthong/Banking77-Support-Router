"""Các metric dùng chung giữa train, đánh giá và serving."""

from __future__ import annotations

import numpy as np


def expected_calibration_error(
    confidences: np.ndarray,
    predictions: np.ndarray,
    targets: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Tính sai số hiệu chuẩn kỳ vọng theo các khoảng confidence."""
    if len(confidences) == 0:
        return 0.0

    boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    correct = (predictions == targets).astype(float)
    ece = 0.0

    for index in range(n_bins):
        lower = boundaries[index]
        upper = boundaries[index + 1]
        if index == n_bins - 1:
            mask = (confidences >= lower) & (confidences <= upper)
        else:
            mask = (confidences >= lower) & (confidences < upper)
        if mask.any():
            ece += float(mask.mean()) * abs(
                float(correct[mask].mean()) - float(confidences[mask].mean())
            )

    return float(ece)


def entropy(
    probabilities: np.ndarray,
    eps: float = 1e-12,
) -> float | np.ndarray:
    """Tính entropy Shannon cho vector hoặc ma trận xác suất."""
    values = np.clip(np.asarray(probabilities, dtype=float), eps, 1.0)
    if values.ndim == 1:
        return float(-np.sum(values * np.log(values)))
    return -np.sum(values * np.log(values), axis=1)
