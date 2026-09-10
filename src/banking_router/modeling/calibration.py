"""Hiệu chuẩn xác suất đa lớp bằng temperature scaling tùy chọn."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import log_loss

from ..evaluation.metrics import expected_calibration_error


def _softmax(logits: np.ndarray, temperature: float) -> np.ndarray:
    scaled = logits / max(float(temperature), 1e-6)
    scaled -= scaled.max(axis=1, keepdims=True)
    exp_values = np.exp(scaled)
    return exp_values / exp_values.sum(axis=1, keepdims=True)


class TemperatureScaledModel:
    """Wrapper nhỏ áp dụng một nhiệt độ dương lên logits của model."""

    def __init__(self, base_model: Any, temperature: float = 1.0) -> None:
        self.base_model = base_model
        # Cung cấp thuộc tính estimator để serving có thể đọc vocabulary
        # mà không cần biết model đang được bọc bởi wrapper.
        self.estimator = base_model
        self.temperature = float(temperature)
        self.classes_ = base_model.classes_

    def _logits(self, texts: Any) -> np.ndarray:
        logits = np.asarray(self.base_model.decision_function(texts), dtype=float)
        if logits.ndim == 1:
            logits = np.column_stack([-logits, logits])
        return logits

    def predict_proba(self, texts: Any) -> np.ndarray:
        return _softmax(self._logits(texts), self.temperature)

    def predict(self, texts: Any) -> np.ndarray:
        probabilities = self.predict_proba(texts)
        return self.classes_[probabilities.argmax(axis=1)]


def fit_temperature(
    base_model: Any, texts: Any, targets: Any
) -> TemperatureScaledModel:
    """Ước lượng một nhiệt độ dương trên calibration split tách riêng."""
    logits = np.asarray(base_model.decision_function(texts), dtype=float)
    if logits.ndim == 1:
        logits = np.column_stack([-logits, logits])
    classes = np.asarray(base_model.classes_)
    target_indices = np.asarray(
        [int(np.where(classes == target)[0][0]) for target in targets]
    )

    # Tìm kiếm một chiều trong khoảng giới hạn, phù hợp với 77 lớp và không
    # tạo thêm một model có một tham số riêng cho từng lớp.
    candidates = np.exp(np.linspace(np.log(0.25), np.log(4.0), 160))
    losses = []
    for temperature in candidates:
        probabilities = _softmax(logits, float(temperature))
        losses.append(
            float(
                -np.mean(
                    np.log(
                        np.clip(
                            probabilities[
                                np.arange(len(target_indices)), target_indices
                            ],
                            1e-12,
                            1.0,
                        )
                    )
                )
            )
        )
    best_temperature = float(candidates[int(np.argmin(losses))])
    return TemperatureScaledModel(base_model, temperature=best_temperature)


def select_probability_model(
    raw_model: Any,
    calibrated_candidate: Any,
    texts: Any,
    targets: Any,
) -> tuple[Any, dict[str, Any]]:
    """Chọn xác suất raw hoặc đã hiệu chuẩn trên validation split."""
    classes = np.asarray(raw_model.classes_)
    target_indices = np.asarray(
        [int(np.where(classes == target)[0][0]) for target in targets]
    )

    def metrics(model: Any) -> dict[str, float]:
        probabilities = np.asarray(model.predict_proba(texts), dtype=float)
        confidence = probabilities.max(axis=1)
        predictions = classes[probabilities.argmax(axis=1)]
        targets_array = np.asarray(targets)
        ece = expected_calibration_error(confidence, predictions, targets_array)
        one_hot = np.zeros_like(probabilities)
        one_hot[np.arange(len(target_indices)), target_indices] = 1.0
        brier = float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))
        return {
            "nll": float(log_loss(targets, probabilities, labels=classes)),
            "ece": ece,
            "brier": brier,
        }

    raw_metrics = metrics(raw_model)
    candidate_metrics = metrics(calibrated_candidate)
    use_candidate = candidate_metrics["nll"] < raw_metrics["nll"]
    return (
        calibrated_candidate if use_candidate else raw_model,
        {
            "method": "temperature" if use_candidate else "none",
            "raw": raw_metrics,
            "temperature_candidate": candidate_metrics,
            "selected": "temperature" if use_candidate else "raw",
            "temperature": getattr(calibrated_candidate, "temperature", 1.0),
        },
    )
