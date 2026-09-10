"""Đánh giá phân loại intent và ánh xạ taxonomy domain."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from ..data.contracts import get_domain_for_intent


def evaluate_classification(
    targets: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    classes: np.ndarray,
) -> dict[str, Any]:
    """Tính metric phân loại intent và độ chính xác ở cấp domain."""
    accuracy = float(accuracy_score(targets, predictions))
    macro_f1 = float(f1_score(targets, predictions, average="macro"))

    # Độ chính xác top-3.
    ranked_indices = probabilities.argsort(axis=1)[:, ::-1]
    top3_classes = classes[ranked_indices[:, :3]]
    top3_correct = np.any(top3_classes == targets[:, None], axis=1)
    top3_accuracy = float(top3_correct.mean())

    # Độ chính xác sau khi quy về domain.
    true_domains = np.array([get_domain_for_intent(i) for i in targets])
    pred_domains = np.array([get_domain_for_intent(i) for i in predictions])
    domain_accuracy = float(accuracy_score(true_domains, pred_domains))

    return {
        "test_accuracy": round(accuracy, 4),
        "test_macro_f1": round(macro_f1, 4),
        "test_top3_accuracy": round(top3_accuracy, 4),
        "domain_taxonomy_accuracy": round(domain_accuracy, 4),
    }
