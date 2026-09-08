"""Metric an toàn theo queue và security risk thực tế."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from ..routing.schemas import RoutingDecision, RoutingResult
from ..routing.taxonomy import TaxonomyResolver


def evaluate_operational_policy_metrics(
    decisions: Sequence[RoutingDecision],
    predictions: np.ndarray,
    targets: np.ndarray,
    high_risk_intents: frozenset[str] | None = None,
    routing_results: Sequence[RoutingResult] | None = None,
    taxonomy: TaxonomyResolver | None = None,
) -> dict[str, Any]:
    """Đo wrong-queue, coverage và security recall của toàn bộ policy.

    ``predictions`` vẫn nhận để tương thích, nhưng auto-route correctness dùng
    queue đích khi có taxonomy/routing result; không đánh tráo queue bằng fine
    intent accuracy.
    """
    n_samples = len(targets)
    if n_samples == 0:
        return {}

    actions = np.array([decision.action for decision in decisions])
    auto_mask = actions == "auto_route"
    human_mask = actions == "human_review"
    priority_mask = actions == "priority_human_review"
    target_queues = (
        np.asarray([taxonomy.get_queue(str(intent)) for intent in targets])
        if taxonomy is not None
        else np.asarray(targets)
    )
    predicted_queues = np.asarray([decision.queue_id for decision in decisions])

    auto_count = int(auto_mask.sum())
    auto_coverage = float(auto_count / n_samples)
    if auto_count:
        queue_correct = predicted_queues[auto_mask] == target_queues[auto_mask]
        auto_queue_accuracy = float(queue_correct.mean())
        auto_queue_error = float(1.0 - auto_queue_accuracy)
        intent_correct = predictions[auto_mask] == targets[auto_mask]
        auto_intent_accuracy = float(intent_correct.mean())
    else:
        auto_queue_accuracy = auto_queue_error = auto_intent_accuracy = 0.0

    risk_intents = (
        taxonomy.get_critical_intents()
        if taxonomy is not None
        else frozenset(high_risk_intents or ())
    )
    true_risk = np.isin(targets, list(risk_intents))
    true_risk_count = int(true_risk.sum())
    correctly_escalated = int((true_risk & priority_mask).sum())
    priority_count = int(priority_mask.sum())
    recall = float(correctly_escalated / true_risk_count) if true_risk_count else 1.0
    precision = float(correctly_escalated / priority_count) if priority_count else 1.0

    result = {
        "operational_auto_route_count": auto_count,
        "operational_auto_route_coverage": round(auto_coverage, 4),
        "operational_auto_route_accuracy": round(auto_queue_accuracy, 4),
        "operational_auto_route_error": round(auto_queue_error, 4),
        "auto_route_queue_accuracy": round(auto_queue_accuracy, 4),
        "auto_route_wrong_queue_rate": round(auto_queue_error, 4),
        "auto_route_intent_accuracy": round(auto_intent_accuracy, 4),
        "operational_human_review_count": int(human_mask.sum()),
        "operational_human_review_rate": round(float(human_mask.mean()), 4),
        "operational_priority_escalation_count": priority_count,
        "operational_priority_escalation_rate": round(float(priority_mask.mean()), 4),
        "high_risk_true_samples": true_risk_count,
        "high_risk_correctly_escalated": correctly_escalated,
        "high_risk_escalation_recall": round(recall, 4),
        "high_risk_escalation_precision": round(precision, 4),
    }
    if routing_results is not None:
        result["risk_group_counts"] = {
            category: sum(1 for item in routing_results if item.risk.risk_category == category)
            for category in sorted({item.risk.risk_category for item in routing_results if item.risk.risk_category})
        }
    return result


def analyze_confusion_pairs(
    texts: list[str],
    true_labels: list[str],
    pred_labels: list[str],
    confidences: np.ndarray,
    margins: np.ndarray,
    top_n: int = 20,
) -> list[dict[str, Any]]:
    """Tìm cặp intent nhầm nhiều nhất kèm ví dụ đại diện."""
    pair_counts: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for text, true_label, pred_label, confidence, margin in zip(
        texts, true_labels, pred_labels, confidences, margins
    ):
        if true_label != pred_label:
            pair_counts.setdefault((true_label, pred_label), []).append({
                "text": text,
                "confidence": round(float(confidence), 4),
                "margin": round(float(margin), 4),
            })

    result = []
    for (true_label, predicted_label), examples in sorted(
        pair_counts.items(), key=lambda item: len(item[1]), reverse=True
    )[:top_n]:
        result.append({
            "true_intent": true_label,
            "predicted_intent": predicted_label,
            "error_count": len(examples),
            "avg_confidence": round(float(np.mean([item["confidence"] for item in examples])), 4),
            "avg_margin": round(float(np.mean([item["margin"] for item in examples])), 4),
            "example_queries": [item["text"] for item in examples[:3]],
        })
    return result
