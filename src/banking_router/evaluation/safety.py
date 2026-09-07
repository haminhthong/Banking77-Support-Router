"""Operational triage safety and high-risk escalation metrics."""

from __future__ import annotations

from typing import Any, Sequence
import numpy as np
from ..data.contracts import DEFAULT_HIGH_RISK_INTENTS
from ..routing.schemas import RoutingDecision


def evaluate_operational_policy_metrics(
    decisions: Sequence[RoutingDecision],
    predictions: np.ndarray,
    targets: np.ndarray,
    high_risk_intents: frozenset[str] = DEFAULT_HIGH_RISK_INTENTS,
) -> dict[str, Any]:
    """Operational triage metrics evaluating the full decision system."""
    n_samples = len(targets)
    if n_samples == 0:
        return {}

    actions = np.array([d.action for d in decisions])
    auto_mask = actions == "auto_route"
    human_mask = actions == "human_review"
    priority_mask = actions == "priority_human_review"

    # Auto-route accuracy: correctness among those auto-routed
    auto_count = int(auto_mask.sum())
    auto_coverage = float(auto_count / n_samples)
    if auto_count > 0:
        auto_correct = predictions[auto_mask] == targets[auto_mask]
        auto_accuracy = float(auto_correct.mean())
        auto_error = float(1.0 - auto_accuracy)
    else:
        auto_accuracy = 0.0
        auto_error = 0.0

    # Human review rate
    human_count = int(human_mask.sum())
    human_rate = float(human_count / n_samples)

    # Priority escalation safety
    priority_count = int(priority_mask.sum())
    priority_rate = float(priority_count / n_samples)

    is_true_high_risk = np.isin(targets, list(high_risk_intents))
    true_high_risk_count = int(is_true_high_risk.sum())
    correctly_escalated = int((is_true_high_risk & priority_mask).sum())

    high_risk_recall = (
        float(correctly_escalated / true_high_risk_count)
        if true_high_risk_count > 0
        else 1.0
    )
    high_risk_precision = (
        float(correctly_escalated / priority_count)
        if priority_count > 0
        else 1.0
    )

    return {
        "operational_auto_route_count": auto_count,
        "operational_auto_route_coverage": round(auto_coverage, 4),
        "operational_auto_route_accuracy": round(auto_accuracy, 4),
        "operational_auto_route_error": round(auto_error, 4),
        "operational_human_review_count": human_count,
        "operational_human_review_rate": round(human_rate, 4),
        "operational_priority_escalation_count": priority_count,
        "operational_priority_escalation_rate": round(priority_rate, 4),
        "high_risk_true_samples": true_high_risk_count,
        "high_risk_correctly_escalated": correctly_escalated,
        "high_risk_escalation_recall": round(high_risk_recall, 4),
        "high_risk_escalation_precision": round(high_risk_precision, 4),
    }


def analyze_confusion_pairs(
    texts: list[str],
    true_labels: list[str],
    pred_labels: list[str],
    confidences: np.ndarray,
    margins: np.ndarray,
    top_n: int = 20,
) -> list[dict[str, Any]]:
    """Identify top confused intent pairs with representative customer queries."""
    pair_counts: dict[tuple[str, str], list[dict[str, Any]]] = {}

    for text, true_l, pred_l, conf, marg in zip(
        texts, true_labels, pred_labels, confidences, margins
    ):
        if true_l != pred_l:
            pair = (true_l, pred_l)
            if pair not in pair_counts:
                pair_counts[pair] = []
            pair_counts[pair].append({
                "text": text,
                "confidence": round(float(conf), 4),
                "margin": round(float(marg), 4),
            })

    sorted_pairs = sorted(
        pair_counts.items(), key=lambda item: len(item[1]), reverse=True
    )[:top_n]

    result = []
    for (true_l, pred_l), examples in sorted_pairs:
        result.append({
            "true_intent": true_l,
            "predicted_intent": pred_l,
            "error_count": len(examples),
            "avg_confidence": round(float(np.mean([e["confidence"] for e in examples])), 4),
            "avg_margin": round(float(np.mean([e["margin"] for e in examples])), 4),
            "example_queries": [e["text"] for e in examples[:3]],
        })
    return result
