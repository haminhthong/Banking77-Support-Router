"""Selective classification metrics and Risk-Coverage curve analysis."""

from __future__ import annotations

from typing import Any
import numpy as np


def compute_risk_coverage_curve(
    confidences: np.ndarray, corrects: np.ndarray, num_points: int = 100
) -> dict[str, Any]:
    """Compute Risk-Coverage Curve and calculate Area Under Risk-Coverage Curve (AURC)."""
    thresholds = np.linspace(0.0, 1.0, num_points)
    curve_points: list[dict[str, float]] = []

    covs = []
    risks = []

    for t in thresholds:
        accepted = confidences >= t
        coverage = float(accepted.mean())
        risk = float(1.0 - corrects[accepted].mean()) if accepted.any() else 0.0
        curve_points.append({
            "threshold": round(float(t), 4),
            "coverage": round(coverage, 4),
            "risk": round(risk, 4),
        })
        covs.append(coverage)
        risks.append(risk)

    # Sort by coverage ascending for trapezoidal integration
    sorted_pairs = sorted(zip(covs, risks), key=lambda p: p[0])
    sorted_covs = np.array([p[0] for p in sorted_pairs])
    sorted_risks = np.array([p[1] for p in sorted_pairs])
    aurc = (
        float(np.trapezoid(sorted_risks, sorted_covs))
        if hasattr(np, "trapezoid")
        else float(np.trapz(sorted_risks, sorted_covs))
    )

    cov_at_5pct = 0.0
    cov_at_3pct = 0.0
    for pt in sorted(curve_points, key=lambda x: -x["coverage"]):
        if pt["risk"] <= 0.05 and pt["coverage"] > cov_at_5pct:
            cov_at_5pct = pt["coverage"]
        if pt["risk"] <= 0.03 and pt["coverage"] > cov_at_3pct:
            cov_at_3pct = pt["coverage"]

    return {
        "aurc": round(aurc, 4),
        "coverage_at_5pct_risk": round(cov_at_5pct, 4),
        "coverage_at_3pct_risk": round(cov_at_3pct, 4),
        "curve": curve_points,
    }


def evaluate_selective_metrics(
    confidences: np.ndarray,
    corrects: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    """Model-level selective classification metrics strictly based on confidence threshold."""
    accepted = confidences >= threshold
    coverage = float(accepted.mean())
    risk = float(1.0 - corrects[accepted].mean()) if accepted.any() else 0.0
    accuracy = float(corrects[accepted].mean()) if accepted.any() else 0.0

    return {
        "classifier_acceptance_coverage": round(coverage, 4),
        "classifier_accepted_accuracy": round(accuracy, 4),
        "classifier_selective_risk": round(risk, 4),
        "classifier_accepted_count": int(accepted.sum()),
    }
