"""Các thành phần public của package đánh giá."""

from .calibration import compute_multiclass_brier_score, evaluate_calibration
from .classification import evaluate_classification
from .evaluator import evaluate_test_benchmark
from .ood_eval import evaluate_ood_benchmark
from .safety import analyze_confusion_pairs, evaluate_routing_metrics
from .selective import compute_risk_coverage_curve, evaluate_selective_metrics

__all__ = [
    "evaluate_classification",
    "compute_multiclass_brier_score",
    "evaluate_calibration",
    "compute_risk_coverage_curve",
    "evaluate_selective_metrics",
    "evaluate_routing_metrics",
    "analyze_confusion_pairs",
    "evaluate_ood_benchmark",
    "evaluate_test_benchmark",
]
