"""Các thành phần public của package đánh giá."""

from .calibration import compute_multiclass_brier_score, evaluate_calibration
from .classification import evaluate_classification
from .evaluator import evaluate_test_benchmark
from .metrics import entropy, expected_calibration_error
from .ood_eval import evaluate_ood_benchmark
from .safety import analyze_confusion_pairs, evaluate_routing_metrics
from .selective import compute_risk_coverage_curve, evaluate_selective_metrics

__all__ = [
    "analyze_confusion_pairs",
    "compute_multiclass_brier_score",
    "compute_risk_coverage_curve",
    "entropy",
    "evaluate_calibration",
    "evaluate_classification",
    "evaluate_ood_benchmark",
    "evaluate_routing_metrics",
    "evaluate_selective_metrics",
    "evaluate_test_benchmark",
    "expected_calibration_error",
]
