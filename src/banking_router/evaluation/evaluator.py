"""Đánh giá classification, calibration, selective routing và scope smoke set."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from ..config import ARTIFACTS_DIR
from ..data.loader import load_official_test
from ..modeling.artifact import load_artifacts
from ..routing.escalation import SensitiveIntentGuard
from ..routing.policy import RoutingPolicy
from ..routing.scope import ScopeGuard
from ..routing.service import RoutingService
from ..routing.taxonomy import TaxonomyResolver
from .calibration import evaluate_calibration
from .classification import evaluate_classification
from .ood_eval import evaluate_ood_benchmark
from .safety import analyze_confusion_pairs, evaluate_routing_metrics
from .selective import compute_risk_coverage_curve, evaluate_selective_metrics


def evaluate_test_benchmark(
    artifacts_dir: str | Path = ARTIFACTS_DIR,
    reports_dir: str | Path = "reports/evaluation",
    raw_dir: str | Path = "data/raw",
    ood_file: str | Path = "data/evaluation/ood.jsonl",
) -> dict[str, Any]:
    """Đánh giá trên test Banking77 chính thức chưa dùng để tuning."""
    artifact_bundle = load_artifacts(artifacts_dir)
    report_dir = Path(reports_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    test_df = load_official_test(raw_dir)
    model = artifact_bundle.intent_model
    policy_cfg = artifact_bundle.routing_policy
    runtime = policy_cfg.get("runtime_thresholds", {})
    sensitive_cfg = policy_cfg.get("sensitive_case", {})
    scope_cfg = policy_cfg.get("scope", {})
    taxonomy = TaxonomyResolver(artifact_bundle.taxonomy)
    queue_threshold = float(runtime.get("queue_probability", 0.45))
    sensitive_threshold = float(runtime.get("sensitive_probability", 0.20))
    policy = RoutingPolicy(
        queue_threshold=queue_threshold,
        queue_margin=float(runtime.get("queue_margin", 0.0)),
        min_margin=float(runtime.get("min_margin", 0.02)),
        max_entropy=float(runtime.get("max_entropy", 3.80)),
        minimum_sensitive_signal=float(
            sensitive_cfg.get("minimum_signal_probability", 0.16)
        ),
        minimum_scope_sensitive_signal=float(
            sensitive_cfg.get("minimum_scope_signal", 0.30)
        ),
        taxonomy_resolver=taxonomy,
    )
    sensitive_guard = SensitiveIntentGuard(
        model.classes_, taxonomy=taxonomy, sensitive_trigger=sensitive_threshold
    )
    scope_guard = ScopeGuard(
        min_chars=int(scope_cfg.get("min_chars", 4)),
        min_tokens=int(scope_cfg.get("min_tokens", 2)),
        lexical_similarity_threshold=float(
            scope_cfg.get("lexical_similarity_threshold", 0.08)
        ),
        low_confidence_threshold=float(scope_cfg.get("low_confidence_threshold", 0.22)),
        unsupported_threshold=float(scope_cfg.get("unsupported_threshold", 0.80)),
    )
    routing_service = RoutingService(
        model=model,
        policy=policy,
        sensitive_guard=sensitive_guard,
        scope_guard=scope_guard,
        scope_model=artifact_bundle.scope_model,
        taxonomy=taxonomy,
        metadata=dict(artifact_bundle.metadata),
    )

    probabilities = model.predict_proba(test_df["text"])
    classes = model.classes_
    ranked = probabilities.argsort(axis=1)[:, ::-1]
    predictions = classes[ranked[:, 0]]
    confidence = probabilities[np.arange(len(test_df)), ranked[:, 0]]
    margin = confidence - probabilities[np.arange(len(test_df)), ranked[:, 1]]
    targets = test_df["intent"].to_numpy()
    correct = predictions == targets
    classification = evaluate_classification(
        targets, predictions, probabilities, classes
    )
    calibration = evaluate_calibration(
        confidence, predictions, targets, probabilities, classes
    )
    selective = evaluate_selective_metrics(confidence, correct, queue_threshold)
    risk_coverage = compute_risk_coverage_curve(confidence, correct)
    results = routing_service.route_batch(test_df["text"].tolist())
    routing_metrics = evaluate_routing_metrics(
        decisions=[item.decision for item in results],
        predictions=predictions,
        targets=targets,
        routing_results=results,
        taxonomy=taxonomy,
    )
    confusion_pairs = analyze_confusion_pairs(
        texts=test_df["text"].tolist(),
        true_labels=targets.tolist(),
        pred_labels=predictions.tolist(),
        confidences=confidence,
        margins=margin,
        top_n=20,
    )
    scope_metrics = evaluate_ood_benchmark(ood_file, routing_service, split="locked")

    report = {
        "benchmark": "official_untouched",
        "total_test_samples": len(test_df),
        "queue_threshold": queue_threshold,
        "sensitive_threshold": sensitive_threshold,
        **classification,
        **calibration,
        **selective,
        "aurc": risk_coverage["aurc"],
        "coverage_at_5pct_risk": risk_coverage["coverage_at_5pct_risk"],
        "coverage_at_3pct_risk": risk_coverage["coverage_at_3pct_risk"],
        **routing_metrics,
        "scope_smoke": scope_metrics,
        "auto_route_coverage": routing_metrics.get(
            "operational_auto_route_coverage", 0.0
        ),
        "auto_route_accuracy": routing_metrics.get(
            "operational_auto_route_accuracy", 0.0
        ),
        "sensitive_case_recall": routing_metrics.get("sensitive_case_recall", 1.0),
    }
    (report_dir / "test_metrics.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (report_dir / "risk_coverage_curve.json").write_text(
        json.dumps(risk_coverage, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (report_dir / "sensitive_case_metrics.json").write_text(
        json.dumps(
            {
                "sensitive_intents": sorted(taxonomy.get_sensitive_intents()),
                "sensitive_threshold": sensitive_threshold,
                "true_samples": routing_metrics.get("sensitive_case_true_samples", 0),
                "correctly_reviewed": routing_metrics.get(
                    "sensitive_case_correctly_reviewed", 0
                ),
                "recall": routing_metrics.get("sensitive_case_recall", 1.0),
                "precision": routing_metrics.get("sensitive_case_precision", 1.0),
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    (report_dir / "confusion_pairs.json").write_text(
        json.dumps(confusion_pairs, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (report_dir / "scope_metrics.json").write_text(
        json.dumps(scope_metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return report
