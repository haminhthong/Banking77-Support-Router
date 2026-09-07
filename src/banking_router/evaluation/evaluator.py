"""Comprehensive evaluation runner for Official and Strict benchmarks, Selective Routing, Safety, and OOD."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report

from .calibration import evaluate_calibration
from .classification import evaluate_classification
from .ood_eval import evaluate_ood_benchmark
from .safety import analyze_confusion_pairs, evaluate_operational_policy_metrics
from .selective import compute_risk_coverage_curve, evaluate_selective_metrics
from ..data import (
    DEFAULT_HIGH_RISK_INTENTS,
    load_official_test,
)
from ..modeling.artifact import load_and_validate_bundle
from ..routing.ood import OODGuard
from ..routing.policy import RoutingPolicy
from ..routing.risk import RiskAssessor
from ..routing.service import RoutingService
from ..routing.taxonomy import TaxonomyResolver

LOGGER = logging.getLogger("banking_router.evaluation")


def evaluate_test_benchmark(
    models_dir: str | Path = "models",
    reports_dir: str | Path = "reports",
    raw_dir: str | Path = "data/raw",
    ood_file: str | Path = "data/evaluation/ood.jsonl",
) -> dict[str, Any]:
    """Run rigorous multi-layer evaluation on the untouched Official Test Benchmark."""
    m_dir = Path(models_dir)
    r_dir = Path(reports_dir)
    r_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load validated ModelBundle and immutable official test set (3,080 samples)
    bundle = load_and_validate_bundle(m_dir, verify_checksum=False)
    test_df = load_official_test(raw_dir)
    model = bundle.model
    cfg = bundle.config

    threshold = float(cfg.get("threshold", 0.45))
    high_risk_trigger = float(cfg.get("high_risk_trigger", 0.20))
    min_margin = cfg.get("min_margin")
    max_entropy = cfg.get("max_entropy")

    taxonomy = TaxonomyResolver()
    policy = RoutingPolicy(
        threshold=threshold,
        high_risk_trigger=high_risk_trigger,
        min_margin=min_margin,
        max_entropy=max_entropy,
        taxonomy_resolver=taxonomy,
    )
    risk_assessor = RiskAssessor(
        classes=model.classes_,
        high_risk_intents=DEFAULT_HIGH_RISK_INTENTS,
        high_risk_trigger=high_risk_trigger,
    )
    ood_guard = OODGuard()

    routing_service = RoutingService(
        model=model,
        policy=policy,
        risk_assessor=risk_assessor,
        ood_guard=ood_guard,
        taxonomy=taxonomy,
        metadata={"model_version": cfg.get("version", "v3")},
    )

    # 2. Vectorized inference on official test
    proba = model.predict_proba(test_df["text"])
    classes = model.classes_
    ranked_indices = proba.argsort(axis=1)[:, ::-1]

    pred = classes[ranked_indices[:, 0]]
    confidence = proba[np.arange(len(test_df)), ranked_indices[:, 0]]
    second_confidence = proba[np.arange(len(test_df)), ranked_indices[:, 1]]
    margin = confidence - second_confidence
    targets = test_df["intent"].to_numpy()
    correct = pred == targets

    # 3. Layer 1: Classification Metrics
    cls_metrics = evaluate_classification(
        targets=targets,
        predictions=pred,
        probabilities=proba,
        classes=classes,
    )

    # 4. Layer 2: Calibration Metrics
    cal_metrics = evaluate_calibration(
        confidences=confidence,
        predictions=pred,
        targets=targets,
        probabilities=proba,
        classes=classes,
    )

    # 5. Layer 3: Model Selective Classification Metrics
    sel_metrics = evaluate_selective_metrics(
        confidences=confidence,
        corrects=correct,
        threshold=threshold,
    )

    # 6. Layer 4: Risk-Coverage Curve & AURC
    rc_curve = compute_risk_coverage_curve(confidence, correct)

    # 7. Layer 5: Operational Routing Policy Evaluation
    routing_results = routing_service.route_batch(test_df["text"].tolist())
    decisions = [r.decision for r in routing_results]
    op_metrics = evaluate_operational_policy_metrics(
        decisions=decisions,
        predictions=pred,
        targets=targets,
        high_risk_intents=DEFAULT_HIGH_RISK_INTENTS,
    )

    # 8. Layer 6: Confusion Pairs Analysis
    confusion_pairs = analyze_confusion_pairs(
        texts=test_df["text"].tolist(),
        true_labels=targets.tolist(),
        pred_labels=pred.tolist(),
        confidences=confidence,
        margins=margin,
        top_n=20,
    )

    # 9. Layer 7: OOD Benchmark Evaluation
    ood_metrics = evaluate_ood_benchmark(
        ood_file=ood_file,
        routing_service=routing_service,
    )

    # Combine canonical metrics payload
    full_report = {
        "benchmark": "official_untouched",
        "total_test_samples": len(test_df),
        "reject_threshold": threshold,
        "high_risk_trigger": high_risk_trigger,
        # Classification
        **cls_metrics,
        # Calibration
        **cal_metrics,
        # Model selective
        **sel_metrics,
        # Risk-coverage curve
        "aurc": rc_curve["aurc"],
        "coverage_at_5pct_risk": rc_curve["coverage_at_5pct_risk"],
        "coverage_at_3pct_risk": rc_curve["coverage_at_3pct_risk"],
        # Operational policy metrics
        **op_metrics,
        # Backward-compatibility aliases for legacy tools/scripts:
        "selective_coverage": sel_metrics["classifier_acceptance_coverage"],
        "selective_risk": sel_metrics["classifier_selective_risk"],
        "accepted_accuracy": sel_metrics["classifier_accepted_accuracy"],
        "accepted_samples": sel_metrics["classifier_accepted_count"],
        "auto_routed_samples": op_metrics.get("operational_auto_route_count", 0),
        "priority_escalated_samples": op_metrics.get("operational_priority_escalation_count", 0),
        "abstained_samples": op_metrics.get("operational_human_review_count", 0),
        "high_risk_escalation_recall": op_metrics.get("high_risk_escalation_recall", 1.0),
        "high_risk_escalation_precision": op_metrics.get("high_risk_escalation_precision", 1.0),
    }

    # Write canonical report artifacts
    (r_dir / "test_metrics.json").write_text(
        json.dumps(full_report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (r_dir / "risk_coverage_curve.json").write_text(
        json.dumps(rc_curve, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (r_dir / "high_risk_metrics.json").write_text(
        json.dumps({
            "high_risk_intents": sorted(list(DEFAULT_HIGH_RISK_INTENTS)),
            "high_risk_trigger": high_risk_trigger,
            "true_high_risk_count": op_metrics.get("high_risk_true_samples", 0),
            "correctly_escalated": op_metrics.get("high_risk_correctly_escalated", 0),
            "escalation_recall": op_metrics.get("high_risk_escalation_recall", 1.0),
            "escalation_precision": op_metrics.get("high_risk_escalation_precision", 1.0),
        }, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (r_dir / "confusion_pairs.json").write_text(
        json.dumps(confusion_pairs, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (r_dir / "ood_metrics.json").write_text(
        json.dumps(ood_metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    return full_report
