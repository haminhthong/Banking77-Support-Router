"""Huấn luyện, calibration và chọn threshold routing trên validation split."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, log_loss

from .artifact import save_artifacts
from .calibration import fit_temperature, select_probability_model
from .pipeline import build_pipeline
from ..config import get_model_config, get_routing_policy_config, get_taxonomy_config
from ..data import load_training_splits, summarize_split_quality
from ..data.normalization import NORMALIZATION_VERSION
from ..data.scope import load_scope_splits
from ..routing.escalation import SensitiveIntentGuard
from ..routing.policy import RoutingPolicy
from ..routing.queue_projector import QueueProjector
from ..routing.schemas import IntentPrediction
from ..routing.scope import ScopeGuard
from ..routing.taxonomy import TaxonomyResolver
from .scope import train_scope_classifier

LOGGER = logging.getLogger("banking_router.training")


def calculate_ece(confidences: np.ndarray, predictions: np.ndarray, targets: np.ndarray, n_bins: int = 10) -> float:
    """Tính Expected Calibration Error."""
    if len(confidences) == 0:
        return 0.0
    boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    correct = (predictions == targets).astype(float)
    ece = 0.0
    for index in range(n_bins):
        lower, upper = boundaries[index], boundaries[index + 1]
        in_bin = (confidences >= lower) & ((confidences <= upper) if index == n_bins - 1 else (confidences < upper))
        if in_bin.any():
            ece += float(in_bin.mean()) * abs(float(correct[in_bin].mean()) - float(confidences[in_bin].mean()))
    return float(ece)


def calculate_entropy(probabilities: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    values = np.clip(probabilities, eps, 1.0)
    return -np.sum(values * np.log(values), axis=1)


def optimize_queue_policy_thresholds(
    probability_model: Any,
    val_texts: pd.Series,
    val_targets: np.ndarray,
    taxonomy: TaxonomyResolver,
    target_wrong_queue_rate: float = 0.05,
    target_sensitive_case_recall: float = 0.95,
    min_coverage_floor: float = 0.65,
    max_entropy: float | None = 3.80,
    scope_guard: ScopeGuard | None = None,
    minimum_sensitive_signal: float = 0.16,
    minimum_scope_sensitive_signal: float = 0.30,
) -> dict[str, Any]:
    """Chọn threshold bằng chính policy được dùng khi serving."""
    probabilities = probability_model.predict_proba(val_texts)
    classes = np.asarray(probability_model.classes_)
    projector = QueueProjector(classes, taxonomy)
    target_queues = np.asarray([taxonomy.get_queue(str(intent)) for intent in val_targets])
    sensitive_targets = np.asarray([str(intent) in taxonomy.get_sensitive_intents() for intent in val_targets])
    scope = scope_guard or ScopeGuard()
    precomputed: list[tuple[IntentPrediction, Any, bool, list[str], np.ndarray]] = []
    for row, text in zip(probabilities, val_texts):
        ranked = row.argsort()[::-1]
        confidence = float(row[ranked[0]])
        margin = float(row[ranked[0]] - row[ranked[1]]) if len(ranked) > 1 else 1.0
        intent = str(classes[ranked[0]])
        prediction = IntentPrediction(
            intent=intent,
            domain=taxonomy.get_domain(intent),
            confidence=confidence,
            margin=margin,
            entropy=float(calculate_entropy(np.asarray([row]))[0]),
        )
        scope_detected, scope_reasons = scope.detect(str(text), confidence, margin)
        precomputed.append((prediction, projector.project(row), scope_detected, scope_reasons, row))

    queue_thresholds = np.linspace(0.45, 0.95, 15)
    queue_margins = np.linspace(0.0, 0.20, 5)
    sensitive_thresholds = np.linspace(0.10, 0.60, 16)
    candidates: list[dict[str, Any]] = []
    for queue_threshold in queue_thresholds:
        for queue_margin in queue_margins:
            for sensitive_threshold in sensitive_thresholds:
                guard = SensitiveIntentGuard(classes, taxonomy=taxonomy, sensitive_trigger=float(sensitive_threshold))
                policy = RoutingPolicy(
                    queue_threshold=float(queue_threshold),
                    queue_margin=float(queue_margin),
                    max_entropy=max_entropy,
                    sensitive_trigger=float(sensitive_threshold),
                    minimum_sensitive_signal=minimum_sensitive_signal,
                    minimum_scope_sensitive_signal=minimum_scope_sensitive_signal,
                    taxonomy_resolver=taxonomy,
                )
                decisions = []
                for prediction, queue_prediction, scope_detected, scope_reasons, row in precomputed:
                    sensitive_case = guard.assess(row, scope_detected, scope_reasons)
                    decisions.append(policy.evaluate(
                        prediction,
                        sensitive_case,
                        scope_detected=scope_detected,
                        scope_reasons=scope_reasons,
                        queue_prediction=queue_prediction,
                    ))
                actions = np.asarray([decision.action for decision in decisions])
                auto = actions == "auto_route"
                priority = actions == "priority_human_review"
                coverage = float(auto.mean()) if len(auto) else 0.0
                routed_queues = np.asarray([decision.queue_id for decision in decisions])
                wrong_queue = float((routed_queues[auto] != target_queues[auto]).mean()) if auto.any() else 0.0
                sensitive_recall = float((priority & sensitive_targets).sum() / sensitive_targets.sum()) if sensitive_targets.any() else 1.0
                candidates.append({
                    "queue_threshold": float(queue_threshold),
                    "queue_margin": float(queue_margin),
                    "sensitive_threshold": float(sensitive_threshold),
                    "auto_route_coverage": coverage,
                    "wrong_queue_rate": wrong_queue,
                    "sensitive_case_recall": sensitive_recall,
                })

    feasible = [
        item for item in candidates
        if item["wrong_queue_rate"] <= target_wrong_queue_rate
        and item["sensitive_case_recall"] >= target_sensitive_case_recall
        and item["auto_route_coverage"] >= min_coverage_floor
    ]
    best = max(
        feasible or candidates,
        key=lambda item: (
            item["auto_route_coverage"],
            item["sensitive_case_recall"],
            -item["wrong_queue_rate"],
        ),
    )
    return {
        "status": "TARGETS_MET" if feasible else "BEST_VALIDATION_TRADEOFF",
        "selected_queue_threshold": round(best["queue_threshold"], 4),
        "selected_queue_margin": round(best["queue_margin"], 4),
        "selected_sensitive_threshold": round(best["sensitive_threshold"], 4),
        "val_auto_route_coverage": round(best["auto_route_coverage"], 4),
        "val_wrong_queue_rate": round(best["wrong_queue_rate"], 4),
        "val_sensitive_case_recall": round(best["sensitive_case_recall"], 4),
        "candidate_count": len(feasible),
    }


def train_and_optimize(
    seed: int = 42,
    benchmark: str = "official",
    artifacts_dir: str | Path = "artifacts",
    reports_dir: str | Path = "reports/training",
) -> dict[str, Any]:
    """Train model, chọn threshold và ghi một bộ artifact canonical."""
    artifact_path = Path(artifacts_dir)
    report_path = Path(reports_dir)
    artifact_path.mkdir(parents=True, exist_ok=True)
    report_path.mkdir(parents=True, exist_ok=True)

    train_df, cal_df, val_df, test_df = load_training_splits(raw_dir="data/raw", seed=seed, benchmark=benchmark)  # type: ignore[arg-type]
    data_quality = summarize_split_quality(train_df, cal_df, val_df, test_df)
    model_cfg = get_model_config()
    pipeline, _ = build_pipeline(model_config=model_cfg)
    pipeline.fit(train_df["text"], train_df["intent"])
    temperature_candidate = fit_temperature(pipeline, cal_df["text"], cal_df["intent"])
    probability_model, calibration_report = select_probability_model(
        pipeline, temperature_candidate, val_df["text"], val_df["intent"]
    )
    raw_proba = pipeline.predict_proba(val_df["text"])
    calibrated_proba = probability_model.predict_proba(val_df["text"])
    classes = probability_model.classes_
    targets_val = val_df["intent"].to_numpy()
    raw_preds = classes[raw_proba.argmax(axis=1)]
    calibrated_preds = classes[calibrated_proba.argmax(axis=1)]
    raw_ece = calculate_ece(raw_proba.max(axis=1), raw_preds, targets_val)
    calibrated_ece = calculate_ece(calibrated_proba.max(axis=1), calibrated_preds, targets_val)
    raw_loss = float(log_loss(targets_val, raw_proba, labels=classes))
    calibrated_loss = float(log_loss(targets_val, calibrated_proba, labels=classes))

    policy_cfg = get_routing_policy_config()
    taxonomy_cfg = get_taxonomy_config()
    taxonomy = TaxonomyResolver(taxonomy_cfg)
    scope_cfg = policy_cfg.get("scope", {})
    scope_threshold = float(scope_cfg.get("unsupported_threshold", 0.80))
    scope_guard = ScopeGuard(
        min_chars=int(scope_cfg.get("min_chars", 4)),
        min_tokens=int(scope_cfg.get("min_tokens", 2)),
        lexical_similarity_threshold=float(scope_cfg.get("lexical_similarity_threshold", 0.08)),
        low_confidence_threshold=float(scope_cfg.get("low_confidence_threshold", 0.22)),
        unsupported_threshold=scope_threshold,
    )
    scope_model = None
    try:
        scope_splits = load_scope_splits("data/evaluation/ood.jsonl")
        scope_model = train_scope_classifier(
            supported_texts=train_df["text"].tolist(),
            unsupported_texts=[str(item["text"]) for item in scope_splits["train"]],
            seed=seed,
        )
        scope_guard.set_scope_classifier(scope_model)
    except (FileNotFoundError, ValueError, KeyError) as exc:
        LOGGER.warning("Không huấn luyện được scope classifier: %s", exc)

    runtime_cfg = policy_cfg.get("runtime_thresholds", {})
    optimization_cfg = policy_cfg.get("optimization", {})
    sensitive_cfg = policy_cfg.get("sensitive_case", {})
    policy_opt = optimize_queue_policy_thresholds(
        probability_model=probability_model,
        val_texts=val_df["text"],
        val_targets=targets_val,
        taxonomy=taxonomy,
        target_wrong_queue_rate=float(optimization_cfg.get("target_wrong_queue_rate", 0.05)),
        target_sensitive_case_recall=float(optimization_cfg.get("target_sensitive_case_recall", 0.95)),
        min_coverage_floor=float(optimization_cfg.get("min_coverage_floor", 0.65)),
        max_entropy=float(runtime_cfg.get("max_entropy", 3.80)),
        scope_guard=scope_guard,
        minimum_sensitive_signal=float(sensitive_cfg.get("minimum_signal_probability", 0.16)),
        minimum_scope_sensitive_signal=float(sensitive_cfg.get("minimum_scope_signal", 0.30)),
    )

    chosen_queue_threshold = float(policy_opt["selected_queue_threshold"])
    chosen_sensitive_threshold = float(policy_opt["selected_sensitive_threshold"])
    policy_payload = {
        "runtime_thresholds": {
            "queue_probability": chosen_queue_threshold,
            "queue_margin": float(policy_opt["selected_queue_margin"]),
            "sensitive_probability": chosen_sensitive_threshold,
            "min_margin": float(runtime_cfg.get("min_margin", 0.02)),
            "max_entropy": float(runtime_cfg.get("max_entropy", 3.80)),
        },
        "sensitive_case": {
            "minimum_signal_probability": float(sensitive_cfg.get("minimum_signal_probability", 0.16)),
            "minimum_scope_signal": float(sensitive_cfg.get("minimum_scope_signal", 0.30)),
        },
        "scope": {
            "min_chars": int(scope_cfg.get("min_chars", 4)),
            "min_tokens": int(scope_cfg.get("min_tokens", 2)),
            "lexical_similarity_threshold": float(scope_cfg.get("lexical_similarity_threshold", 0.08)),
            "low_confidence_threshold": float(scope_cfg.get("low_confidence_threshold", 0.22)),
            "unsupported_threshold": scope_threshold,
        },
    }
    metadata = {
        "model": "word-char-tfidf-logistic-regression",
        "classes": len(classes),
        "calibration": calibration_report.get("selected_method", "temperature_or_raw"),
        "dataset": "Banking77",
        "test_samples": len(test_df),
        "normalization": NORMALIZATION_VERSION,
        "benchmark": benchmark,
        "thresholds": policy_opt,
        "validation": {
            "raw_ece": raw_ece,
            "calibrated_ece": calibrated_ece,
            "raw_log_loss": raw_loss,
            "calibrated_log_loss": calibrated_loss,
        },
    }
    save_artifacts(
        artifacts_dir=artifact_path,
        intent_model=probability_model,
        scope_model=scope_model,
        metadata=metadata,
        taxonomy=taxonomy_cfg,
        routing_policy=policy_payload,
    )

    report = {
        "benchmark": benchmark,
        "macro_f1": float(f1_score(targets_val, calibrated_preds, average="macro")),
        "accuracy": float(accuracy_score(targets_val, calibrated_preds)),
        "raw_validation_log_loss": raw_loss,
        "calibrated_validation_log_loss": calibrated_loss,
        "raw_validation_ece": raw_ece,
        "calibrated_validation_ece": calibrated_ece,
        "calibration": calibration_report,
        "policy_optimization": policy_opt,
        "split_rows": {
            "train": len(train_df),
            "calibration": len(cal_df),
            "threshold_validation": len(val_df),
            "test": len(test_df),
        },
        "data_quality": data_quality,
    }
    (report_path / "validation_metrics.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    LOGGER.info("Đã lưu artifact canonical và báo cáo validation.")
    return report
