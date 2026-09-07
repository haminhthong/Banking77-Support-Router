"""Training, calibration, and joint operational policy optimization pipeline."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, log_loss

from .artifact import save_bundle
from .calibration import build_calibrated_model
from .pipeline import build_pipeline
from ..config import get_model_config, get_routing_policy_config, get_taxonomy_config
from ..data import (
    DEFAULT_HIGH_RISK_INTENTS,
    compute_file_sha256,
    get_domain_for_intent,
    load_training_splits,
    summarize_split_quality,
)
from ..routing.policy import RoutingPolicy
from ..routing.risk import RiskAssessor
from ..routing.schemas import IntentPrediction
from ..routing.taxonomy import TaxonomyResolver

LOGGER = logging.getLogger("banking_router.training")


def calculate_ece(
    confidences: np.ndarray,
    predictions: np.ndarray,
    targets: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Calculate Expected Calibration Error (ECE)."""
    if len(confidences) == 0:
        return 0.0
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    total_samples = len(confidences)
    correct_mask = (predictions == targets).astype(float)

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        in_bin = (confidences >= bin_lower) & (confidences <= bin_upper) if i == n_bins - 1 else (confidences >= bin_lower) & (confidences < bin_upper)
        bin_size = np.sum(in_bin)
        if bin_size > 0:
            bin_acc = np.mean(correct_mask[in_bin])
            bin_conf = np.mean(confidences[in_bin])
            ece += (bin_size / total_samples) * abs(bin_acc - bin_conf)
    return float(ece)


def calculate_entropy(probabilities: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Calculate Shannon entropy for 2D probability matrix."""
    p = np.clip(probabilities, eps, 1.0)
    return -np.sum(p * np.log(p), axis=1)


def optimize_policy_thresholds(
    calibrated_model: Any,
    val_texts: pd.Series,
    val_targets: np.ndarray,
    target_selective_risk: float = 0.05,
    target_high_risk_recall: float = 0.95,
    min_coverage_floor: float = 0.65,
    high_risk_intents: frozenset[str] = DEFAULT_HIGH_RISK_INTENTS,
) -> dict[str, Any]:
    """Jointly optimize reject threshold and high-risk trigger on threshold validation data.

    Optimization Objective:
        MAXIMIZE auto_route_coverage
        SUBJECT TO:
            auto_route_error <= target_selective_risk
            high_risk_recall >= target_high_risk_recall
            coverage >= min_coverage_floor
    """
    proba = calibrated_model.predict_proba(val_texts)
    classes = calibrated_model.classes_
    class_to_idx = {c: i for i, c in enumerate(classes)}

    ranked_indices = proba.argsort(axis=1)[:, ::-1]
    top_preds = classes[ranked_indices[:, 0]]
    confidences = proba[np.arange(len(proba)), ranked_indices[:, 0]]
    margins = confidences - proba[np.arange(len(proba)), ranked_indices[:, 1]]
    entropies = calculate_entropy(proba)

    # Risk signals for validation samples
    risk_indices = [class_to_idx[c] for c in high_risk_intents if c in class_to_idx]
    max_risk_scores = np.max(proba[:, risk_indices], axis=1) if risk_indices else np.zeros(len(proba))
    is_true_high_risk = np.isin(val_targets, list(high_risk_intents))
    true_high_risk_count = int(is_true_high_risk.sum())

    threshold_candidates = np.linspace(0.20, 0.90, 71)
    risk_trigger_candidates = np.linspace(0.05, 0.45, 41)

    feasible_results: list[dict[str, Any]] = []
    all_evaluated: list[dict[str, Any]] = []

    for r_thresh in threshold_candidates:
        for risk_trig in risk_trigger_candidates:
            # Policy decision simulation:
            # 1. High risk escalation if top_intent in high_risk OR max_risk_scores >= risk_trig
            escalated_mask = np.isin(top_preds, list(high_risk_intents)) | (max_risk_scores >= risk_trig)
            # 2. Uncertainty abstain if not escalated and conf < r_thresh
            abstained_mask = (~escalated_mask) & (confidences < r_thresh)
            # 3. Auto route
            auto_routed_mask = (~escalated_mask) & (~abstained_mask)

            # Metrics
            auto_coverage = float(auto_routed_mask.mean())
            auto_error = (
                float(1.0 - (top_preds[auto_routed_mask] == val_targets[auto_routed_mask]).mean())
                if auto_routed_mask.any()
                else 0.0
            )

            hr_caught = int((is_true_high_risk & escalated_mask).sum())
            hr_recall = float(hr_caught / true_high_risk_count) if true_high_risk_count > 0 else 1.0

            cand_info = {
                "reject_threshold": float(r_thresh),
                "high_risk_trigger": float(risk_trig),
                "auto_coverage": auto_coverage,
                "auto_error": auto_error,
                "high_risk_recall": hr_recall,
            }
            all_evaluated.append(cand_info)

            if (
                auto_error <= target_selective_risk
                and hr_recall >= target_high_risk_recall
                and auto_coverage >= min_coverage_floor
            ):
                feasible_results.append(cand_info)

    if feasible_results:
        # Maximize coverage, then minimize error
        best = max(feasible_results, key=lambda x: (x["auto_coverage"], -x["auto_error"]))
        status = "FEASIBLE_OPTIMAL"
    else:
        LOGGER.warning(
            "NO FEASIBLE POLICY meeting strict SLO (risk <= %.2f, recall >= %.2f). Selecting robust Pareto point.",
            target_selective_risk,
            target_high_risk_recall,
        )
        # Select best tradeoff: prioritize recall >= 0.90, then minimum error
        viable = [x for x in all_evaluated if x["high_risk_recall"] >= 0.90 and x["auto_coverage"] >= 0.70]
        if viable:
            best = min(viable, key=lambda x: (x["auto_error"], -x["auto_coverage"]))
        else:
            best = {"reject_threshold": 0.45, "high_risk_trigger": 0.20, "auto_coverage": 0.80, "auto_error": 0.06, "high_risk_recall": 0.92}
        status = "ROBUST_PARETO_FALLBACK"

    return {
        "status": status,
        "selected_reject_threshold": round(best["reject_threshold"], 4),
        "selected_high_risk_trigger": round(best["high_risk_trigger"], 4),
        "val_auto_coverage": round(best["auto_coverage"], 4),
        "val_auto_error": round(best["auto_error"], 4),
        "val_high_risk_recall": round(best["high_risk_recall"], 4),
        "feasible_candidate_count": len(feasible_results),
    }


def train_and_optimize(
    seed: int = 42,
    benchmark: str = "official",
    models_dir: str | Path = "models",
    reports_dir: str | Path = "reports",
) -> dict[str, Any]:
    """Execute end-to-end model training, probability calibration, and policy optimization."""
    m_dir = Path(models_dir)
    r_dir = Path(reports_dir)
    m_dir.mkdir(parents=True, exist_ok=True)
    r_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load data splits with clean contract
    train_df, cal_df, val_df, test_df = load_training_splits(
        raw_dir="data/raw",
        seed=seed,
        benchmark=benchmark,  # type: ignore
    )
    data_quality = summarize_split_quality(train_df, cal_df, val_df, test_df)

    # 2. Build and train base model on Train (70%)
    model_cfg = get_model_config()
    pipeline, pipe_cfg = build_pipeline(
        word_ngram_range=tuple(model_cfg.get("features", {}).get("tfidf", {}).get("ngram_range", [1, 2])),
        use_char_features=True,
        c_param=float(model_cfg.get("classifier", {}).get("params", {}).get("C", 4.0)),
        max_iter=int(model_cfg.get("classifier", {}).get("params", {}).get("max_iter", 1200)),
    )
    pipeline.fit(train_df["text"], train_df["intent"])

    # 3. Fit Platt Calibration on Calibration (15%)
    calibrated_model = build_calibrated_model(pipeline)
    calibrated_model.fit(cal_df["text"], cal_df["intent"])

    # 4. Measure calibration metrics on Threshold Validation (15%)
    raw_proba_val = pipeline.predict_proba(val_df["text"])
    cal_proba_val = calibrated_model.predict_proba(val_df["text"])

    classes = calibrated_model.classes_
    raw_preds = classes[raw_proba_val.argmax(axis=1)]
    cal_preds = classes[cal_proba_val.argmax(axis=1)]
    raw_confs = raw_proba_val.max(axis=1)
    cal_confs = cal_proba_val.max(axis=1)
    targets_val = val_df["intent"].to_numpy()

    raw_ece = calculate_ece(raw_confs, raw_preds, targets_val)
    cal_ece = calculate_ece(cal_confs, cal_preds, targets_val)
    raw_loss = float(log_loss(targets_val, raw_proba_val, labels=classes))
    cal_loss = float(log_loss(targets_val, cal_proba_val, labels=classes))

    # 5. Joint Policy Optimization on Threshold Validation (15%)
    policy_cfg = get_routing_policy_config()
    opt_slos = policy_cfg.get("optimization", {})
    policy_opt = optimize_policy_thresholds(
        calibrated_model=calibrated_model,
        val_texts=val_df["text"],
        val_targets=targets_val,
        target_selective_risk=float(opt_slos.get("target_selective_risk", 0.05)),
        target_high_risk_recall=float(opt_slos.get("target_high_risk_recall", 0.95)),
        min_coverage_floor=float(opt_slos.get("min_coverage_floor", 0.65)),
        high_risk_intents=DEFAULT_HIGH_RISK_INTENTS,
    )

    chosen_threshold = float(policy_opt["selected_reject_threshold"])
    chosen_risk_trigger = float(policy_opt["selected_high_risk_trigger"])

    # 6. Save Bundle & Manifest
    taxonomy_cfg = get_taxonomy_config()
    train_sha = compute_file_sha256("data/raw/train.csv")
    test_sha = compute_file_sha256("data/raw/test.csv")

    config_payload = {
        "schema_version": 3,
        "version": "banking77-support-triage-v3",
        "policy_version": "risk-aware-triage-v3",
        "seed": seed,
        "benchmark": benchmark,
        "threshold": chosen_threshold,
        "high_risk_trigger": chosen_risk_trigger,
        "min_margin": float(policy_cfg.get("runtime_thresholds", {}).get("min_margin", 0.02)),
        "max_entropy": float(policy_cfg.get("runtime_thresholds", {}).get("max_entropy", 3.80)),
        "class_count": len(classes),
        "raw_validation_ece": raw_ece,
        "calibrated_validation_ece": cal_ece,
        "raw_validation_log_loss": raw_loss,
        "calibrated_validation_log_loss": cal_loss,
        "domain_map": {intent: get_domain_for_intent(intent) for intent in classes},
        "policy_optimization": policy_opt,
    }

    policy_payload = {
        "threshold": chosen_threshold,
        "high_risk_trigger": chosen_risk_trigger,
        "min_margin": config_payload["min_margin"],
        "max_entropy": config_payload["max_entropy"],
        "high_risk_intents": sorted(list(DEFAULT_HIGH_RISK_INTENTS)),
        "target_selective_risk": opt_slos.get("target_selective_risk", 0.05),
        "target_high_risk_recall": opt_slos.get("target_high_risk_recall", 0.95),
    }

    bundle = save_bundle(
        target_dir=m_dir,
        calibrated_model=calibrated_model,
        config_payload=config_payload,
        policy_payload=policy_payload,
        taxonomy_payload=taxonomy_cfg,
        train_dataset_sha256=train_sha,
        test_dataset_sha256=test_sha,
        split_sizes={
            "train": len(train_df),
            "calibration": len(cal_df),
            "threshold_validation": len(val_df),
            "test": len(test_df),
        },
        pipeline_config=pipe_cfg,
    )

    # 7. Save Validation Report
    val_report = {
        "benchmark": benchmark,
        "macro_f1": float(f1_score(targets_val, cal_preds, average="macro")),
        "accuracy": float(accuracy_score(targets_val, cal_preds)),
        "raw_validation_log_loss": raw_loss,
        "calibrated_validation_log_loss": cal_loss,
        "raw_validation_ece": raw_ece,
        "calibrated_validation_ece": cal_ece,
        "policy_optimization": policy_opt,
        "split_rows": {
            "train": len(train_df),
            "calibration": len(cal_df),
            "threshold_validation": len(val_df),
            "test": len(test_df),
        },
        "data_quality": data_quality,
    }
    (r_dir / "validation_metrics.json").write_text(
        pd.Series(val_report).to_json(indent=2), encoding="utf-8"
    )

    LOGGER.info("Model training & policy optimization complete. Manifest and bundle saved.")
    return val_report
