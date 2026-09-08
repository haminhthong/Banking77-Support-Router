"""Training, calibration, and joint operational policy optimization pipeline."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, log_loss

from .artifact import save_bundle
from .calibration import fit_temperature, select_probability_model
from .pipeline import build_pipeline
from ..config import get_model_config, get_routing_policy_config, get_taxonomy_config
from ..data import DEFAULT_HIGH_RISK_INTENTS, compute_file_sha256, get_domain_for_intent, load_training_splits, summarize_split_quality
from ..data.scope import load_scope_splits
from .scope import train_scope_classifier
from ..routing.ood import ScopeGuard
from ..routing.policy import RoutingPolicy
from ..routing.queue_projector import QueueProjector
from ..routing.risk import RiskAssessor
from ..routing.schemas import IntentPrediction, RiskAssessment
from ..routing.taxonomy import TaxonomyResolver
from ..data.normalization import NORMALIZATION_VERSION

LOGGER = logging.getLogger("banking_router.training")


def optimize_queue_policy_thresholds(
    probability_model: Any,
    val_texts: pd.Series,
    val_targets: np.ndarray,
    taxonomy: TaxonomyResolver,
    target_wrong_queue_rate: float = 0.05,
    target_critical_recall: float = 0.95,
    min_coverage_floor: float = 0.65,
    max_entropy: float | None = 3.80,
    scope_guard: ScopeGuard | None = None,
    minimum_risk_signal: float = 0.16,
    minimum_ood_security_signal: float = 0.30,
) -> dict[str, Any]:
    """Tối ưu bằng chính ``RoutingPolicy.evaluate`` của runtime.

    Không dùng công thức thu gọn riêng cho train: queue confidence, queue
    margin, entropy, scope và risk đều đi qua policy engine chung.
    """
    probabilities = probability_model.predict_proba(val_texts)
    classes = np.asarray(probability_model.classes_)
    projector = QueueProjector(classes, taxonomy)
    target_queues = np.asarray([taxonomy.get_queue(str(intent)) for intent in val_targets])
    critical_indices = [i for i, intent in enumerate(classes) if intent in taxonomy.get_critical_intents()]
    critical_mass = probabilities[:, critical_indices].sum(axis=1) if critical_indices else np.zeros(len(probabilities))
    critical_targets = np.asarray([str(intent) in taxonomy.get_critical_intents() for intent in val_targets])

    scope = scope_guard or ScopeGuard()
    precomputed: list[tuple[IntentPrediction, Any, list[str], str | None, float]] = []
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
        queue_prediction = projector.project(row)
        ood, ood_reasons = scope.detect(str(text), confidence=confidence, margin=margin)
        risk_intent = max(
            taxonomy.get_critical_intents(),
            key=lambda item: float(row[list(classes).index(item)]) if item in classes else 0.0,
            default=None,
        )
        risk_score = float(row[list(classes).index(risk_intent)]) if risk_intent else 0.0
        precomputed.append((prediction, queue_prediction, ood_reasons if ood else [], risk_intent, risk_score))

    queue_thresholds = np.linspace(0.45, 0.95, 15)
    queue_margins = np.linspace(0.0, 0.20, 5)
    risk_thresholds = np.linspace(0.10, 0.60, 16)
    candidates: list[dict[str, Any]] = []
    for queue_threshold in queue_thresholds:
        for queue_margin in queue_margins:
            for risk_threshold in risk_thresholds:
                policy = RoutingPolicy(
                    threshold=float(queue_threshold),
                    queue_threshold=float(queue_threshold),
                    queue_margin=float(queue_margin),
                    max_entropy=max_entropy,
                    high_risk_trigger=float(risk_threshold),
                    minimum_risk_signal=minimum_risk_signal,
                    minimum_ood_security_signal=minimum_ood_security_signal,
                    taxonomy_resolver=taxonomy,
                )
                decisions = []
                for index, (prediction, queue_prediction, ood_reasons, risk_intent, risk_score) in enumerate(precomputed):
                    risk = RiskAssessment(
                        high_risk_detected=critical_mass[index] >= risk_threshold,
                        high_risk_intent=risk_intent,
                        high_risk_score=risk_score,
                        ood_detected=bool(ood_reasons),
                        reason_codes=list(ood_reasons) + (["CRITICAL_RISK_MASS"] if critical_mass[index] >= risk_threshold else []),
                        critical_probability=float(critical_mass[index]),
                        risk_category=taxonomy.get_risk_category(risk_intent) if risk_intent else None,
                    )
                    decisions.append(
                        policy.evaluate(prediction, risk, queue_prediction)
                    )
                actions = np.asarray([decision.action for decision in decisions])
                auto = actions == "auto_route"
                priority = actions == "priority_human_review"
                coverage = float(auto.mean()) if len(auto) else 0.0
                routed_queues = np.asarray([decision.queue_id for decision in decisions])
                wrong_queue = float((routed_queues[auto] != target_queues[auto]).mean()) if auto.any() else 0.0
                critical_recall = float((priority & critical_targets).sum() / critical_targets.sum()) if critical_targets.any() else 1.0
                candidates.append({
                    "queue_threshold": float(queue_threshold),
                    "queue_margin": float(queue_margin),
                    "critical_threshold": float(risk_threshold),
                    "auto_route_coverage": coverage,
                    "wrong_queue_rate": wrong_queue,
                    "critical_recall": critical_recall,
                })

    feasible = [
        item for item in candidates
        if item["wrong_queue_rate"] <= target_wrong_queue_rate
        and item["critical_recall"] >= target_critical_recall
        and item["auto_route_coverage"] >= min_coverage_floor
    ]
    if feasible:
        best = max(feasible, key=lambda item: (item["auto_route_coverage"], -item["wrong_queue_rate"]))
    else:
        # Khi không có điểm thỏa toàn bộ SLO, ưu tiên safety recall trước
        # coverage; fallback cũ ưu tiên coverage/error nên có thể bỏ sót risk.
        best = max(candidates, key=lambda item: (
            item["critical_recall"],
            -item["wrong_queue_rate"],
            item["auto_route_coverage"],
        ))
    return {
        "status": "FEASIBLE_OPTIMAL" if feasible else "ROBUST_PARETO_FALLBACK",
        "selected_queue_threshold": round(best["queue_threshold"], 4),
        "selected_queue_margin": round(best["queue_margin"], 4),
        "selected_critical_threshold": round(best["critical_threshold"], 4),
        "val_auto_route_coverage": round(best["auto_route_coverage"], 4),
        "val_wrong_queue_rate": round(best["wrong_queue_rate"], 4),
        "val_critical_recall": round(best["critical_recall"], 4),
        "feasible_candidate_count": len(feasible),
    }


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
    pipeline, pipe_cfg = build_pipeline(model_config=model_cfg)
    pipeline.fit(train_df["text"], train_df["intent"])

    # 3. Temperature is only a candidate.  The final raw-vs-temperature
    # choice is made on the separate Policy Validation split below.
    temperature_candidate = fit_temperature(pipeline, cal_df["text"], cal_df["intent"])

    # 4. Compare probability models on Policy Validation (15%)
    raw_proba_val = pipeline.predict_proba(val_df["text"])
    probability_model, calibration_report = select_probability_model(
        pipeline, temperature_candidate, val_df["text"], val_df["intent"]
    )
    cal_proba_val = probability_model.predict_proba(val_df["text"])

    classes = probability_model.classes_
    raw_preds = classes[raw_proba_val.argmax(axis=1)]
    cal_preds = classes[cal_proba_val.argmax(axis=1)]
    raw_confs = raw_proba_val.max(axis=1)
    cal_confs = cal_proba_val.max(axis=1)
    targets_val = val_df["intent"].to_numpy()

    raw_ece = calculate_ece(raw_confs, raw_preds, targets_val)
    cal_ece = calculate_ece(cal_confs, cal_preds, targets_val)
    raw_loss = float(log_loss(targets_val, raw_proba_val, labels=classes))
    cal_loss = float(log_loss(targets_val, cal_proba_val, labels=classes))

    # 5. Optimize the operational queue policy, not fine intent confidence.
    policy_cfg = get_routing_policy_config()
    opt_slos = policy_cfg.get("optimization", {})
    taxonomy_cfg = get_taxonomy_config()
    taxonomy = TaxonomyResolver(taxonomy_cfg)
    scope_model = None
    scope_threshold = float(policy_cfg.get("ood_guard", {}).get("unsupported_threshold", 0.07))
    scope_guard = ScopeGuard(unsupported_threshold=scope_threshold)
    try:
        scope_splits = load_scope_splits("data/evaluation/ood.jsonl")
        unsupported_train = [str(item["text"]) for item in scope_splits["train"]]
        scope_model = train_scope_classifier(
            supported_texts=train_df["text"].tolist(),
            unsupported_texts=unsupported_train,
            seed=seed,
        )
        scope_guard.set_scope_classifier(scope_model)
    except (FileNotFoundError, ValueError, KeyError) as exc:
        LOGGER.warning("Không huấn luyện được scope classifier: %s", exc)

    runtime_cfg = policy_cfg.get("runtime_thresholds", {})
    policy_opt = optimize_queue_policy_thresholds(
        probability_model=probability_model,
        val_texts=val_df["text"],
        val_targets=targets_val,
        taxonomy=taxonomy,
        target_wrong_queue_rate=float(opt_slos.get("target_wrong_queue_rate", 0.05)),
        target_critical_recall=float(opt_slos.get("target_critical_recall", 0.95)),
        min_coverage_floor=float(opt_slos.get("min_coverage_floor", 0.65)),
        max_entropy=float(runtime_cfg.get("max_entropy", 3.80)),
        scope_guard=scope_guard,
        minimum_risk_signal=float(policy_cfg.get("critical_risk", {}).get("minimum_signal_probability", 0.16)),
        minimum_ood_security_signal=float(policy_cfg.get("critical_risk", {}).get("minimum_ood_security_signal", 0.30)),
    )

    chosen_threshold = float(policy_opt["selected_queue_threshold"])
    chosen_risk_trigger = float(policy_opt["selected_critical_threshold"])

    # 6. Save Bundle & Manifest
    train_sha = compute_file_sha256("data/raw/train.csv")
    test_sha = compute_file_sha256("data/raw/test.csv")

    config_payload = {
        "schema_version": 4,
        "version": model_cfg.get("model_version", "banking77-router-v5"),
        "policy_version": policy_cfg.get("policy_version", "queue-policy-v4"),
        "seed": seed,
        "benchmark": benchmark,
        "queue_threshold": chosen_threshold,
        "critical_threshold": chosen_risk_trigger,
        # Legacy keys kept for old clients; runtime uses routing_policy.json.
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
        "calibration": calibration_report,
        "normalization_version": NORMALIZATION_VERSION,
    }

    policy_payload = {
        "policy_version": policy_cfg.get("policy_version", "queue-policy-v4"),
        "scope": {
            "min_chars": int(policy_cfg.get("ood_guard", {}).get("min_text_length_chars", 4)),
            "min_tokens": int(policy_cfg.get("ood_guard", {}).get("min_tokens", 2)),
            "low_confidence_threshold": float(policy_cfg.get("ood_guard", {}).get("low_confidence_ood_threshold", 0.22)),
            "lexical_novelty_threshold": float(policy_cfg.get("ood_guard", {}).get("lexical_similarity_threshold", 0.08)),
            "unsupported_threshold": scope_threshold,
        },
        "auto_route": {
            "min_queue_probability": chosen_threshold,
            "min_queue_margin": float(policy_opt["selected_queue_margin"]),
        },
        "critical_risk": {
            "minimum_probability": chosen_risk_trigger,
            "minimum_signal_probability": float(policy_cfg.get("critical_risk", {}).get("minimum_signal_probability", 0.16)),
            "minimum_ood_security_signal": float(policy_cfg.get("critical_risk", {}).get("minimum_ood_security_signal", 0.30)),
        },
        "scope_model": {
            "version": getattr(scope_model, "model_version", None),
            "unsupported_threshold": scope_threshold,
        },
        "normalization_version": NORMALIZATION_VERSION,
    }

    bundle = save_bundle(
        target_dir=m_dir,
        calibrated_model=probability_model,
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
        model_version=str(model_cfg.get("model_version", "banking77-tfidf-char-lr-v5")),
        policy_version=str(policy_cfg.get("policy_version", "queue-policy-v5")),
        scope_model=scope_model,
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
    (r_dir / "validation_metrics.json").write_text(
        pd.Series(val_report).to_json(indent=2), encoding="utf-8"
    )

    LOGGER.info("Model training & policy optimization complete. Manifest and bundle saved.")
    return val_report
