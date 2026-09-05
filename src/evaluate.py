"""Mô-đun đánh giá toàn diện AI Customer Support Router trên tập kiểm thử độc lập (Official Test).

Đánh giá đa tầng theo chuẩn AI Engineering:
1. Intent Classifier: Accuracy, Macro-F1, Top-3 Accuracy, Per-class F1.
2. Calibration: Expected Calibration Error (ECE), Multi-class Brier Score, Log-loss.
3. Post-hoc Taxonomy Projection: Domain-level Accuracy (10 miền nghiệp vụ).
4. Selective Routing: Coverage, Selective Risk, Accepted Accuracy.
5. Risk-Coverage Curve & AURC: Diện tích dưới đường cong rủi ro-độ phủ, Coverage@5% Risk, Coverage@3% Risk.
6. High-Risk Safety: Escalation Recall & Escalation Precision cho các ca nhạy cảm.
7. Error Analysis: Top-20 Confusion Pairs kèm ví dụ thực tế và độ chênh margin.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, f1_score, log_loss

from .data import get_domain_for_intent, load_test_split
from .policy import DEFAULT_HIGH_RISK_INTENTS, RoutingPolicy
from .utils import calculate_ece, calculate_entropy, save_json, setup_logging


def compute_multiclass_brier_score(
    probabilities: np.ndarray, targets: np.ndarray, classes: np.ndarray
) -> float:
    """Tính toán Multi-class Brier Score = mean(sum((p_c - y_c)^2))."""
    class_to_idx = {c: i for i, c in enumerate(classes)}
    n_samples = len(targets)
    n_classes = len(classes)
    y_one_hot = np.zeros((n_samples, n_classes), dtype=float)

    for i, target in enumerate(targets):
        if target in class_to_idx:
            y_one_hot[i, class_to_idx[target]] = 1.0

    brier = np.mean(np.sum((probabilities - y_one_hot) ** 2, axis=1))
    return float(brier)


def compute_risk_coverage_curve(
    confidences: np.ndarray, corrects: np.ndarray, num_points: int = 100
) -> dict[str, Any]:
    """Xây dựng Risk-Coverage curve và tính AURC (Area Under Risk-Coverage Curve)."""
    thresholds = np.linspace(0.0, 1.0, num_points)
    curve_points: list[dict[str, float]] = []

    covs = []
    risks = []

    for t in thresholds:
        accepted = confidences >= t
        coverage = float(accepted.mean())
        risk = float(1.0 - corrects[accepted].mean()) if accepted.any() else 0.0
        curve_points.append(
            {"threshold": round(float(t), 4), "coverage": round(coverage, 4), "risk": round(risk, 4)}
        )
        covs.append(coverage)
        risks.append(risk)

    # Tính AURC (Area Under the Risk-Coverage Curve)
    # Sắp xếp theo coverage tăng dần để tích phân trapezoidal
    sorted_pairs = sorted(zip(covs, risks), key=lambda p: p[0])
    sorted_covs = np.array([p[0] for p in sorted_pairs])
    sorted_risks = np.array([p[1] for p in sorted_pairs])
    aurc = float(np.trapezoid(sorted_risks, sorted_covs)) if hasattr(np, "trapezoid") else float(np.trapz(sorted_risks, sorted_covs))

    # Tìm Coverage tối đa tại các mốc rủi ro mục tiêu
    cov_at_5pct = 0.0
    cov_at_3pct = 0.0
    for pt in sorted(curve_points, key=lambda x: -x["coverage"]):
        if pt["risk"] <= 0.05 and pt["coverage"] > cov_at_5pct:
            cov_at_5pct = pt["coverage"]
        if pt["risk"] <= 0.03 and pt["coverage"] > cov_at_3pct:
            cov_at_3pct = pt["coverage"]

    return {
        "aurc": aurc,
        "coverage_at_5pct_risk": cov_at_5pct,
        "coverage_at_3pct_risk": cov_at_3pct,
        "curve": curve_points,
    }


def analyze_confusion_pairs(
    texts: list[str],
    true_labels: list[str],
    pred_labels: list[str],
    confidences: np.ndarray,
    margins: np.ndarray,
    top_n: int = 20,
) -> list[dict[str, Any]]:
    """Tìm top các cặp intent thường xuyên bị nhầm lẫn kèm ví dụ đại diện."""
    pair_counts: dict[tuple[str, str], list[dict[str, Any]]] = {}

    for text, true_l, pred_l, conf, marg in zip(
        texts, true_labels, pred_labels, confidences, margins
    ):
        if true_l != pred_l:
            pair = (true_l, pred_l)
            if pair not in pair_counts:
                pair_counts[pair] = []
            pair_counts[pair].append(
                {
                    "text": text,
                    "confidence": round(float(conf), 4),
                    "margin": round(float(marg), 4),
                }
            )

    sorted_pairs = sorted(pair_counts.items(), key=lambda item: len(item[1]), reverse=True)[
        :top_n
    ]

    result = []
    for (true_l, pred_l), examples in sorted_pairs:
        result.append(
            {
                "true_intent": true_l,
                "predicted_intent": pred_l,
                "error_count": len(examples),
                "avg_confidence": round(float(np.mean([e["confidence"] for e in examples])), 4),
                "avg_margin": round(float(np.mean([e["margin"] for e in examples])), 4),
                "example_queries": [e["text"] for e in examples[:3]],
            }
        )
    return result


def evaluate_model() -> dict[str, Any]:
    """Thực hiện đánh giá toàn diện mô hình trên tập Test độc lập."""
    setup_logging()
    te = load_test_split()

    model_path = Path("models/router.joblib")
    config_path = Path("models/config.json")

    if not model_path.exists() or not config_path.exists():
        raise FileNotFoundError(
            "Không tìm thấy mô hình hoặc config! Vui lòng chạy python -m src.train trước."
        )

    model = joblib.load(model_path)
    cfg = json.loads(config_path.read_text(encoding="utf-8"))

    threshold = float(cfg["threshold"])
    high_risk_trigger = float(cfg.get("high_risk_trigger", 0.20))
    policy = RoutingPolicy(
        threshold=threshold,
        high_risk_trigger=high_risk_trigger,
        min_margin=cfg.get("min_margin"),
        max_entropy=cfg.get("max_entropy"),
    )

    # 1. Dự đoán trên tập Test
    proba = model.predict_proba(te.text)
    classes = model.classes_
    ranked_indices = proba.argsort(axis=1)[:, ::-1]

    pred = classes[ranked_indices[:, 0]]
    confidence = proba[np.arange(len(te)), ranked_indices[:, 0]]

    # Tính margin = p1 - p2
    second_confidence = proba[np.arange(len(te)), ranked_indices[:, 1]]
    margin = confidence - second_confidence
    entropy = calculate_entropy(proba)

    # 2. Phân loại ý định (Intent Layer)
    targets = te.intent.to_numpy()
    correct = pred == targets
    test_accuracy = float(accuracy_score(targets, pred))
    test_macro_f1 = float(f1_score(targets, pred, average="macro"))

    # Top-3 Accuracy
    top3_classes = classes[ranked_indices[:, :3]]
    top3_correct = np.any(top3_classes == targets[:, None], axis=1)
    test_top3_accuracy = float(top3_correct.mean())

    # 3. Đánh giá Hiệu chỉnh xác suất (Calibration Layer)
    test_ece = calculate_ece(confidence, pred, targets)
    test_log_loss = float(log_loss(targets, proba, labels=classes))
    test_brier_score = compute_multiclass_brier_score(proba, targets, classes)

    # 4. Post-hoc Hierarchical Taxonomy Projection (Domain Level)
    true_domains = np.array([get_domain_for_intent(i) for i in targets])
    pred_domains = np.array([get_domain_for_intent(i) for i in pred])
    domain_accuracy = float(accuracy_score(true_domains, pred_domains))

    # 5. Đánh giá Phân loại có chọn lọc (Selective Routing)
    accepted = confidence >= threshold
    selective_coverage = float(accepted.mean())
    selective_risk = float(1.0 - correct[accepted].mean()) if accepted.any() else 0.0
    accepted_accuracy = float(correct[accepted].mean()) if accepted.any() else 0.0

    # 6. Đường cong Risk-Coverage và AURC
    rc_results = compute_risk_coverage_curve(confidence, correct)

    # 7. Đánh giá An toàn Rủi ro cao (High-Risk Escalation Safety)
    is_true_high_risk = np.isin(targets, list(DEFAULT_HIGH_RISK_INTENTS))

    # Chạy Routing Policy cho toàn bộ tập Test
    decisions = []
    for i in range(len(te)):
        top_k_cands = [
            (str(classes[ranked_indices[i, k]]), float(proba[i, ranked_indices[i, k]]))
            for k in range(min(3, len(classes)))
        ]
        dec = policy.decide(
            top_intent=str(pred[i]),
            confidence=float(confidence[i]),
            top_domain=str(pred_domains[i]),
            top_k_candidates=top_k_cands,
            margin=float(margin[i]),
            entropy=float(entropy[i]),
        )
        decisions.append(dec)

    escalated_mask = np.array([d.route == "priority_human_review" for d in decisions])
    abstained_mask = np.array([d.abstained for d in decisions])
    auto_routed_mask = np.array([d.decision == "auto_route" for d in decisions])

    true_high_risk_count = int(is_true_high_risk.sum())
    escalated_count = int(escalated_mask.sum())
    correctly_escalated = int((is_true_high_risk & escalated_mask).sum())

    escalation_recall = (
        float(correctly_escalated / true_high_risk_count) if true_high_risk_count > 0 else 1.0
    )
    escalation_precision = (
        float(correctly_escalated / escalated_count) if escalated_count > 0 else 1.0
    )

    # 8. Phân tích lỗi (Top Confusion Pairs)
    confusion_pairs = analyze_confusion_pairs(
        texts=te.text.tolist(),
        true_labels=targets.tolist(),
        pred_labels=pred.tolist(),
        confidences=confidence,
        margins=margin,
        top_n=20,
    )

    # 9. Tổng hợp và lưu báo cáo JSON Canonical
    test_metrics = {
        "test_accuracy": test_accuracy,
        "test_macro_f1": test_macro_f1,
        "test_top3_accuracy": test_top3_accuracy,
        "domain_taxonomy_accuracy": domain_accuracy,
        "test_ece": test_ece,
        "test_log_loss": test_log_loss,
        "test_brier_score": test_brier_score,
        "reject_threshold": threshold,
        "selective_coverage": selective_coverage,
        "selective_risk": selective_risk,
        "accepted_accuracy": accepted_accuracy,
        "aurc": rc_results["aurc"],
        "coverage_at_5pct_risk": rc_results["coverage_at_5pct_risk"],
        "coverage_at_3pct_risk": rc_results["coverage_at_3pct_risk"],
        "high_risk_escalation_recall": escalation_recall,
        "high_risk_escalation_precision": escalation_precision,
        "accepted_samples": int(accepted.sum()),
        "abstained_samples": int(abstained_mask.sum()),
        "priority_escalated_samples": escalated_count,
        "auto_routed_samples": int(auto_routed_mask.sum()),
        "total_test_samples": len(te),
    }

    save_json("reports/test_metrics.json", test_metrics)
    save_json("reports/risk_coverage_curve.json", rc_results)
    save_json(
        "reports/high_risk_metrics.json",
        {
            "true_high_risk_count": true_high_risk_count,
            "escalated_count": escalated_count,
            "correctly_escalated": correctly_escalated,
            "escalation_recall": escalation_recall,
            "escalation_precision": escalation_precision,
            "high_risk_intents": sorted(list(DEFAULT_HIGH_RISK_INTENTS)),
            "high_risk_trigger": high_risk_trigger,
        },
    )
    save_json("reports/confusion_pairs.json", confusion_pairs)

    print("=== BÁO CÁO KẾT QUẢ ĐÁNH GIÁ CANONICAL TRÊN TẬP TEST ĐỘC LẬP ===")
    for k, v in test_metrics.items():
        if isinstance(v, float):
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}")

    print("\n--- CHI TIẾT BÁO CÁO PHÂN LOẠI (CLASSIFICATION REPORT) ---")
    print(classification_report(targets, pred, zero_division=0))

    return test_metrics


def main() -> None:
    evaluate_model()


if __name__ == "__main__":
    main()
