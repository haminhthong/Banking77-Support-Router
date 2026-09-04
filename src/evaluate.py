"""Mô-đun đánh giá hiệu năng mô hình AI Customer Support Router trên tập kiểm thử độc lập.

Đánh giá toàn diện các chỉ số:
- Độ chính xác tổng thể (Accuracy) & Macro-F1.
- Độ chính xác cấp miền nghiệp vụ (Domain-level Accuracy).
- Chỉ số sai lệch xác suất (Expected Calibration Error - ECE).
- Chỉ số tự động hóa an toàn: Selective Coverage & Selective Risk.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, f1_score

from .data import get_domain_for_intent, load_test_split
from .utils import calculate_ece, save_json, setup_logging


def evaluate_model() -> dict[str, Any]:
    """Thực hiện đánh giá chi tiết mô hình trên tập Test độc lập.

    Returns:
        dict[str, Any]: Từ điển chứa tất cả các chỉ số hiệu năng đã tính toán.
    """
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

    # 1. Dự đoán trên tập Test
    proba = model.predict_proba(te.text)
    pred = model.classes_[proba.argmax(axis=1)]
    confidence = proba.max(axis=1)

    # 2. Tính toán các chỉ số phân loại cơ bản
    test_accuracy = float(accuracy_score(te.intent, pred))
    test_macro_f1 = float(f1_score(te.intent, pred, average="macro"))

    # 3. Tính toán độ chính xác phân loại cấp Miền Nghiệp Vụ (Domain Level)
    true_domains = np.array([get_domain_for_intent(i) for i in te.intent])
    pred_domains = np.array([get_domain_for_intent(i) for i in pred])
    domain_accuracy = float(accuracy_score(true_domains, pred_domains))

    # 4. Tính toán Expected Calibration Error (ECE) trên tập Test
    test_ece = calculate_ece(confidence, pred, te.intent.to_numpy())

    # 5. Đánh giá Selective Prediction (Coverage & Risk) với ngưỡng chọn từ Validation
    accepted = confidence >= threshold
    correct = pred == te.intent.to_numpy()

    selective_coverage = float(accepted.mean())
    selective_risk = float(1.0 - correct[accepted].mean()) if accepted.any() else 0.0

    metrics = {
        "test_accuracy": test_accuracy,
        "test_macro_f1": test_macro_f1,
        "domain_accuracy": domain_accuracy,
        "test_ece": test_ece,
        "reject_threshold": threshold,
        "selective_coverage": selective_coverage,
        "selective_risk": selective_risk,
        "accepted_samples": int(accepted.sum()),
        "total_test_samples": len(te),
    }

    save_json("reports/test_metrics.json", metrics)

    print("=== BÁO CÁO KẾT QUẢ ĐÁNH GIÁ TRÊN TẬP TEST ĐỘC LẬP ===")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}")

    print("\n--- CHI TIẾT BÁO CÁO PHÂN LOẠI (CLASSIFICATION REPORT) ---")
    print(classification_report(te.intent, pred, zero_division=0))

    return metrics


def main() -> None:
    evaluate_model()


if __name__ == "__main__":
    main()
