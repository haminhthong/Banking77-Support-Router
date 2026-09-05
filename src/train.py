"""Mô-đun huấn luyện pipeline phân loại ý định khách hàng và hiệu chỉnh xác suất (Probability Calibration).

Thực hiện:
- Phân chia 4 split độc lập (Train 70%, Calibration 15%, Threshold Validation 15%, Test chính thức).
- Huấn luyện TF-IDF + Logistic Regression trên tập Train.
- Hiệu chỉnh xác suất dự đoán (Platt Scaling) trên tập Calibration độc lập.
- Đo lường Expected Calibration Error (ECE) và Log-loss trước/sau hiệu chỉnh.
- Quét chọn ngưỡng từ chối (Reject Threshold) trên Threshold Validation với ràng buộc Coverage >= 80%.
- Xuất toàn diện model artifact, config, model_manifest và validation_metrics.
"""

from __future__ import annotations

import platform
import subprocess
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.pipeline import Pipeline

from .data import (
    BANKING77_77_CLASSES,
    compute_file_sha256,
    get_domain_for_intent,
    load_training_splits,
    summarize_split_quality,
)
from .policy import DEFAULT_HIGH_RISK_INTENTS
from .utils import calculate_ece, save_json, set_seed, setup_logging

SEED: int = 42
MIN_COVERAGE: float = 0.80
MODEL_VERSION: str = "banking77-tfidf-calibrated-lr-v2"
POLICY_VERSION: str = "risk-aware-policy-v2"
HIGH_RISK_TRIGGER: float = 0.20


def select_reject_threshold(
    confidence: np.ndarray,
    correct: np.ndarray,
    minimum_coverage: float = MIN_COVERAGE,
) -> float:
    """Tự động chọn ngưỡng từ chối tối ưu để giảm thiểu rủi ro tự động hóa (Selective Risk).

    Duyệt qua 151 ứng viên ngưỡng từ 0.20 đến 0.95 để tìm ngưỡng có Selective Risk thấp nhất
    mà vẫn đảm bảo tỷ lệ chấp nhận tự động (Coverage) >= minimum_coverage.

    Args:
        confidence (np.ndarray): Mảng xác suất tin cậy của các dự đoán.
        correct (np.ndarray): Mảng boolean chỉ ra dự đoán đúng/sai (True/False).
        minimum_coverage (float): Tỷ lệ độ phủ tối thiểu bắt buộc (mặc định 0.80 - 80%).

    Returns:
        float: Ngưỡng xác suất tối ưu chọn lựa được.
    """
    candidates = np.linspace(0.2, 0.95, 151)
    feasible: list[tuple[float, float, float]] = []

    for threshold in candidates:
        accepted = confidence >= threshold
        coverage = float(accepted.mean())
        if coverage >= minimum_coverage and accepted.any():
            risk = float(1.0 - correct[accepted].mean())
            # Điểm (-coverage) giúp ưu tiên độ phủ cao hơn khi giá trị risk bằng nhau
            feasible.append((risk, -coverage, float(threshold)))

    return min(feasible)[2] if feasible else 0.0


def _predict_with_confidence(
    model: Any, texts: pd.Series
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Dự đoán probabilities, predictions và confidence score cho danh sách văn bản."""
    proba = model.predict_proba(texts)
    pred = model.classes_[proba.argmax(axis=1)]
    conf = proba.max(axis=1)
    return proba, pred, conf


def _get_git_commit() -> str:
    """Lấy mã commit git hiện tại phục vụ truy vết metadata (Reproducibility)."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return res.stdout.strip()
    except Exception:
        return "unknown"


def main() -> None:
    """Quy trình huấn luyện, hiệu chỉnh và xuất artifacts hoàn chỉnh."""
    setup_logging()
    set_seed(SEED)

    # 1. Tải 4 split dữ liệu độc lập
    tr, calibration, threshold_val, te = load_training_splits(seed=SEED)

    # Kiểm định Data Quality Contract chéo giữa cả 4 split
    data_quality = summarize_split_quality(tr, calibration, threshold_val, te)
    if data_quality["missing_threshold_validation_labels"]:
        raise ValueError("Tập Threshold Validation chứa ý định không xuất hiện trong tập Train!")
    if data_quality["missing_calibration_labels"]:
        raise ValueError("Tập Calibration chứa ý định không xuất hiện trong tập Train!")

    # 2. Định nghĩa Pipeline baseline TF-IDF + Logistic Regression
    tfidf_params = {
        "ngram_range": (1, 2),
        "min_df": 2,
        "max_df": 0.98,
        "sublinear_tf": True,
        "max_features": 50000,
    }
    lr_params = {
        "max_iter": 1200,
        "class_weight": "balanced",
        "n_jobs": None,
        "C": 4.0,
    }

    base_model = Pipeline(
        [
            ("tfidf", TfidfVectorizer(**tfidf_params)),
            ("clf", LogisticRegression(**lr_params)),
        ]
    )

    # 3. Huấn luyện mô hình gốc trên tập Train (70%)
    base_model.fit(tr.text, tr.intent)

    # Dự đoán chưa hiệu chỉnh trên Threshold Validation
    raw_proba_val, raw_pred_val, raw_conf_val = _predict_with_confidence(
        base_model, threshold_val.text
    )

    # 4. Hiệu chỉnh xác suất dự đoán (Platt Scaling) trên tập Calibration (15%)
    try:
        from sklearn.frozen import FrozenEstimator

        calibrated_model = CalibratedClassifierCV(
            estimator=FrozenEstimator(base_model), method="sigmoid"
        )
    except (ImportError, ModuleNotFoundError):
        calibrated_model = CalibratedClassifierCV(
            estimator=base_model, cv="prefit", method="sigmoid"
        )

    calibrated_model.fit(calibration.text, calibration.intent)

    # Dự đoán đã hiệu chỉnh trên Threshold Validation
    cal_proba_val, cal_pred_val, cal_conf_val = _predict_with_confidence(
        calibrated_model, threshold_val.text
    )

    # 5. Tính toán ECE và Log-Loss trước và sau hiệu chỉnh
    threshold_targets = threshold_val.intent.to_numpy()
    raw_ece = calculate_ece(raw_conf_val, raw_pred_val, threshold_targets)
    cal_ece = calculate_ece(cal_conf_val, cal_pred_val, threshold_targets)

    # 6. Chọn ngưỡng từ chối tối ưu trên Threshold Validation
    correct_val = cal_pred_val == threshold_targets
    threshold = select_reject_threshold(cal_conf_val, correct_val, MIN_COVERAGE)
    accepted_val = cal_conf_val >= threshold

    metrics = {
        "threshold_val_macro_f1": float(
            f1_score(threshold_val.intent, cal_pred_val, average="macro")
        ),
        "threshold_val_accuracy": float(
            accuracy_score(threshold_val.intent, cal_pred_val)
        ),
        "raw_validation_log_loss": float(
            log_loss(threshold_val.intent, raw_proba_val, labels=base_model.classes_)
        ),
        "calibrated_validation_log_loss": float(
            log_loss(
                threshold_val.intent,
                cal_proba_val,
                labels=calibrated_model.classes_,
            )
        ),
        "raw_validation_ece": float(raw_ece),
        "calibrated_validation_ece": float(cal_ece),
        "reject_threshold": float(threshold),
        "selective_coverage": float(accepted_val.mean()),
        "selective_risk": float(1.0 - correct_val[accepted_val].mean()),
    }

    # 7. Tính SHA-256 dữ liệu nguồn
    train_path = Path("data/raw/train.csv")
    test_path = Path("data/raw/test.csv")
    train_sha256 = compute_file_sha256(train_path) if train_path.exists() else "missing"
    test_sha256 = compute_file_sha256(test_path) if test_path.exists() else "missing"

    # 8. Lưu trữ Model Artifacts & File Cấu Hình
    Path("models").mkdir(exist_ok=True)
    joblib.dump(calibrated_model, "models/router.joblib")

    # Bản đồ ánh xạ 77 ý định sang Domain
    unique_intents = list(calibrated_model.classes_)
    domain_map = {intent: get_domain_for_intent(intent) for intent in unique_intents}

    runtime_info = {
        "python": platform.python_version(),
        "scikit_learn": sklearn.__version__,
        "numpy": np.__version__,
        "joblib": joblib.__version__,
    }

    config_payload = {
        "schema_version": 2,
        "version": MODEL_VERSION,
        "policy_version": POLICY_VERSION,
        "seed": SEED,
        "threshold": threshold,
        "minimum_coverage": MIN_COVERAGE,
        "high_risk_trigger": HIGH_RISK_TRIGGER,
        "class_count": len(unique_intents),
        "split_contract": {
            "train": "fit TF-IDF và classifier (70%)",
            "calibration": "fit Platt scaling (15%)",
            "threshold_validation": "chọn reject threshold (15%)",
            "test": "đánh giá cuối duy nhất",
        },
        "raw_validation_ece": raw_ece,
        "calibrated_validation_ece": cal_ece,
        "domain_map": domain_map,
        "runtime": runtime_info,
    }
    save_json("models/config.json", config_payload)

    # Lưu Model Manifest chuyên sâu cho Production
    manifest_payload = {
        "model_version": MODEL_VERSION,
        "policy_version": POLICY_VERSION,
        "git_commit": _get_git_commit(),
        "dataset_checksums": {
            "train_csv_sha256": train_sha256,
            "test_csv_sha256": test_sha256,
        },
        "split_sizes": {
            "train_rows": len(tr),
            "calibration_rows": len(calibration),
            "threshold_validation_rows": len(threshold_val),
            "test_rows": len(te),
        },
        "tfidf_config": tfidf_params,
        "classifier_config": lr_params,
        "calibration_method": "sigmoid_platt_scaling",
        "high_risk_intents": sorted(list(DEFAULT_HIGH_RISK_INTENTS)),
        "high_risk_trigger": HIGH_RISK_TRIGGER,
        "class_labels_count": len(unique_intents),
        "runtime": runtime_info,
    }
    save_json("models/model_manifest.json", manifest_payload)

    # Lưu kết quả Validation
    Path("reports").mkdir(exist_ok=True)
    save_json(
        "reports/validation_metrics.json",
        {
            **metrics,
            "split_rows": {
                "train": len(tr),
                "calibration": len(calibration),
                "threshold_validation": len(threshold_val),
                "test": len(te),
            },
            "data_quality": data_quality,
        },
    )

    print("=== KẾT QUẢ HUẤN LUYỆN & HIỆU CHỈNH TRÊN THRESHOLD VALIDATION ===")
    for metric_name, value in metrics.items():
        print(
            f"{metric_name}: {value:.4f}"
            if isinstance(value, float)
            else f"{metric_name}: {value}"
        )


if __name__ == "__main__":
    main()
