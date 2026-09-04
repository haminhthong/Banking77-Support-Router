"""Mô-đun huấn luyện pipeline phân loại ý định khách hàng và hiệu chỉnh xác suất (Probability Calibration).

Thực hiện huấn luyện TF-IDF + Logistic Regression, hiệu chỉnh xác suất dự đoán (Platt Scaling),
tính toán chỉ số Expected Calibration Error (ECE), và tự động tối ưu hóa ngưỡng từ chối (Reject Threshold)
trên tập Validation với ràng buộc độ phủ (Coverage >= 80%).
"""

from __future__ import annotations

import platform
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.pipeline import Pipeline

from .data import get_domain_for_intent, load_training_splits, summarize_split_quality
from .utils import calculate_ece, save_json, set_seed, setup_logging

SEED: int = 42
MIN_COVERAGE: float = 0.80


def select_reject_threshold(
    confidence: np.ndarray,
    correct: np.ndarray,
    minimum_coverage: float = MIN_COVERAGE,
) -> float:
    """Tự động chọn ngưỡng từ chối tối ưu để giảm thiểu rủi ro tự động hóa (Selective Risk).

    Duyệt qua 151 ứng viên ngưỡng từ 0.20 đến 0.95 để tìm ngưỡng có Selective Risk thấp nhất
    mà vẫn đảm bảo tỷ lệ chấp nhận tự động (Coverage) tối thiểu theo yêu cầu nghiệp vụ.

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


def main() -> None:
    """Quy trình huấn luyện và chọn ngưỡng hoàn chỉnh."""
    setup_logging()
    set_seed(SEED)

    # 1. Tải tập dữ liệu đã phân tầng
    tr, calibration, threshold_val, te = load_training_splits(seed=SEED)
    data_quality = summarize_split_quality(tr, threshold_val, te)
    if data_quality["missing_validation_labels"]:
        raise ValueError("Tập Validation chứa ý định không xuất hiện trong tập Train!")

    # 2. Định nghĩa Pipeline phân loại baseline
    base_model = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.98,
                    sublinear_tf=True,
                    max_features=50000,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=1200, class_weight="balanced", n_jobs=None, C=4.0
                ),
            ),
        ]
    )

    # 3. Huấn luyện mô hình gốc trên tập Train
    base_model.fit(tr.text, tr.intent)

    # Dự đoán chưa hiệu chỉnh trên Validation
    raw_proba_val, raw_pred_val, raw_conf_val = _predict_with_confidence(
        base_model, threshold_val.text
    )

    # 4. Hiệu chỉnh xác suất dự đoán (Probability Calibration bằng Platt Scaling)
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

    # Dự đoán đã hiệu chỉnh trên Validation
    cal_proba_val, cal_pred_val, cal_conf_val = _predict_with_confidence(
        calibrated_model, threshold_val.text
    )

    # 5. Tính toán Expected Calibration Error (ECE) trước và sau hiệu chỉnh
    threshold_targets = threshold_val.intent.to_numpy()
    raw_ece = calculate_ece(raw_conf_val, raw_pred_val, threshold_targets)
    cal_ece = calculate_ece(cal_conf_val, cal_pred_val, threshold_targets)

    # 6. Chọn ngưỡng từ chối tối ưu (Reject Threshold) trên tập Validation
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
        "unknown_threshold": float(threshold),
        "selective_coverage": float(accepted_val.mean()),
        "selective_risk": float(1.0 - correct_val[accepted_val].mean()),
    }

    # 7. Lưu trữ Model Artifacts & File Cấu Hình
    Path("models").mkdir(exist_ok=True)
    joblib.dump(calibrated_model, "models/router.joblib")

    # Tạo bản đồ ánh xạ 77 ý định sang Domain nghiệp vụ
    unique_intents = list(calibrated_model.classes_)
    domain_map = {intent: get_domain_for_intent(intent) for intent in unique_intents}

    save_json(
        "models/config.json",
        {
            "schema_version": 2,
            "version": "banking77-tfidf-calibrated-lr-v2",
            "seed": SEED,
            "threshold": threshold,
            "minimum_coverage": MIN_COVERAGE,
            "class_count": len(unique_intents),
            "split_contract": {
                "train": "fit TF-IDF và classifier",
                "calibration": "fit Platt scaling",
                "threshold_validation": "chọn reject threshold",
                "test": "đánh giá cuối duy nhất",
            },
            "raw_validation_ece": raw_ece,
            "calibrated_validation_ece": cal_ece,
            "domain_map": domain_map,
            "runtime": {
                "python": platform.python_version(),
                "scikit_learn": sklearn.__version__,
                "numpy": np.__version__,
                "joblib": joblib.__version__,
            },
        },
    )

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

    print("=== KẾT QUẢ HUẤN LUYỆN & HIỆU CHỈNH TRÊN VALIDATION ===")
    for metric_name, value in metrics.items():
        print(
            f"{metric_name}: {value:.4f}"
            if isinstance(value, float)
            else f"{metric_name}: {value}"
        )


if __name__ == "__main__":
    main()
