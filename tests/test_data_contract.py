"""Các unit test kiểm tra hợp đồng dữ liệu (Data Quality Contract) và thuật toán chọn ngưỡng."""

from pathlib import Path

import numpy as np
import pandas as pd
from src.data import (
    _read_banking77,
    get_domain_for_intent,
    load_training_splits,
    summarize_split_quality,
)
from src.train import select_reject_threshold


def test_read_banking77_normalizes_columns_and_order(tmp_path: Path):
    """Kiểm tra đọc file CSV bất kỳ và chuẩn hóa cột về ['text', 'intent']."""
    csv_path = tmp_path / "sample.csv"
    csv_path.write_text(
        "category,text,unused\ncard_arrival,Where is my card?,ignored\n",
        encoding="utf-8",
    )

    result = _read_banking77(csv_path)

    assert list(result.columns) == ["text", "intent"]
    assert result.iloc[0].to_dict() == {
        "text": "Where is my card?",
        "intent": "card_arrival",
    }


def test_get_domain_for_intent_mapping():
    """Kiểm tra ánh xạ intent chi tiết sang miền nghiệp vụ cấp cao."""
    assert get_domain_for_intent("card_arrival") == "card_services"
    assert get_domain_for_intent("cash_withdrawal_not_recognised") == "atm_cash"
    assert get_domain_for_intent("topup_failed") == "topup_recharge"
    assert get_domain_for_intent("transfer_fee_charged") == "transfers_payments"
    assert get_domain_for_intent("unknown_custom_intent") == "general_banking"


def test_reject_threshold_keeps_minimum_coverage():
    """Kiểm tra thuật toán chọn ngưỡng luôn đáp ứng độ phủ tối thiểu (minimum_coverage)."""
    confidence = np.array([0.95, 0.85, 0.75, 0.65, 0.10])
    correct = np.array([True, True, True, True, False])

    threshold = select_reject_threshold(confidence, correct, minimum_coverage=0.80)

    assert float((confidence >= threshold).mean()) >= 0.80
    assert threshold > 0.10


def test_split_quality_detects_text_overlap():
    """Kiểm tra báo cáo chất lượng phát hiện văn bản bị trùng giữa các tập dữ liệu."""
    train = pd.DataFrame({"text": ["same", "train"], "intent": ["a", "b"]})
    validation = pd.DataFrame({"text": ["validation"], "intent": ["a"]})
    test = pd.DataFrame({"text": ["same"], "intent": ["a"]})

    summary = summarize_split_quality(train, validation, test)

    assert summary["train_test_text_overlap"] == 1
    assert summary["missing_test_labels"] == []


def test_training_splits_have_distinct_roles():
    """Calibration và threshold-validation không được dùng chung mẫu."""
    train, calibration, threshold_validation, test = load_training_splits()
    text_sets = [
        set(frame["text"]) for frame in (train, calibration, threshold_validation)
    ]
    assert text_sets[0].isdisjoint(text_sets[1])
    assert text_sets[0].isdisjoint(text_sets[2])
    assert text_sets[1].isdisjoint(text_sets[2])
    assert len(test) > 0
