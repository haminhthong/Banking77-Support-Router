"""Kiểm thử invariant chất lượng dữ liệu, chống rò rỉ và tập benchmark chuẩn."""

import pandas as pd
import pytest

from src.banking_router.data.audit import audit_conflicting_labels, clean_dataset
from src.banking_router.data.loader import load_official_test
from src.banking_router.data.split import load_training_splits, summarize_split_quality


def test_raw_conflicts_detected_before_dedup():
    """Invariant: phải bắt xung đột nhãn trước khi loại bản ghi trùng."""
    df_raw = pd.DataFrame(
        {
            "text": ["Where is my card?", "Where is my card?", "Cancel my transfer"],
            "intent": ["card_arrival", "card_delivery_estimate", "cancel_transfer"],
        }
    )

    # Audit trước khi làm sạch.
    conflicts = audit_conflicting_labels(df_raw)
    assert len(conflicts) == 1
    assert conflicts[0]["text"] == "Where is my card?"
    assert conflicts[0]["intents"] == ["card_arrival", "card_delivery_estimate"]

    # Nếu loại trùng trước, xung đột có thể bị che mất.
    # Hợp đồng clean_dataset phải ném ValueError khi conflict_action="error".
    with pytest.raises(ValueError, match="conflicting labels"):
        clean_dataset(df_raw, conflict_action="error")


def test_normalized_overlap_fails_contract():
    """Invariant: benchmark khử trùng không có overlap chuẩn hóa giữa các split."""
    tr, cal, val, te = load_training_splits(
        raw_dir="data/raw", benchmark="strict_decontaminated"
    )
    summary = summarize_split_quality(tr, cal, val, te)

    # Ở chế độ khử trùng, train/calibration/validation không overlap chuẩn hóa với test.
    assert summary["normalized_pairwise_overlap"]["train_test"] == 0
    assert summary["normalized_pairwise_overlap"]["calibration_test"] == 0
    assert summary["normalized_pairwise_overlap"]["threshold_validation_test"] == 0


def test_official_test_is_not_mutated():
    """Invariant: tập test chính thức phải giữ đúng 3.080 dòng."""
    test_df = load_official_test("data/raw")
    assert len(test_df) == 3080
    assert "text" in test_df.columns
    assert "intent" in test_df.columns
    assert test_df["text"].isna().sum() == 0
    assert test_df["intent"].isna().sum() == 0
