"""Unit tests verifying data quality invariants, leakage-safety, and immutable benchmarks."""

from pathlib import Path
import pandas as pd
import pytest

from src.banking_router.data.audit import audit_conflicting_labels, clean_dataset
from src.banking_router.data.loader import load_official_test, normalize_dataset
from src.banking_router.data.split import load_training_splits, summarize_split_quality


def test_raw_conflicts_detected_before_dedup():
    """INVARIANT: Conflicting labels on identical text must be caught BEFORE deduplication."""
    df_raw = pd.DataFrame({
        "text": ["Where is my card?", "Where is my card?", "Cancel my transfer"],
        "intent": ["card_arrival", "card_delivery_estimate", "cancel_transfer"],
    })

    # Audit before cleaning
    conflicts = audit_conflicting_labels(df_raw)
    assert len(conflicts) == 1
    assert conflicts[0]["text"] == "Where is my card?"
    assert conflicts[0]["intents"] == ["card_arrival", "card_delivery_estimate"]

    # If deduplication were mistakenly called first, the conflict would be masked!
    # Our clean_dataset contract must raise ValueError when conflict_action="error"
    with pytest.raises(ValueError, match="conflicting labels"):
        clean_dataset(df_raw, conflict_action="error")


def test_normalized_overlap_fails_contract():
    """INVARIANT: Strict decontaminated benchmark guarantees 0 cross-split leakage."""
    tr, cal, val, te = load_training_splits(raw_dir="data/raw", benchmark="strict_decontaminated")
    summary = summarize_split_quality(tr, cal, val, te)

    # In strict decontaminated mode, train/calibration/validation have 0 normalized overlap with test!
    assert summary["normalized_pairwise_overlap"]["train_test"] == 0
    assert summary["normalized_pairwise_overlap"]["calibration_test"] == 0
    assert summary["normalized_pairwise_overlap"]["threshold_validation_test"] == 0


def test_official_test_is_not_mutated():
    """INVARIANT: Official published test benchmark must remain exactly 3,080 rows."""
    test_df = load_official_test("data/raw")
    assert len(test_df) == 3080
    assert "text" in test_df.columns
    assert "intent" in test_df.columns
    assert test_df["text"].isna().sum() == 0
    assert test_df["intent"].isna().sum() == 0
