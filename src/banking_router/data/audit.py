"""Audit functions for label conflicts and duplicates prior to deduplication."""

from __future__ import annotations

from typing import Any
import pandas as pd
from .normalization import normalize_text_for_audit


def audit_conflicting_labels(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Detect queries that have identical text but conflicting intent labels.

    CRITICAL: Must be run on the raw/normalized dataset BEFORE any deduplication,
    otherwise dropping duplicates will mask true label conflicts.
    """
    conflicts: list[dict[str, Any]] = []
    # Group on exact stripped text
    grouped = df.groupby("text")["intent"].unique()
    for text, labels in grouped.items():
        if len(labels) > 1:
            conflicts.append({
                "text": str(text),
                "intents": sorted(list(labels)),
                "count": int((df["text"] == text).sum()),
            })
    return conflicts


def audit_normalized_duplicates(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Detect queries that resolve to the same normalized fingerprint with conflicting labels."""
    temp_df = df.copy()
    temp_df["norm_text"] = temp_df["text"].apply(normalize_text_for_audit)
    grouped = temp_df.groupby("norm_text")["intent"].unique()
    norm_conflicts: list[dict[str, Any]] = []
    for norm_text, labels in grouped.items():
        if len(labels) > 1:
            norm_conflicts.append({
                "normalized_text": str(norm_text),
                "intents": sorted(list(labels)),
                "count": int((temp_df["norm_text"] == norm_text).sum()),
            })
    return norm_conflicts


def audit_dataset(df: pd.DataFrame) -> dict[str, Any]:
    """Comprehensive data quality audit report on uncleaned dataset."""
    total_rows = len(df)
    exact_duplicates = int(df.duplicated(subset=["text"]).sum())
    conflicts = audit_conflicting_labels(df)
    norm_conflicts = audit_normalized_duplicates(df)

    temp_df = df.copy()
    temp_df["norm_text"] = temp_df["text"].apply(normalize_text_for_audit)
    normalized_duplicates = int(temp_df.duplicated(subset=["norm_text"]).sum())

    return {
        "total_rows": total_rows,
        "exact_duplicates": exact_duplicates,
        "normalized_duplicates": normalized_duplicates,
        "conflicting_label_count": len(conflicts),
        "normalized_conflict_count": len(norm_conflicts),
        "conflicts": conflicts[:10],
    }


def clean_dataset(
    df: pd.DataFrame,
    conflict_action: str = "error",
) -> pd.DataFrame:
    """Clean dataset by resolving conflicts and removing exact duplicates AFTER auditing.

    Args:
        df: Input DataFrame with 'text' and 'intent'.
        conflict_action: 'error' (raise exception if conflicting labels exist) or 'drop_all'.

    Returns:
        Deduplicated, clean DataFrame.
    """
    conflicts = audit_conflicting_labels(df)
    if conflicts:
        if conflict_action == "error":
            raise ValueError(
                f"Data quality violation: Found {len(conflicts)} conflicting labels! "
                f"Example: {conflicts[0]}"
            )
        elif conflict_action == "drop_all":
            conflict_texts = {c["text"] for c in conflicts}
            df = df[~df["text"].isin(conflict_texts)].copy()

    # Drop exact duplicates and reset index
    cleaned = (
        df.dropna()
        .drop_duplicates(subset=["text"])
        .reset_index(drop=True)
    )
    return cleaned
