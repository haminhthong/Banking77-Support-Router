"""Dataset split protocols with dual benchmark support: Official vs Strict Decontaminated."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal
import pandas as pd
from sklearn.model_selection import train_test_split

from .audit import audit_conflicting_labels, audit_dataset, clean_dataset
from .loader import load_official_test, normalize_dataset, read_raw_dataset
from .normalization import normalize_text_for_audit


def load_training_splits(
    raw_dir: str | Path = "data/raw",
    seed: int = 42,
    benchmark: Literal["official", "strict_decontaminated"] = "official",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split dataset into 4 distinct roles:
    - Train (70%): Feature representation & classifier fitting.
    - Calibration (15%): Fit a temperature candidate only.
    - Threshold Validation (15%): Select raw vs temperature and optimize queue/sensitive-case policy.
    - Test: Untouched test benchmark (3,080 samples).

    Benchmarks:
    1. 'official': Uses published train.csv (cleaned of internal duplicates) and untouched test.csv.
       Preserves the published Banking77 split for standard external comparisons.
    2. 'strict_decontaminated': Purges any training/holdout examples that share normalized
       text fingerprints with the test set BEFORE splitting, guaranteeing 100% leak-free development.
    """
    train_path = Path(raw_dir) / "train.csv"
    if not train_path.exists():
        raise FileNotFoundError(f"Missing train dataset: {train_path}")

    # 1. Read raw train & normalize whitespace
    raw_train = read_raw_dataset(train_path)
    norm_train = normalize_dataset(raw_train)

    # 2. Audit conflicts on raw normalized data BEFORE deduplicating
    audit_res = audit_dataset(norm_train)
    if audit_res["conflicting_label_count"] > 0:
        raise ValueError(
            f"Pre-dedup audit detected conflicting labels in train.csv: {audit_res['conflicts']}"
        )

    # 3. Clean and deduplicate development pool
    clean_train_pool = clean_dataset(norm_train)

    # 4. Load untouched official test benchmark (exactly 3,080 rows)
    test = load_official_test(raw_dir)

    # 5. Apply decontaminated filtering if requested
    if benchmark == "strict_decontaminated":
        test_fingerprints = set(test["text"].apply(normalize_text_for_audit))
        clean_train_pool["_fp"] = clean_train_pool["text"].apply(normalize_text_for_audit)
        leaked_mask = clean_train_pool["_fp"].isin(test_fingerprints)
        clean_train_pool = clean_train_pool[~leaked_mask].drop(columns=["_fp"]).reset_index(drop=True)

    # 6. Stratified Split: 70% Train, 30% Holdout
    train, holdout = train_test_split(
        clean_train_pool,
        test_size=0.30,
        random_state=seed,
        stratify=clean_train_pool["intent"],
    )

    # 7. Stratified Split of Holdout: 15% Calibration, 15% Threshold Validation
    calibration, threshold_validation = train_test_split(
        holdout,
        test_size=0.50,
        random_state=seed,
        stratify=holdout["intent"],
    )

    return (
        train.reset_index(drop=True),
        calibration.reset_index(drop=True),
        threshold_validation.reset_index(drop=True),
        test.reset_index(drop=True),
    )


def summarize_split_quality(
    train: pd.DataFrame,
    calibration: pd.DataFrame,
    threshold_val: pd.DataFrame,
    test: pd.DataFrame,
) -> dict[str, Any]:
    """Verify Data Quality Contract across all 4 splits."""
    splits = {
        "train": set(train["text"]),
        "calibration": set(calibration["text"]),
        "threshold_validation": set(threshold_val["text"]),
        "test": set(test["text"]),
    }

    norm_splits = {
        name: set(df["text"].apply(normalize_text_for_audit))
        for name, df in [
            ("train", train),
            ("calibration", calibration),
            ("threshold_validation", threshold_val),
            ("test", test),
        ]
    }

    train_labels = set(train["intent"])

    exact_pairwise_overlap: dict[str, int] = {
        "train_calibration": len(splits["train"] & splits["calibration"]),
        "train_threshold_validation": len(splits["train"] & splits["threshold_validation"]),
        "train_test": len(splits["train"] & splits["test"]),
        "calibration_threshold_validation": len(splits["calibration"] & splits["threshold_validation"]),
        "calibration_test": len(splits["calibration"] & splits["test"]),
        "threshold_validation_test": len(splits["threshold_validation"] & splits["test"]),
    }

    normalized_pairwise_overlap: dict[str, int] = {
        "train_calibration": len(norm_splits["train"] & norm_splits["calibration"]),
        "train_threshold_validation": len(norm_splits["train"] & norm_splits["threshold_validation"]),
        "train_test": len(norm_splits["train"] & norm_splits["test"]),
        "calibration_threshold_validation": len(norm_splits["calibration"] & norm_splits["threshold_validation"]),
        "calibration_test": len(norm_splits["calibration"] & norm_splits["test"]),
        "threshold_validation_test": len(norm_splits["threshold_validation"] & norm_splits["test"]),
    }

    conflicts_train = audit_conflicting_labels(train)
    conflicts_test = audit_conflicting_labels(test)

    return {
        "split_rows": {
            "train": len(train),
            "calibration": len(calibration),
            "threshold_validation": len(threshold_val),
            "test": len(test),
        },
        "class_count": len(train_labels),
        "missing_calibration_labels": sorted(set(calibration["intent"]) - train_labels),
        "missing_threshold_validation_labels": sorted(set(threshold_val["intent"]) - train_labels),
        "missing_test_labels": sorted(set(test["intent"]) - train_labels),
        "exact_pairwise_overlap": exact_pairwise_overlap,
        "normalized_pairwise_overlap": normalized_pairwise_overlap,
        # Keep pairwise_text_overlap for backward compatibility with existing reports
        "pairwise_text_overlap": exact_pairwise_overlap,
        "conflicting_labels": {
            "train_conflict_count": len(conflicts_train),
            "test_conflict_count": len(conflicts_test),
        },
    }
