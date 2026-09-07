"""Dataset loader enforcing schema validation and immutable test benchmarks."""

from __future__ import annotations

from pathlib import Path
import pandas as pd
from .contracts import BANKING77_77_CLASSES
from .normalization import normalize_whitespace


def read_raw_dataset(path: Path | str) -> pd.DataFrame:
    """Read raw BANKING77 CSV and normalize column structure to ['text', 'intent'].

    Preserves exact row count and does NOT drop duplicates or filter rows.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Dataset file not found: {p}")

    df = pd.read_csv(p)
    lower_map = {str(c).lower().strip(): c for c in df.columns}
    text_col = lower_map.get("text") or lower_map.get("query")
    intent_col = (
        lower_map.get("category") or lower_map.get("intent") or lower_map.get("label")
    )

    if text_col is None or intent_col is None:
        # Fallback for headerless CSV: category, text or intent, text
        raw = pd.read_csv(p, header=None, names=["intent", "text"])
        result = raw[["text", "intent"]].copy()
    else:
        result = df[[text_col, intent_col]].rename(
            columns={text_col: "text", intent_col: "intent"}
        ).copy()

    # Schema & type validation
    result["text"] = result["text"].astype(str)
    result["intent"] = result["intent"].astype(str)
    return result


def normalize_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Apply basic whitespace normalization to strings without altering row count."""
    norm_df = df.copy()
    norm_df["text"] = norm_df["text"].apply(normalize_whitespace)
    norm_df["intent"] = norm_df["intent"].str.strip()
    return norm_df


def load_official_test(raw_dir: Path | str = "data/raw") -> pd.DataFrame:
    """Load the official untouched BANKING77 test benchmark.

    CRITICAL PRODUCTION CONTRACT:
    - Never deduplicate.
    - Never drop rows or suspicious samples.
    - Preserves immutable benchmark integrity (exactly 3,080 published rows).
    - Report-only!
    """
    test_path = Path(raw_dir) / "test.csv"
    raw = read_raw_dataset(test_path)
    normalized = normalize_dataset(raw)

    # Validate all intents belong to the known 77 classes
    unknown_intents = set(normalized["intent"]) - set(BANKING77_77_CLASSES)
    if unknown_intents:
        raise ValueError(
            f"Official test dataset contains unknown intents: {unknown_intents}"
        )

    return normalized.reset_index(drop=True)
