"""Mô-đun quản lý dữ liệu cho dự án AI Customer Support Router (BANKING77).

Cung cấp facade tương thích ngược cho toàn bộ API dữ liệu chuẩn hóa,
giao tiếp trực tiếp với package banking_router.data.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import pandas as pd

from src.banking_router.data import (
    BANKING77_77_CLASSES,
    DEFAULT_HIGH_RISK_INTENTS,
    INTENT_TO_DOMAIN,
    audit_conflicting_labels,
    audit_dataset,
    audit_normalized_duplicates,
    clean_dataset,
    compute_file_sha256,
    get_domain_for_intent,
    load_official_test,
    load_training_splits,
    normalize_dataset,
    normalize_text_for_audit,
    normalize_whitespace,
    read_raw_dataset,
    summarize_split_quality,
)


def _read_banking77(path: Path) -> pd.DataFrame:
    """Hàm đọc dữ liệu tương thích ngược cho các unit test cũ."""
    return read_raw_dataset(path)


def _load_and_clean_csv(path: Path) -> pd.DataFrame:
    """Hàm làm sạch tương thích ngược tuân thủ thứ tự: normalize -> audit -> clean."""
    raw = read_raw_dataset(path)
    norm = normalize_dataset(raw)
    return clean_dataset(norm, conflict_action="error")


def load_test_split(raw_dir: str | Path = "data/raw") -> pd.DataFrame:
    """Tải tập Test chính thức độc lập (bảo tồn nguyên vẹn 3,080 hàng benchmark)."""
    return load_official_test(raw_dir)


__all__ = [
    "BANKING77_77_CLASSES",
    "DEFAULT_HIGH_RISK_INTENTS",
    "INTENT_TO_DOMAIN",
    "get_domain_for_intent",
    "normalize_text_for_audit",
    "compute_file_sha256",
    "_read_banking77",
    "audit_conflicting_labels",
    "audit_normalized_duplicates",
    "_load_and_clean_csv",
    "load_test_split",
    "load_official_test",
    "load_training_splits",
    "summarize_split_quality",
]
