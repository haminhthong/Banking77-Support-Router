"""Kiểm tra xung đột nhãn và bản ghi trùng trước khi làm sạch dữ liệu."""

from __future__ import annotations

from typing import Any
import pandas as pd
from .normalization import normalize_text_for_audit


def audit_conflicting_labels(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Phát hiện câu có cùng nội dung nhưng khác nhãn intent.

    LƯU Ý: Phải chạy trên dữ liệu thô đã chuẩn hóa trước khi loại bản ghi trùng;
    nếu làm ngược lại, bước loại trùng có thể che mất xung đột nhãn.
    """
    conflicts: list[dict[str, Any]] = []
    # Gom nhóm theo nội dung đã loại khoảng trắng thừa.
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
    """Phát hiện fingerprint chuẩn hóa trùng nhưng có các nhãn khác nhau."""
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
    """Tạo báo cáo tổng hợp chất lượng trên dữ liệu chưa làm sạch."""
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
    """Làm sạch dữ liệu sau khi audit bằng cách xử lý xung đột và loại bản ghi trùng.

    Tham số:
        df: DataFrame đầu vào có cột ``text`` và ``intent``.
        conflict_action: ``error`` để dừng khi có xung đột hoặc ``drop_all`` để loại toàn bộ câu xung đột.

    Kết quả:
        DataFrame đã loại bản ghi trùng và đặt lại chỉ mục.
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

    # Loại bản ghi trùng chính xác và đặt lại chỉ mục.
    cleaned = (
        df.dropna()
        .drop_duplicates(subset=["text"])
        .reset_index(drop=True)
    )
    return cleaned
