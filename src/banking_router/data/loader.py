"""Bộ tải dữ liệu, kiểm tra schema và bảo toàn tập test chuẩn."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .contracts import BANKING77_77_CLASSES
from .normalization import normalize_pii_semantically


def read_raw_dataset(path: Path | str) -> pd.DataFrame:
    """Đọc CSV BANKING77 và đưa schema về ``['text', 'intent']``.

    Hàm giữ nguyên số dòng, không loại bản ghi trùng và không lọc mẫu.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Không tìm thấy file dữ liệu: {p}")

    df = pd.read_csv(p)
    lower_map = {str(c).lower().strip(): c for c in df.columns}
    text_col = lower_map.get("text") or lower_map.get("query")
    intent_col = (
        lower_map.get("category") or lower_map.get("intent") or lower_map.get("label")
    )

    if text_col is None or intent_col is None:
        # Fallback cho CSV không có header: category, text hoặc intent, text.
        raw = pd.read_csv(p, header=None, names=["intent", "text"])
        result = raw[["text", "intent"]].copy()
    else:
        result = df[[text_col, intent_col]].rename(
            columns={text_col: "text", intent_col: "intent"}
        ).copy()

    # Kiểm tra schema và chuyển kiểu dữ liệu về chuỗi.
    result["text"] = result["text"].astype(str)
    result["intent"] = result["intent"].astype(str)
    return result


def normalize_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Chuẩn hóa text mà không làm thay đổi số dòng."""
    norm_df = df.copy()
    # Giữ cùng phép biến đổi với serving. Placeholder ngữ nghĩa giữ lại các từ
    # như "account" nhưng loại bỏ định danh cụ thể.
    norm_df["text"] = norm_df["text"].apply(normalize_pii_semantically)
    norm_df["intent"] = norm_df["intent"].str.strip()
    return norm_df


def load_official_test(raw_dir: Path | str = "data/raw") -> pd.DataFrame:
    """Đọc tập test BANKING77 chính thức mà không chỉnh sửa mẫu.

    Quy ước của tập đánh giá:
    - Không loại bản ghi trùng.
    - Không loại dòng hoặc mẫu bị nghi ngờ.
    - Giữ đủ 3.080 mẫu theo bản công bố.
    - Chỉ dùng để báo cáo kết quả.
    """
    test_path = Path(raw_dir) / "test.csv"
    raw = read_raw_dataset(test_path)
    normalized = normalize_dataset(raw)

    # Kiểm tra mọi intent đều thuộc 77 lớp đã biết.
    unknown_intents = set(normalized["intent"]) - set(BANKING77_77_CLASSES)
    if unknown_intents:
        raise ValueError(
            f"Tập test chính thức chứa intent không xác định: {unknown_intents}"
        )

    return normalized.reset_index(drop=True)
