"""Mô-đun quản lý dữ liệu cho dự án AI Customer Support Router.

Thực hiện đọc file dữ liệu BANKING77, làm sạch, khử trùng lặp văn bản,
phân chia stratified split (train/validation/test), gom nhóm intent thành các miền nghiệp vụ (domain),
và kiểm định hợp đồng chất lượng dữ liệu (Data Quality Contract).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

# Danh sách 10 miền nghiệp vụ chính gom nhóm từ 77 ý định chi tiết
DOMAIN_MAPPING_RULES: dict[str, str] = {
    "card": "card_services",
    "atm": "atm_cash",
    "cash": "atm_cash",
    "topup": "topup_recharge",
    "transfer": "transfers_payments",
    "beneficiary": "transfers_payments",
    "direct_debit": "transfers_payments",
    "pin": "account_security",
    "passcode": "account_security",
    "compromised": "account_security",
    "verify": "account_security",
    "security": "account_security",
    "fee": "fees_rates",
    "rate": "fees_rates",
    "exchange": "fees_rates",
    "transaction": "transactions_refunds",
    "refund": "transactions_refunds",
    "declined": "transactions_refunds",
    "balance": "account_management",
    "statement": "account_management",
    "detail": "account_management",
    "terminate": "account_management",
    "app": "app_features",
    "device": "app_features",
    "country": "international_services",
    "travel": "international_services",
}


def get_domain_for_intent(intent: str) -> str:
    """Ánh xạ một intent chi tiết (trên BANKING77) thành miền nghiệp vụ (Domain) cấp cao.

    Args:
        intent (str): Nhãn ý định chi tiết (ví dụ: 'card_arrival', 'cash_withdrawal_not_recognised').

    Returns:
        str: Miền nghiệp vụ tương ứng (ví dụ: 'card_services', 'atm_cash').
    """
    clean_intent = intent.lower().strip()
    for keyword, domain in DOMAIN_MAPPING_RULES.items():
        if keyword in clean_intent:
            return domain
    return "general_banking"


def _read_banking77(path: Path) -> pd.DataFrame:
    """Đọc và chuẩn hóa cấu trúc cột của tập dữ liệu BANKING77 CSV.

    Args:
        path (Path): Đường dẫn tới file CSV.

    Returns:
        pd.DataFrame: DataFrame gồm 2 cột chuẩn hóa ['text', 'intent'].
    """
    df = pd.read_csv(path)
    # Tìm tên cột không phân biệt hoa thường
    lower_map = {c.lower().strip(): c for c in df.columns}
    text_col = lower_map.get("text") or lower_map.get("query")
    intent_col = (
        lower_map.get("category") or lower_map.get("intent") or lower_map.get("label")
    )

    if text_col is None or intent_col is None:
        # Trường hợp CSV không có tiêu đề cột chuẩn
        raw = pd.read_csv(path, header=None, names=["intent", "text"])
        return raw[["text", "intent"]]

    return df[[text_col, intent_col]].rename(
        columns={text_col: "text", intent_col: "intent"}
    )


def _load_and_clean_csv(path: Path) -> pd.DataFrame:
    """Đọc, loại bỏ dòng rỗng, chuẩn hóa khoảng trắng và khử trùng lặp văn bản."""
    df = _read_banking77(path).dropna().copy()
    df["text"] = df["text"].astype(str).str.strip()
    df["intent"] = df["intent"].astype(str).str.strip()
    return df[df["text"].ne("")].drop_duplicates(subset=["text"]).reset_index(drop=True)


def load_test_split(raw_dir: str | Path = "data/raw") -> pd.DataFrame:
    """Tải và làm sạch tập dữ liệu Test độc lập."""
    test_path = Path(raw_dir) / "test.csv"
    if not test_path.exists():
        raise FileNotFoundError(
            f"Thiếu file dữ liệu: {test_path}. Vui lòng chạy scripts/download_data.py trước!"
        )
    return _load_and_clean_csv(test_path)


def load_splits(
    raw_dir: str | Path = "data/raw", seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Tải và phân chia tập dữ liệu thành Train (85%), Validation (15%), và Test độc lập."""
    train_path = Path(raw_dir) / "train.csv"
    test_path = Path(raw_dir) / "test.csv"
    missing = [str(p) for p in (train_path, test_path) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Thiếu file dữ liệu: {missing}. Vui lòng chạy scripts/download_data.py trước!"
        )

    full = _load_and_clean_csv(train_path)
    tr, va = train_test_split(
        full, test_size=0.15, random_state=seed, stratify=full["intent"]
    )
    te = _load_and_clean_csv(test_path)

    return tr.reset_index(drop=True), va.reset_index(drop=True), te


def load_training_splits(
    raw_dir: str | Path = "data/raw", seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Tạo train/calibration/threshold-validation và giữ test chính thức độc lập."""
    train_path = Path(raw_dir) / "train.csv"
    test_path = Path(raw_dir) / "test.csv"
    missing = [str(path) for path in (train_path, test_path) if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Thiếu file dữ liệu: {missing}")

    full = _load_and_clean_csv(train_path)
    train, holdout = train_test_split(
        full,
        test_size=0.30,
        random_state=seed,
        stratify=full["intent"],
    )
    calibration, threshold_validation = train_test_split(
        holdout,
        test_size=0.50,
        random_state=seed,
        stratify=holdout["intent"],
    )
    test = _load_and_clean_csv(test_path)

    return tuple(
        frame.reset_index(drop=True)
        for frame in (train, calibration, threshold_validation, test)
    )


def summarize_split_quality(
    train: pd.DataFrame, validation: pd.DataFrame, test: pd.DataFrame
) -> dict[str, Any]:
    """Tóm tắt báo cáo Data Quality Contract để phát hiện rò rỉ dữ liệu (Data Leakage).

    Args:
        train (pd.DataFrame): Tập dữ liệu huấn luyện.
        validation (pd.DataFrame): Tập dữ liệu hiệu chỉnh & chọn ngưỡng.
        test (pd.DataFrame): Tập dữ liệu đánh giá cuối cùng.

    Returns:
        dict[str, Any]: Báo cáo tổng số dòng, số lớp, các nhãn bị thiếu và mức độ overlap văn bản.
    """
    train_texts = set(train["text"])
    val_texts = set(validation["text"])
    test_texts = set(test["text"])
    train_labels = set(train["intent"])

    return {
        "train_rows": len(train),
        "validation_rows": len(validation),
        "test_rows": len(test),
        "class_count": len(train_labels),
        "missing_validation_labels": sorted(set(validation["intent"]) - train_labels),
        "missing_test_labels": sorted(set(test["intent"]) - train_labels),
        "train_validation_text_overlap": len(train_texts & val_texts),
        "train_test_text_overlap": len(train_texts & test_texts),
    }
