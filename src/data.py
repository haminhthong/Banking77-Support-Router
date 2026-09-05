"""Mô-đun quản lý dữ liệu cho dự án AI Customer Support Router (BANKING77).

Thực hiện:
- Đọc, chuẩn hóa, làm sạch và khử trùng lặp dữ liệu BANKING77.
- Cung cấp bảng taxonomy chính xác (Exact 77-class Taxonomy) ánh xạ từng ý định sang 10 miền nghiệp vụ.
- Phân chia 4 split độc lập: Train (70%), Calibration (15%), Threshold Validation (15%), Test (chính thức).
- Kiểm tra hợp đồng chất lượng dữ liệu (Data Quality Contract): phát hiện rò rỉ chéo giữa cả 4 split,
  audit conflicting labels và normalized duplicates.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

# Danh sách 77 nhãn chuẩn của BANKING77
BANKING77_77_CLASSES: list[str] = sorted(
    [
        "Refund_not_showing_up",
        "activate_my_card",
        "age_limit",
        "apple_pay_or_google_pay",
        "atm_support",
        "automatic_top_up",
        "balance_not_updated_after_bank_transfer",
        "balance_not_updated_after_cheque_or_cash_deposit",
        "beneficiary_not_allowed",
        "cancel_transfer",
        "card_about_to_expire",
        "card_acceptance",
        "card_arrival",
        "card_delivery_estimate",
        "card_linking",
        "card_not_working",
        "card_payment_fee_charged",
        "card_payment_not_recognised",
        "card_payment_wrong_exchange_rate",
        "card_swallowed",
        "cash_withdrawal_charge",
        "cash_withdrawal_not_recognised",
        "change_pin",
        "compromised_card",
        "contactless_not_working",
        "country_support",
        "declined_card_payment",
        "declined_cash_withdrawal",
        "declined_transfer",
        "direct_debit_payment_not_recognised",
        "disposable_card_limits",
        "edit_personal_details",
        "exchange_charge",
        "exchange_rate",
        "exchange_via_app",
        "extra_charge_on_statement",
        "failed_transfer",
        "fiat_currency_support",
        "get_disposable_virtual_card",
        "get_physical_card",
        "getting_spare_card",
        "getting_virtual_card",
        "lost_or_stolen_card",
        "lost_or_stolen_phone",
        "order_physical_card",
        "passcode_forgotten",
        "pending_card_payment",
        "pending_cash_withdrawal",
        "pending_top_up",
        "pending_transfer",
        "pin_blocked",
        "receiving_money",
        "request_refund",
        "reverted_card_payment?",
        "supported_cards_and_currencies",
        "terminate_account",
        "top_up_by_bank_transfer_charge",
        "top_up_by_card_charge",
        "top_up_by_cash_or_cheque",
        "top_up_failed",
        "top_up_limits",
        "top_up_reverted",
        "topping_up_by_card",
        "transaction_charged_twice",
        "transfer_fee_charged",
        "transfer_into_account",
        "transfer_not_received_by_recipient",
        "transfer_timing",
        "unable_to_verify_identity",
        "verify_my_identity",
        "verify_source_of_funds",
        "verify_top_up",
        "virtual_card_not_working",
        "visa_or_mastercard",
        "why_verify_identity",
        "wrong_amount_of_cash_received",
        "wrong_exchange_rate_for_cash_withdrawal",
    ]
)

# Ánh xạ tường minh (Exact Taxonomy) cho toàn bộ 77 intent vào 10 miền nghiệp vụ
INTENT_TO_DOMAIN: dict[str, str] = {
    # 1. card_services (17 intents)
    "activate_my_card": "card_services",
    "card_about_to_expire": "card_services",
    "card_acceptance": "card_services",
    "card_arrival": "card_services",
    "card_delivery_estimate": "card_services",
    "card_linking": "card_services",
    "card_not_working": "card_services",
    "contactless_not_working": "card_services",
    "disposable_card_limits": "card_services",
    "get_disposable_virtual_card": "card_services",
    "get_physical_card": "card_services",
    "getting_spare_card": "card_services",
    "getting_virtual_card": "card_services",
    "order_physical_card": "card_services",
    "supported_cards_and_currencies": "card_services",
    "virtual_card_not_working": "card_services",
    "visa_or_mastercard": "card_services",
    # 2. atm_cash (9 intents)
    "atm_support": "atm_cash",
    "balance_not_updated_after_cheque_or_cash_deposit": "atm_cash",
    "card_swallowed": "atm_cash",
    "cash_withdrawal_charge": "atm_cash",
    "cash_withdrawal_not_recognised": "atm_cash",
    "declined_cash_withdrawal": "atm_cash",
    "pending_cash_withdrawal": "atm_cash",
    "wrong_amount_of_cash_received": "atm_cash",
    "wrong_exchange_rate_for_cash_withdrawal": "atm_cash",
    # 3. account_security (10 intents)
    "change_pin": "account_security",
    "compromised_card": "account_security",
    "lost_or_stolen_card": "account_security",
    "lost_or_stolen_phone": "account_security",
    "passcode_forgotten": "account_security",
    "pin_blocked": "account_security",
    "unable_to_verify_identity": "account_security",
    "verify_my_identity": "account_security",
    "verify_source_of_funds": "account_security",
    "why_verify_identity": "account_security",
    # 4. transfers_payments (12 intents)
    "balance_not_updated_after_bank_transfer": "transfers_payments",
    "beneficiary_not_allowed": "transfers_payments",
    "cancel_transfer": "transfers_payments",
    "declined_transfer": "transfers_payments",
    "direct_debit_payment_not_recognised": "transfers_payments",
    "failed_transfer": "transfers_payments",
    "pending_transfer": "transfers_payments",
    "receiving_money": "transfers_payments",
    "transfer_fee_charged": "transfers_payments",
    "transfer_into_account": "transfers_payments",
    "transfer_not_received_by_recipient": "transfers_payments",
    "transfer_timing": "transfers_payments",
    # 5. topup_recharge (10 intents)
    "automatic_top_up": "topup_recharge",
    "pending_top_up": "topup_recharge",
    "top_up_by_bank_transfer_charge": "topup_recharge",
    "top_up_by_card_charge": "topup_recharge",
    "top_up_by_cash_or_cheque": "topup_recharge",
    "top_up_failed": "topup_recharge",
    "top_up_limits": "topup_recharge",
    "top_up_reverted": "topup_recharge",
    "topping_up_by_card": "topup_recharge",
    "verify_top_up": "topup_recharge",
    # 6. transactions_refunds (10 intents)
    "Refund_not_showing_up": "transactions_refunds",
    "card_payment_fee_charged": "transactions_refunds",
    "card_payment_not_recognised": "transactions_refunds",
    "card_payment_wrong_exchange_rate": "transactions_refunds",
    "declined_card_payment": "transactions_refunds",
    "extra_charge_on_statement": "transactions_refunds",
    "pending_card_payment": "transactions_refunds",
    "request_refund": "transactions_refunds",
    "reverted_card_payment?": "transactions_refunds",
    "transaction_charged_twice": "transactions_refunds",
    # 7. fees_rates (4 intents)
    "exchange_charge": "fees_rates",
    "exchange_rate": "fees_rates",
    "exchange_via_app": "fees_rates",
    "fiat_currency_support": "fees_rates",
    # 8. account_management (3 intents)
    "age_limit": "account_management",
    "edit_personal_details": "account_management",
    "terminate_account": "account_management",
    # 9. app_features (1 intent)
    "apple_pay_or_google_pay": "app_features",
    # 10. international_services (1 intent)
    "country_support": "international_services",
}

# Đảm bảo tính toàn vẹn ngay khi nạp module
assert set(INTENT_TO_DOMAIN.keys()) == set(
    BANKING77_77_CLASSES
), "Bảng taxonomy thiếu hoặc thừa intent so với chuẩn 77 nhãn BANKING77!"


def get_domain_for_intent(intent: str) -> str:
    """Ánh xạ ý định (Intent) chi tiết sang miền nghiệp vụ (Domain) cấp cao.

    Đây là quy trình Post-hoc Hierarchical Taxonomy Projection phục vụ
    coarse-routing và phân luồng phòng ban, không phải mô hình phân loại domain độc lập.

    Args:
        intent (str): Nhãn ý định chi tiết (ví dụ: 'card_arrival', 'top_up_failed').

    Returns:
        str: Miền nghiệp vụ tương ứng (mặc định 'general_banking' nếu không tìm thấy).
    """
    clean_intent = intent.strip()
    return INTENT_TO_DOMAIN.get(clean_intent, "general_banking")


def normalize_text_for_audit(text: str) -> str:
    """Chuẩn hóa văn bản phục vụ kiểm tra trùng lặp ngữ nghĩa (Normalized Audit)."""
    text = unicodedata.normalize("NFKC", text.lower().strip())
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def compute_file_sha256(path: Path) -> str:
    """Tính mã băm SHA-256 của file để đảm bảo tính tái lập (Reproducibility)."""
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def _read_banking77(path: Path) -> pd.DataFrame:
    """Đọc và chuẩn hóa cấu trúc cột của tập dữ liệu BANKING77 CSV."""
    df = pd.read_csv(path)
    lower_map = {c.lower().strip(): c for c in df.columns}
    text_col = lower_map.get("text") or lower_map.get("query")
    intent_col = (
        lower_map.get("category") or lower_map.get("intent") or lower_map.get("label")
    )

    if text_col is None or intent_col is None:
        raw = pd.read_csv(path, header=None, names=["intent", "text"])
        return raw[["text", "intent"]]

    return df[[text_col, intent_col]].rename(
        columns={text_col: "text", intent_col: "intent"}
    )


def audit_conflicting_labels(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Phát hiện các mẫu câu có văn bản giống hệt nhau nhưng bị gán nhiều nhãn intent khác nhau."""
    conflicts: list[dict[str, Any]] = []
    grouped = df.groupby("text")["intent"].unique()
    for text, labels in grouped.items():
        if len(labels) > 1:
            conflicts.append({"text": str(text), "intents": sorted(list(labels))})
    return conflicts


def audit_normalized_duplicates(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Phát hiện các mẫu câu trùng lặp sau khi đã chuẩn hóa chữ thường, dấu câu và khoảng trắng."""
    temp_df = df.copy()
    temp_df["norm_text"] = temp_df["text"].apply(normalize_text_for_audit)
    grouped = temp_df.groupby("norm_text")["intent"].unique()
    norm_conflicts: list[dict[str, Any]] = []
    for norm_text, labels in grouped.items():
        if len(labels) > 1:
            norm_conflicts.append(
                {"normalized_text": str(norm_text), "intents": sorted(list(labels))}
            )
    return norm_conflicts


def _load_and_clean_csv(path: Path) -> pd.DataFrame:
    """Đọc, loại bỏ dòng rỗng, chuẩn hóa khoảng trắng và khử trùng lặp chính xác."""
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


def load_training_splits(
    raw_dir: str | Path = "data/raw", seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Phân tách 4 vai trò rõ ràng:

    - Train (70%): Fit TF-IDF và Classifier
    - Calibration (15%): Fit Platt Scaling (Sigmoid Calibrator)
    - Threshold Validation (15%): Quét tìm Reject Threshold với Coverage >= 80%
    - Official Test: Đánh giá duy nhất và giữ nguyên độc lập.
    """
    train_path = Path(raw_dir) / "train.csv"
    test_path = Path(raw_dir) / "test.csv"
    missing = [str(path) for path in (train_path, test_path) if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Thiếu file dữ liệu: {missing}")

    full = _load_and_clean_csv(train_path)

    # 70% Train, 30% Holdout
    train, holdout = train_test_split(
        full,
        test_size=0.30,
        random_state=seed,
        stratify=full["intent"],
    )

    # Chia đều 30% Holdout thành 15% Calibration và 15% Threshold Validation
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
    train: pd.DataFrame,
    calibration: pd.DataFrame,
    threshold_val: pd.DataFrame,
    test: pd.DataFrame,
) -> dict[str, Any]:
    """Kiểm định Data Quality Contract toàn diện:

    - Kích thước từng split.
    - Kiểm tra số lượng nhãn và nhãn bị thiếu giữa các split.
    - Kiểm định rò rỉ dữ liệu chéo giữa tất cả 6 cặp split (Pairwise Text Overlap).
    - Audit conflicting labels trên dữ liệu huấn luyện và kiểm thử.
    """
    splits = {
        "train": set(train["text"]),
        "calibration": set(calibration["text"]),
        "threshold_validation": set(threshold_val["text"]),
        "test": set(test["text"]),
    }

    train_labels = set(train["intent"])

    # Tính toán overlap giữa tất cả các cặp split
    pairwise_overlap: dict[str, int] = {
        "train_calibration": len(splits["train"] & splits["calibration"]),
        "train_threshold_validation": len(
            splits["train"] & splits["threshold_validation"]
        ),
        "train_test": len(splits["train"] & splits["test"]),
        "calibration_threshold_validation": len(
            splits["calibration"] & splits["threshold_validation"]
        ),
        "calibration_test": len(splits["calibration"] & splits["test"]),
        "threshold_validation_test": len(
            splits["threshold_validation"] & splits["test"]
        ),
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
        "missing_calibration_labels": sorted(
            set(calibration["intent"]) - train_labels
        ),
        "missing_threshold_validation_labels": sorted(
            set(threshold_val["intent"]) - train_labels
        ),
        "missing_test_labels": sorted(set(test["intent"]) - train_labels),
        "pairwise_text_overlap": pairwise_overlap,
        "conflicting_labels": {
            "train_conflict_count": len(conflicts_train),
            "test_conflict_count": len(conflicts_test),
        },
    }
