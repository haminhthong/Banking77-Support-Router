"""Kiểm thử hợp đồng dữ liệu và tính nhất quán của taxonomy."""

from pathlib import Path

import pandas as pd
from src.banking_router.data import (
    BANKING77_77_CLASSES,
    INTENT_TO_DOMAIN,
    audit_conflicting_labels,
    get_domain_for_intent,
    load_training_splits,
    read_raw_dataset,
    summarize_split_quality,
)


def test_read_banking77_normalizes_columns_and_order(tmp_path: Path):
    """Kiểm tra đọc file CSV bất kỳ và chuẩn hóa cột về ['text', 'intent']."""
    csv_path = tmp_path / "sample.csv"
    csv_path.write_text(
        "category,text,unused\ncard_arrival,Where is my card?,ignored\n",
        encoding="utf-8",
    )

    result = read_raw_dataset(csv_path)

    assert list(result.columns) == ["text", "intent"]
    assert result.iloc[0].to_dict() == {
        "text": "Where is my card?",
        "intent": "card_arrival",
    }


def test_exact_77_intent_to_domain_taxonomy_complete():
    """Bắt buộc đủ 77 intent BANKING77 trong bảng taxonomy khi chạy CI."""
    assert len(BANKING77_77_CLASSES) == 77
    assert len(INTENT_TO_DOMAIN) == 77
    assert set(INTENT_TO_DOMAIN.keys()) == set(BANKING77_77_CLASSES), (
        "Taxonomy không khớp hoàn toàn với danh sách 77 nhãn chuẩn!"
    )


def test_get_domain_for_real_banking77_intents():
    """Kiểm tra ánh xạ các nhãn thật của BANKING77, đặc biệt là top_up_* và các nhãn đặc biệt."""
    assert get_domain_for_intent("card_arrival") == "card_services"
    assert get_domain_for_intent("top_up_failed") == "topup_recharge"
    assert get_domain_for_intent("automatic_top_up") == "topup_recharge"
    assert get_domain_for_intent("top_up_by_card_charge") == "topup_recharge"
    assert get_domain_for_intent("cash_withdrawal_not_recognised") == "atm_cash"
    assert get_domain_for_intent("transfer_fee_charged") == "transfers_payments"
    assert get_domain_for_intent("Refund_not_showing_up") == "transactions_refunds"
    assert get_domain_for_intent("reverted_card_payment?") == "transactions_refunds"
    assert get_domain_for_intent("apple_pay_or_google_pay") == "app_features"
    assert get_domain_for_intent("country_support") == "international_services"
    # Kiểm tra fallback cho intent không tồn tại.
    assert get_domain_for_intent("non_existent_intent") == "general_banking"


def test_split_quality_detects_pairwise_text_overlap():
    """Kiểm tra báo cáo chất lượng phát hiện văn bản bị trùng giữa các cặp split."""
    train = pd.DataFrame({"text": ["same", "train"], "intent": ["a", "b"]})
    cal = pd.DataFrame({"text": ["calibration"], "intent": ["a"]})
    threshold_val = pd.DataFrame({"text": ["validation"], "intent": ["a"]})
    test = pd.DataFrame({"text": ["same"], "intent": ["a"]})

    summary = summarize_split_quality(train, cal, threshold_val, test)

    assert summary["pairwise_text_overlap"]["train_test"] == 1
    assert summary["pairwise_text_overlap"]["train_calibration"] == 0
    assert summary["missing_test_labels"] == []


def test_training_splits_have_distinct_roles_and_pairwise_disjoint():
    """Tất cả các split train, calibration, threshold-validation phải hoàn toàn độc lập."""
    train, calibration, threshold_validation, test = load_training_splits()
    sets = [set(frame["text"]) for frame in (train, calibration, threshold_validation)]
    assert sets[0].isdisjoint(sets[1])
    assert sets[0].isdisjoint(sets[2])
    assert sets[1].isdisjoint(sets[2])
    assert len(test) > 0


def test_audit_conflicting_labels():
    """Kiểm tra phát hiện mẫu câu xung đột nhãn (cùng text nhưng khác intent)."""
    df = pd.DataFrame(
        {
            "text": ["Where is my card?", "Where is my card?", "Transfer money"],
            "intent": [
                "card_arrival",
                "card_delivery_estimate",
                "transfer_into_account",
            ],
        }
    )
    conflicts = audit_conflicting_labels(df)
    assert len(conflicts) == 1
    assert conflicts[0]["text"] == "Where is my card?"
    assert conflicts[0]["intents"] == ["card_arrival", "card_delivery_estimate"]
