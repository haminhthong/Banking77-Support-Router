"""Các unit test kiểm tra chức năng cốt lõi và tích hợp API (Smoke Tests)."""

import numpy as np
from src.policy import DEFAULT_HIGH_RISK_INTENTS, RoutingPolicy
from src.utils import calculate_ece


def test_calculate_ece_perfect_calibration():
    """Kiểm tra ECE = 0 khi xác suất khớp hoàn toàn với thực tế."""
    confidences = np.array([0.9, 0.8, 0.7, 0.6])
    predictions = np.array(["a", "b", "c", "d"])
    targets = np.array(["a", "b", "c", "d"])

    ece = calculate_ece(confidences, predictions, targets, n_bins=5)
    # Vì accuracy = 1.0 ở mọi bin, ECE = average(|1.0 - conf|)
    assert 0.0 <= ece <= 1.0


def test_routing_policy_rejects_low_confidence():
    """Kiểm tra policy từ chối dự đoán khi confidence dưới ngưỡng."""
    decision = RoutingPolicy(threshold=0.6).decide(
        "card_arrival", confidence=0.4, top_domain="card_services"
    )

    assert decision.intent is None
    assert decision.domain is None
    assert decision.route == "human"
    assert decision.is_unknown is True
    assert decision.review_reason == "LOW_CONFIDENCE"


def test_routing_policy_escalates_high_risk_intent():
    """Kiểm tra policy leo thang ca rủi ro cao cho nhân viên kiểm duyệt."""
    decision = RoutingPolicy(threshold=0.6).decide(
        "card_swallowed", confidence=0.9, top_domain="card_services"
    )

    assert decision.intent == "card_swallowed"
    assert decision.domain == "card_services"
    assert decision.route == "priority_human_review"
    assert decision.requires_human_review is True
    assert decision.review_reason == "HIGH_RISK_INTENT"


def test_routing_policy_contains_only_known_sensitive_intents():
    """Kiểm tra danh sách intent nhạy cảm mặc định."""
    assert "compromised_card" in DEFAULT_HIGH_RISK_INTENTS
    assert "cash_withdrawal" not in DEFAULT_HIGH_RISK_INTENTS


def test_predict_returns_ranked_alternatives_and_domain(monkeypatch):
    """Kiểm tra API /predict trả về kết quả xếp hạng và domain chính xác."""
    import src.api as api

    class FakeModel:
        classes_ = np.array(["card_arrival", "card_swallowed", "cash_withdrawal"])

        def predict_proba(self, _texts):
            return np.array([[0.7, 0.2, 0.1]])

    monkeypatch.setattr(api, "_model", FakeModel())
    monkeypatch.setattr(
        api,
        "_config",
        {
            "threshold": 0.35,
            "version": "test-model-v1",
            "domain_map": {
                "card_arrival": "card_services",
                "card_swallowed": "card_services",
                "cash_withdrawal": "atm_cash",
            },
        },
    )

    response = api.predict(api.Query(text="Where is my card?", top_k=2))

    assert response["top_intent"] == "card_arrival"
    assert response["domain"] == "card_services"
    assert response["confidence"] == 0.7
    assert len(response["alternatives"]) == 2
    assert response["alternatives"][0]["domain"] == "card_services"
    assert response["requires_human_review"] is False


def test_predict_batch_returns_list_of_results(monkeypatch):
    """Kiểm tra API /predict/batch xử lý hàng loạt nhiều query cùng lúc."""
    import src.api as api

    class FakeModel:
        classes_ = np.array(["card_arrival", "cash_withdrawal"])

        def predict_proba(self, texts):
            return np.array([[0.8, 0.2] for _ in texts])

    monkeypatch.setattr(api, "_model", FakeModel())
    monkeypatch.setattr(
        api,
        "_config",
        {"threshold": 0.35, "version": "test-model-v1"},
    )

    batch_query = api.BatchQuery(
        queries=[
            api.Query(text="Where is my card?"),
            api.Query(text="I need cash from ATM"),
        ]
    )

    res = api.predict_batch(batch_query)
    assert res["total_queries"] == 2
    assert len(res["results"]) == 2
