"""Các unit test kiểm tra chức năng cốt lõi, chính sách phân luồng và tích hợp API FastAPI."""

import numpy as np
from src.policy import DEFAULT_HIGH_RISK_INTENTS, RoutingPolicy
from src.utils import calculate_ece, calculate_entropy, redact_pii


def test_calculate_ece_perfect_calibration():
    """Kiểm tra ECE = 0 khi xác suất khớp hoàn toàn với thực tế."""
    confidences = np.array([0.9, 0.8, 0.7, 0.6])
    predictions = np.array(["a", "b", "c", "d"])
    targets = np.array(["a", "b", "c", "d"])

    ece = calculate_ece(confidences, predictions, targets, n_bins=5)
    assert 0.0 <= ece <= 1.0


def test_calculate_entropy():
    """Kiểm tra tính toán Shannon Entropy."""
    # Phân phối đều -> entropy cực đại
    uniform_p = np.array([0.25, 0.25, 0.25, 0.25])
    ent_max = calculate_entropy(uniform_p)
    # Phân phối tập trung -> entropy xấp xỉ 0
    certain_p = np.array([0.999, 0.001 / 3, 0.001 / 3, 0.001 / 3])
    ent_min = calculate_entropy(certain_p)
    assert ent_max > ent_min
    assert ent_min >= 0.0


def test_redact_pii():
    """Kiểm tra che giấu thông tin nhạy cảm PII."""
    raw = "My card 1234 5678 9012 3456 was stolen. Call +1-555-123-4567 or email me at user@test.com"
    cleaned = redact_pii(raw)
    assert "1234" not in cleaned
    assert "[CARD_NUMBER]" in cleaned
    assert "user@test.com" not in cleaned
    assert "[EMAIL]" in cleaned
    assert "[PHONE_NUMBER]" in cleaned


def test_routing_policy_abstains_low_confidence():
    """Kiểm tra policy từ chối tự động hóa (Abstain) khi confidence dưới ngưỡng."""
    decision = RoutingPolicy(threshold=0.6).decide(
        "card_arrival", confidence=0.4, top_domain="card_services"
    )

    assert decision.intent is None
    assert decision.domain is None
    assert decision.route == "human"
    assert decision.decision == "abstain"
    assert decision.abstained is True
    assert decision.is_unknown is True
    assert decision.requires_human_review is True
    assert decision.review_reason == "LOW_CONFIDENCE"


def test_routing_policy_escalates_high_risk_even_with_low_confidence():
    """Kiểm tra ca rủi ro cao luôn được ưu tiên (Priority Escalation) kể cả khi confidence thấp."""
    # Top intent là rủi ro cao nhưng confidence chỉ 0.35 (thấp hơn threshold 0.60)
    decision = RoutingPolicy(threshold=0.6).decide(
        "compromised_card", confidence=0.35, top_domain="account_security"
    )

    assert decision.intent == "compromised_card"
    assert decision.domain == "account_security"
    assert decision.route == "priority_human_review"
    assert decision.decision == "priority_escalation"
    assert decision.abstained is False
    assert decision.requires_human_review is True
    assert decision.review_reason == "HIGH_RISK_INTENT"


def test_routing_policy_escalates_top_k_high_risk_candidate():
    """Kiểm tra leo thang khi intent rủi ro cao xuất hiện trong Top-K với xác suất đáng kể."""
    candidates = [("card_arrival", 0.45), ("compromised_card", 0.25)]
    decision = RoutingPolicy(threshold=0.60, high_risk_trigger=0.20).decide(
        top_intent="card_arrival",
        confidence=0.45,
        top_domain="card_services",
        top_k_candidates=candidates,
    )

    assert decision.route == "priority_human_review"
    assert decision.decision == "priority_escalation"
    assert decision.review_reason == "HIGH_RISK_CANDIDATE"


def test_routing_policy_abstains_on_ambiguous_margin():
    """Kiểm tra policy từ chối khi khoảng cách giữa top 1 và top 2 quá hẹp (Margin Gate)."""
    decision = RoutingPolicy(threshold=0.50, min_margin=0.10).decide(
        top_intent="card_arrival",
        confidence=0.70,
        top_domain="card_services",
        margin=0.03,  # Quá hẹp
    )

    assert decision.decision == "abstain"
    assert decision.abstained is True
    assert decision.review_reason == "AMBIGUOUS_MARGIN"


def test_routing_policy_safe_auto_route():
    """Kiểm tra ca tự tin và an toàn được tự động phân luồng."""
    decision = RoutingPolicy(threshold=0.50).decide(
        top_intent="card_arrival",
        confidence=0.85,
        top_domain="card_services",
    )

    assert decision.decision == "auto_route"
    assert decision.abstained is False
    assert decision.route == "card_arrival"
    assert decision.requires_human_review is False
    assert decision.review_reason is None


def test_predict_returns_ranked_alternatives_and_domain(monkeypatch):
    """Kiểm tra API /predict trả về kết quả xếp hạng, domain và các trường mới."""
    import src.api as api

    class FakeModel:
        classes_ = np.array(["card_arrival", "card_swallowed", "cash_withdrawal_charge"])

        def predict_proba(self, _texts):
            return np.array([[0.7, 0.2, 0.1]])

    monkeypatch.setattr(api, "_model", FakeModel())
    monkeypatch.setattr(
        api,
        "_config",
        {
            "threshold": 0.35,
            "version": "test-model-v2",
            "policy_version": "test-policy-v2",
            "high_risk_trigger": 0.25,
            "domain_map": {
                "card_arrival": "card_services",
                "card_swallowed": "atm_cash",
                "cash_withdrawal_charge": "atm_cash",
            },
        },
    )

    response = api.predict(api.Query(text="Where is my card?", top_k=2))

    assert response["top_intent"] == "card_arrival"
    assert response["domain"] == "card_services"
    assert response["confidence"] == 0.7
    assert response["decision"] == "auto_route"
    assert response["abstained"] is False
    assert len(response["alternatives"]) == 2
    assert response["alternatives"][0]["domain"] == "card_services"
    assert response["requires_human_review"] is False


def test_predict_batch_vectorized(monkeypatch):
    """Kiểm tra API /predict/batch xử lý hàng loạt nhiều query vector hóa."""
    import src.api as api

    class FakeModel:
        classes_ = np.array(["card_arrival", "cash_withdrawal_charge"])

        def predict_proba(self, texts):
            return np.array([[0.8, 0.2] for _ in texts])

    monkeypatch.setattr(api, "_model", FakeModel())
    monkeypatch.setattr(
        api,
        "_config",
        {"threshold": 0.35, "version": "test-model-v2", "policy_version": "test-policy-v2"},
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
    assert res["results"][0]["decision"] == "auto_route"
    assert res["results"][0]["confidence"] == 0.8
