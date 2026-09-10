"""Smoke tests cho calibration, PII và routing policy."""

import numpy as np

from src.banking_router.modeling.training import calculate_ece, calculate_entropy
from src.banking_router.routing.policy import RoutingPolicy
from src.banking_router.routing.schemas import IntentPrediction, QueuePrediction, SensitiveCaseAssessment
from src.banking_router.telemetry.privacy import redact_pii


def prediction(confidence: float = 0.85, margin: float = 0.50) -> IntentPrediction:
    return IntentPrediction("card_arrival", "card_services", confidence, margin, 0.5)


def safe_case() -> SensitiveCaseAssessment:
    return SensitiveCaseAssessment(False, None, 0.0, False)


def test_calculate_ece_and_entropy():
    assert calculate_ece(np.array([1.0, 1.0]), np.array(["a", "b"]), np.array(["a", "b"])) == 0.0
    entropy = calculate_entropy(np.array([[0.25, 0.25, 0.25, 0.25]]))[0]
    assert entropy > 1.0


def test_redact_pii():
    raw = "My card 1234 5678 9012 3456 was stolen. Email user@test.com"
    cleaned = redact_pii(raw)
    assert "1234" not in cleaned
    assert "[CARD_NUMBER]" in cleaned
    assert "[EMAIL]" in cleaned


def test_policy_routes_low_confidence_to_human_review():
    decision = RoutingPolicy(queue_threshold=0.60).evaluate(prediction(0.40, 0.10), safe_case())
    assert decision.action == "human_review"
    assert decision.requires_human_review is True
    assert decision.reason_codes == ["LOW_CONFIDENCE"]


def test_policy_prioritizes_sensitive_case_before_confidence():
    sensitive = SensitiveCaseAssessment(
        True, "compromised_card", 0.25, False, ["SENSITIVE_INTENT_MASS"],
        0.40, "account_compromise", {"account_compromise": 0.40},
    )
    decision = RoutingPolicy(queue_threshold=0.90).evaluate(prediction(0.35), sensitive)
    assert decision.action == "priority_human_review"
    assert decision.queue_id == "sensitive_case_review_queue"


def test_policy_rejects_scope_even_with_high_confidence():
    decision = RoutingPolicy(queue_threshold=0.50).evaluate(
        prediction(0.99), safe_case(), scope_detected=True, scope_reasons=["OUT_OF_SCOPE_QUERY"]
    )
    assert decision.action == "human_review"
    assert decision.queue_id == "general_human_review_queue"


def test_policy_auto_routes_confident_queue():
    queue = QueuePrediction("card_queue", 0.90, 0.40, {"card_queue": 0.90})
    decision = RoutingPolicy(queue_threshold=0.80).evaluate(prediction(), safe_case(), queue_prediction=queue)
    assert decision.action == "auto_route"
    assert decision.queue_id == "card_queue"
    assert decision.requires_human_review is False
