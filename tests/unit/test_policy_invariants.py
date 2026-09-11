"""Kiểm thử invariant cho luồng xác suất, calibration, PII và routing policy."""

from __future__ import annotations

import numpy as np

from src.banking_router.data.contracts import (
    BANKING77_77_CLASSES,
    get_domain_for_intent,
)
from src.banking_router.evaluation.metrics import entropy, expected_calibration_error
from src.banking_router.routing.escalation import SensitiveIntentGuard
from src.banking_router.routing.policy import RoutingPolicy
from src.banking_router.routing.schemas import (
    IntentPrediction,
    QueuePrediction,
    SensitiveCaseAssessment,
)
from src.banking_router.routing.scope import ScopeGuard
from src.banking_router.routing.service import RoutingService
from src.banking_router.routing.taxonomy import TaxonomyResolver
from src.banking_router.telemetry.privacy import redact_pii


class MockModel:
    def __init__(self, probabilities: list[float] | np.ndarray) -> None:
        self.classes_ = np.asarray(BANKING77_77_CLASSES)
        self.probabilities = np.asarray(probabilities, dtype=float)

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        return np.asarray([self.probabilities for _ in texts])


def _sample_prediction(
    confidence: float = 0.85, margin: float = 0.50
) -> IntentPrediction:
    return IntentPrediction("card_arrival", "card_services", confidence, margin, 0.5)


def _safe_case() -> SensitiveCaseAssessment:
    return SensitiveCaseAssessment(False, None, 0.0, False)


def test_calibration_error_and_entropy() -> None:
    """Kiểm tra độ chính xác của hàm tính ECE và Shannon entropy."""
    assert (
        expected_calibration_error(
            np.array([1.0, 1.0]),
            np.array(["a", "b"]),
            np.array(["a", "b"]),
        )
        == 0.0
    )
    entropy_value = entropy(np.array([0.25, 0.25, 0.25, 0.25]))
    assert float(entropy_value) > 1.0


def test_redact_pii() -> None:
    """Kiểm tra che định danh cá nhân và số thẻ trong log telemetry."""
    raw = "My card 1234 5678 9012 3456 was stolen. Email user@test.com"
    cleaned = redact_pii(raw)
    assert "1234" not in cleaned
    assert "[CARD_NUMBER]" in cleaned
    assert "[EMAIL]" in cleaned


def test_top_k_does_not_change_sensitive_decision() -> None:
    """Invariant: tham số top_k không được làm thay đổi quyết định sensitive review."""
    probabilities = np.zeros(len(BANKING77_77_CLASSES))
    probabilities[BANKING77_77_CLASSES.index("card_arrival")] = 0.45
    probabilities[BANKING77_77_CLASSES.index("card_delivery_estimate")] = 0.15
    probabilities[BANKING77_77_CLASSES.index("card_linking")] = 0.15
    probabilities[BANKING77_77_CLASSES.index("compromised_card")] = 0.25
    model = MockModel(probabilities)
    taxonomy = TaxonomyResolver()
    service = RoutingService(
        model=model,
        policy=RoutingPolicy(queue_threshold=0.45, taxonomy_resolver=taxonomy),
        sensitive_guard=SensitiveIntentGuard(model.classes_, taxonomy=taxonomy),
        scope_guard=ScopeGuard(),
        taxonomy=taxonomy,
    )
    first = service.route("Where is my card?", top_k=1)
    fifth = service.route("Where is my card?", top_k=5)
    assert first.sensitive_case.requires_priority_review is True
    assert fifth.sensitive_case.requires_priority_review is True
    assert first.decision.action == fifth.decision.action == "priority_human_review"


def test_sensitive_review_does_not_overwrite_top_intent() -> None:
    """Invariant: escalations do not mutate predicted fine intent."""
    prediction = IntentPrediction("card_arrival", "card_services", 0.45, 0.20, 1.2)
    sensitive = SensitiveCaseAssessment(
        True,
        "compromised_card",
        0.25,
        False,
        ["SENSITIVE_INTENT_MASS"],
        0.25,
        "account_compromise",
        {"account_compromise": 0.25},
    )
    decision = RoutingPolicy(queue_threshold=0.45).evaluate(prediction, sensitive)
    assert decision.action == "priority_human_review"
    assert decision.intent == "card_arrival"
    assert decision.domain == "card_services"


def test_taxonomy_domain_mapping_is_complete() -> None:
    """Mọi intent thuộc 77 lớp phải có domain mapping nhất quán."""
    taxonomy = TaxonomyResolver()
    for intent in BANKING77_77_CLASSES:
        assert taxonomy.get_domain(intent) == get_domain_for_intent(intent)


def test_policy_action_partition_and_scope_guard() -> None:
    """Kiểm tra phân định action của policy và chặn truy vấn ngoài phạm vi."""
    taxonomy = TaxonomyResolver()
    policy = RoutingPolicy(queue_threshold=0.50, taxonomy_resolver=taxonomy)
    safe = _safe_case()
    prediction = _sample_prediction(0.85, 0.50)
    auto = policy.evaluate(prediction, safe)
    review = policy.evaluate(
        prediction, safe, scope_detected=True, scope_reasons=["OUT_OF_SCOPE_QUERY"]
    )
    assert auto.action in {"auto_route", "human_review", "priority_human_review"}
    assert review.action == "human_review"
    assert review.requires_human_review is True


def test_policy_routes_low_confidence_to_human_review() -> None:
    """Độ tin cậy thấp hơn ngưỡng queue phải chuyển sang human review."""
    decision = RoutingPolicy(queue_threshold=0.60).evaluate(
        _sample_prediction(0.40, 0.10), _safe_case()
    )
    assert decision.action == "human_review"
    assert decision.requires_human_review is True
    assert decision.reason_codes == ["LOW_CONFIDENCE"]


def test_policy_prioritizes_sensitive_case_before_confidence() -> None:
    """Tín hiệu ca nhạy cảm phải được ưu tiên trước ngưỡng độ tin cậy."""
    sensitive = SensitiveCaseAssessment(
        True,
        "compromised_card",
        0.25,
        False,
        ["SENSITIVE_INTENT_MASS"],
        0.40,
        "account_compromise",
        {"account_compromise": 0.40},
    )
    decision = RoutingPolicy(queue_threshold=0.90).evaluate(
        _sample_prediction(0.35, 0.10), sensitive
    )
    assert decision.action == "priority_human_review"
    assert decision.queue_id == "sensitive_case_review_queue"


def test_policy_auto_routes_confident_queue() -> None:
    """Queue có độ tin cậy vượt ngưỡng được phép tự động route."""
    queue = QueuePrediction("card_queue", 0.90, 0.40, {"card_queue": 0.90})
    decision = RoutingPolicy(queue_threshold=0.80).evaluate(
        _sample_prediction(), _safe_case(), queue_prediction=queue
    )
    assert decision.action == "auto_route"
    assert decision.queue_id == "card_queue"
    assert decision.requires_human_review is False
