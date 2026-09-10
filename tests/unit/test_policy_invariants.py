"""Kiểm thử invariant cho luồng xác suất và routing policy."""

import numpy as np

from src.banking_router.data.contracts import (
    BANKING77_77_CLASSES,
    get_domain_for_intent,
)
from src.banking_router.routing.escalation import SensitiveIntentGuard
from src.banking_router.routing.policy import RoutingPolicy
from src.banking_router.routing.schemas import IntentPrediction, SensitiveCaseAssessment
from src.banking_router.routing.scope import ScopeGuard
from src.banking_router.routing.service import RoutingService
from src.banking_router.routing.taxonomy import TaxonomyResolver


class MockModel:
    def __init__(self, probabilities):
        self.classes_ = np.asarray(BANKING77_77_CLASSES)
        self.probabilities = np.asarray(probabilities, dtype=float)

    def predict_proba(self, texts):
        return np.asarray([self.probabilities for _ in texts])


def test_top_k_does_not_change_sensitive_decision():
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


def test_sensitive_review_does_not_overwrite_top_intent():
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


def test_taxonomy_domain_mapping_is_complete():
    taxonomy = TaxonomyResolver()
    for intent in BANKING77_77_CLASSES:
        assert taxonomy.get_domain(intent) == get_domain_for_intent(intent)


def test_policy_action_partition_and_scope_guard():
    taxonomy = TaxonomyResolver()
    policy = RoutingPolicy(queue_threshold=0.50, taxonomy_resolver=taxonomy)
    safe = SensitiveCaseAssessment(False, None, 0.0, False)
    prediction = IntentPrediction("card_arrival", "card_services", 0.85, 0.50, 0.5)
    auto = policy.evaluate(prediction, safe)
    review = policy.evaluate(
        prediction, safe, scope_detected=True, scope_reasons=["OUT_OF_SCOPE_QUERY"]
    )
    assert auto.action in {"auto_route", "human_review", "priority_human_review"}
    assert review.action == "human_review"
    assert review.requires_human_review is True
