"""Kiểm thử hồi quy cho queue projection, sensitive mass và scope."""

import numpy as np

from src.banking_router.data.contracts import BANKING77_77_CLASSES
from src.banking_router.routing.escalation import SensitiveIntentGuard
from src.banking_router.routing.policy import RoutingPolicy
from src.banking_router.routing.scope import ScopeGuard
from src.banking_router.routing.service import RoutingService
from src.banking_router.routing.taxonomy import TaxonomyResolver


class MockModel:
    def __init__(self, probabilities):
        self.classes_ = np.asarray(BANKING77_77_CLASSES)
        self.probabilities = np.asarray(probabilities, dtype=float)

    def predict_proba(self, texts):
        return np.asarray([self.probabilities for _ in texts])


def test_queue_probability_is_aggregated_across_intents():
    probabilities = np.zeros(len(BANKING77_77_CLASSES))
    for intent, value in {
        "pending_transfer": 0.32,
        "receiving_money": 0.28,
        "transfer_timing": 0.20,
        "card_arrival": 0.20,
    }.items():
        probabilities[BANKING77_77_CLASSES.index(intent)] = value
    taxonomy = TaxonomyResolver()
    model = MockModel(probabilities)
    service = RoutingService(
        model=model,
        policy=RoutingPolicy(
            queue_threshold=0.75, queue_margin=0.10, taxonomy_resolver=taxonomy
        ),
        sensitive_guard=SensitiveIntentGuard(model.classes_, taxonomy=taxonomy),
        taxonomy=taxonomy,
    )
    result = service.route("pending transfer timing")
    assert result.queue_prediction.queue == "transfers_queue"
    assert abs(result.queue_prediction.confidence - 0.80) < 1e-6
    assert result.decision.action == "auto_route"


def test_sensitive_guard_uses_probability_mass_not_only_top_class():
    probabilities = np.zeros(len(BANKING77_77_CLASSES))
    probabilities[BANKING77_77_CLASSES.index("compromised_card")] = 0.19
    probabilities[BANKING77_77_CLASSES.index("lost_or_stolen_card")] = 0.18
    probabilities[BANKING77_77_CLASSES.index("card_arrival")] = 0.63
    guard = SensitiveIntentGuard(
        BANKING77_77_CLASSES, taxonomy=TaxonomyResolver(), sensitive_trigger=0.36
    )
    assessment = guard.assess(probabilities)
    assert assessment.requires_priority_review is True
    assert abs(assessment.sensitive_probability_mass - 0.37) < 1e-6
    assert "SENSITIVE_INTENT_MASS" in assessment.reason_codes


def test_scope_guard_allows_confident_typo():
    guard = ScopeGuard(
        vocabulary={"transfer", "pending"}, low_confidence_threshold=0.22
    )
    assert guard.detect("my trasfer is pendng", confidence=0.65) == (False, [])


def test_scope_guard_rejects_unknown_low_confidence_query():
    guard = ScopeGuard(vocabulary={"card", "transfer"}, low_confidence_threshold=0.22)
    assert guard.detect("mortgage refinancing advice", confidence=0.10) == (
        True,
        ["OUT_OF_SCOPE_QUERY"],
    )
