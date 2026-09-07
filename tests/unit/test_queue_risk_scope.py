"""Regression tests for the queue-level safety decision contract."""

import numpy as np

from src.banking_router.data.contracts import BANKING77_77_CLASSES
from src.banking_router.routing.ood import ScopeGuard
from src.banking_router.routing.policy import RoutingPolicy
from src.banking_router.routing.risk import RiskAssessor
from src.banking_router.routing.service import RoutingService
from src.banking_router.routing.taxonomy import TaxonomyResolver


class MockModel:
    def __init__(self, probabilities):
        self.classes_ = np.asarray(BANKING77_77_CLASSES)
        self.probabilities = np.asarray(probabilities, dtype=float)

    def predict_proba(self, texts):
        return np.asarray([self.probabilities for _ in texts])


def test_queue_probability_is_aggregated_across_intents():
    classes = BANKING77_77_CLASSES
    probabilities = np.zeros(len(classes))
    for intent, value in {
        "pending_transfer": 0.32,
        "receiving_money": 0.28,
        "transfer_timing": 0.20,
        "card_arrival": 0.20,
    }.items():
        probabilities[classes.index(intent)] = value

    service = RoutingService(
        model=MockModel(probabilities),
        policy=RoutingPolicy(queue_threshold=0.75, queue_margin=0.10),
        risk_assessor=RiskAssessor(classes, high_risk_trigger=0.40),
    )
    result = service.route("pending transfer timing")

    assert result.queue_prediction is not None
    assert result.queue_prediction.queue == "transfers_queue"
    assert abs(result.queue_prediction.confidence - 0.80) < 1e-6
    assert result.decision.action == "auto_route"


def test_critical_risk_uses_probability_mass_not_max_class():
    classes = BANKING77_77_CLASSES
    probabilities = np.zeros(len(classes))
    probabilities[classes.index("compromised_card")] = 0.19
    probabilities[classes.index("lost_or_stolen_card")] = 0.18
    probabilities[classes.index("card_arrival")] = 0.63

    risk = RiskAssessor(classes, high_risk_trigger=0.36).assess(
        probabilities=probabilities,
        top_intent="card_arrival",
    )

    assert risk.high_risk_detected is True
    assert abs(risk.critical_probability - 0.37) < 1e-6
    assert "CRITICAL_RISK_MASS" in risk.reason_codes


def test_scope_guard_does_not_reject_typo_when_model_is_confident():
    guard = ScopeGuard(vocabulary={"transfer", "pending"}, low_confidence_ood_threshold=0.22)

    is_out, reasons = guard.detect("my trasfer is pendng", confidence=0.65)

    assert is_out is False
    assert reasons == []


def test_scope_guard_rejects_low_confidence_banking_adjacent_unknown():
    guard = ScopeGuard(vocabulary={"card", "transfer"}, low_confidence_ood_threshold=0.22)

    is_out, reasons = guard.detect("mortgage refinancing advice", confidence=0.10)

    assert is_out is True
    assert reasons == ["OUT_OF_SCOPE_QUERY"]
