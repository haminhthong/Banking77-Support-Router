"""Unit tests verifying operational policy invariants, safety decoupling, and OOD guards."""

import numpy as np
import pytest

from src.banking_router.data.contracts import (
    BANKING77_77_CLASSES,
    DEFAULT_HIGH_RISK_INTENTS,
    get_domain_for_intent,
)
from src.banking_router.routing.ood import OODGuard
from src.banking_router.routing.policy import RoutingPolicy
from src.banking_router.routing.risk import RiskAssessor
from src.banking_router.routing.schemas import IntentPrediction, RiskAssessment
from src.banking_router.routing.service import RoutingService
from src.banking_router.routing.taxonomy import TaxonomyResolver


class MockModel:
    def __init__(self, classes, prob_dist):
        self.classes_ = np.array(classes)
        self.prob_dist = np.array(prob_dist)

    def predict_proba(self, texts):
        return np.array([self.prob_dist for _ in texts])


def test_top_k_parameter_does_not_change_safety_decision():
    """INVARIANT: Client display top_k parameter must NEVER affect the safety engine's escalation decision."""
    classes = sorted(list(BANKING77_77_CLASSES))
    # Create probability distribution where a high-risk class is ranked 4th with prob 0.25
    # (e.g. compromised_card at 0.25 >= high_risk_trigger 0.20)
    prob_dist = np.zeros(len(classes))
    prob_dist[classes.index("card_arrival")] = 0.45
    prob_dist[classes.index("card_delivery_estimate")] = 0.15
    prob_dist[classes.index("card_linking")] = 0.15
    prob_dist[classes.index("compromised_card")] = 0.25

    model = MockModel(classes, prob_dist)
    policy = RoutingPolicy(threshold=0.45, high_risk_trigger=0.20)
    risk_assessor = RiskAssessor(classes=classes, high_risk_trigger=0.20)
    service = RoutingService(model=model, policy=policy, risk_assessor=risk_assessor)

    # Call with top_k=1 (display only 1 candidate)
    res_top1 = service.route("Where is my card?", top_k=1)
    # Call with top_k=5 (display 5 candidates)
    res_top5 = service.route("Where is my card?", top_k=5)

    # Both must detect the threat and escalate to priority_human_review!
    assert res_top1.risk.high_risk_detected is True
    assert res_top5.risk.high_risk_detected is True
    assert res_top1.decision.action == "priority_human_review"
    assert res_top5.decision.action == "priority_human_review"
    assert res_top1.decision.action == res_top5.decision.action


def test_high_risk_candidate_keeps_prediction_intact():
    """INVARIANT: Safety escalation must NOT overwrite or mutate the model's true predicted intent."""
    pred = IntentPrediction(
        intent="card_arrival",
        domain="card_services",
        confidence=0.45,
        margin=0.20,
        entropy=1.2,
    )
    risk = RiskAssessment(
        high_risk_detected=True,
        high_risk_intent="compromised_card",
        high_risk_score=0.25,
        ood_detected=False,
        reason_codes=["HIGH_RISK_CANDIDATE"],
    )

    policy = RoutingPolicy(threshold=0.45, high_risk_trigger=0.20)
    decision = policy.evaluate(pred, risk)

    assert decision.action == "priority_human_review"
    assert decision.queue_id == "fraud_security_queue"
    assert decision.priority == "critical"
    # Prediction intent and domain remain intact!
    assert decision.intent == "card_arrival"
    assert decision.domain == "card_services"


def test_prediction_intent_matches_prediction_domain():
    """INVARIANT: Predicted intent must always match its hierarchical domain mapping."""
    taxonomy = TaxonomyResolver()
    for intent in BANKING77_77_CLASSES:
        expected_domain = get_domain_for_intent(intent)
        resolved_domain = taxonomy.get_domain(intent)
        assert resolved_domain == expected_domain


def test_policy_output_partition_is_exhaustive():
    """INVARIANT: Policy action is strictly partitioned into auto_route, human_review, or priority_human_review."""
    policy = RoutingPolicy(threshold=0.50)
    pred = IntentPrediction(
        intent="card_arrival",
        domain="card_services",
        confidence=0.85,
        margin=0.50,
        entropy=0.5,
    )
    risk_safe = RiskAssessment(False, None, 0.0, False)
    dec_safe = policy.evaluate(pred, risk_safe)

    pred_low = IntentPrediction(
        intent="card_arrival",
        domain="card_services",
        confidence=0.30,
        margin=0.05,
        entropy=3.0,
    )
    dec_low = policy.evaluate(pred_low, risk_safe)

    risk_high = RiskAssessment(True, "compromised_card", 0.40, False, ["HIGH_RISK_INTENT"])
    dec_high = policy.evaluate(pred, risk_high)

    allowed_actions = {"auto_route", "human_review", "priority_human_review"}
    for dec in (dec_safe, dec_low, dec_high):
        assert dec.action in allowed_actions


def test_auto_route_and_escalation_mutually_exclusive():
    """INVARIANT: A ticket can never be simultaneously auto_routed and human reviewed."""
    policy = RoutingPolicy(threshold=0.50)
    pred = IntentPrediction(
        intent="card_arrival",
        domain="card_services",
        confidence=0.85,
        margin=0.50,
        entropy=0.5,
    )
    risk = RiskAssessment(False, None, 0.0, False)
    decision = policy.evaluate(pred, risk)

    if decision.action == "auto_route":
        assert decision.requires_human_review is False
        assert decision.abstained is False
    else:
        assert decision.requires_human_review is True


def test_ood_never_auto_routes():
    """INVARIANT: An out-of-scope query must NEVER be auto-routed to operational queues."""
    policy = RoutingPolicy(threshold=0.50)
    pred = IntentPrediction(
        intent="card_arrival",
        domain="card_services",
        confidence=0.99,  # Even with high classifier confidence
        margin=0.90,
        entropy=0.1,
    )
    risk_ood = RiskAssessment(
        high_risk_detected=False,
        high_risk_intent=None,
        high_risk_score=0.01,
        ood_detected=True,
        reason_codes=["OUT_OF_SCOPE_QUERY"],
    )

    decision = policy.evaluate(pred, risk_ood)
    assert decision.action == "human_review"
    assert decision.queue_id == "general_human_review_queue"
    assert decision.requires_human_review is True
    assert decision.action != "auto_route"
