"""Operational Routing Policy Engine implementing multi-tier risk and uncertainty triage."""

from __future__ import annotations

from typing import Any, Sequence
from .schemas import IntentPrediction, QueuePrediction, RiskAssessment, RoutingDecision
from .taxonomy import TaxonomyResolver


class RoutingPolicy:
    """Multi-tiered decision engine for banking customer support triage.

    Strict Precedence Order:
    1. SECURITY RISK: High-risk security queries escalate to priority review.
    2. OUT-OF-DISTRIBUTION (OOD): Queries outside banking scope route to human review.
    3. UNCERTAINTY GATE: Queries with low confidence, narrow margin, or high entropy abstain.
    4. SAFE AUTO-ROUTE: High-confidence, in-scope queries route to designated operations queues.
    """

    def __init__(
        self,
        threshold: float = 0.45,
        queue_threshold: float | None = None,
        queue_margin: float | None = None,
        min_margin: float | None = None,
        max_entropy: float | None = None,
        high_risk_trigger: float = 0.20,
        taxonomy_resolver: TaxonomyResolver | None = None,
    ) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"threshold must be in [0.0, 1.0], got {threshold}")
        if not 0.0 <= high_risk_trigger <= 1.0:
            raise ValueError(f"high_risk_trigger must be in [0.0, 1.0], got {high_risk_trigger}")
        if min_margin is not None and not 0.0 <= min_margin <= 1.0:
            raise ValueError(f"min_margin must be in [0.0, 1.0], got {min_margin}")
        if max_entropy is not None and max_entropy < 0.0:
            raise ValueError(f"max_entropy must be >= 0.0, got {max_entropy}")

        self.threshold = float(threshold)
        self.queue_threshold = float(queue_threshold if queue_threshold is not None else threshold)
        self.queue_margin = float(queue_margin if queue_margin is not None else (min_margin or 0.0))
        self.min_margin = float(min_margin) if min_margin is not None else None
        self.max_entropy = float(max_entropy) if max_entropy is not None else None
        self.high_risk_trigger = float(high_risk_trigger)
        self.taxonomy = taxonomy_resolver or TaxonomyResolver()

    def evaluate(
        self,
        prediction: IntentPrediction,
        risk: RiskAssessment,
        queue_prediction: QueuePrediction | None = None,
    ) -> RoutingDecision:
        """Evaluate triage policy on decoupled prediction and risk signals."""
        # 1. High-Risk Security Override
        if risk.high_risk_detected or risk.critical_probability >= self.high_risk_trigger:
            return RoutingDecision(
                action="priority_human_review",
                queue_id="fraud_security_queue",
                priority="critical",
                requires_human_review=True,
                reason_codes=risk.reason_codes,
                intent=prediction.intent,
                domain=prediction.domain,
            )

        # 2. Out-of-Distribution / Out-of-Scope Gate
        if risk.ood_detected:
            return RoutingDecision(
                action="human_review",
                queue_id="general_human_review_queue",
                priority="normal",
                requires_human_review=True,
                reason_codes=risk.reason_codes or ["OUT_OF_SCOPE_QUERY"],
                intent=None,
                domain=None,
            )

        # 3. Uncertainty Gate (Selective Classification)
        # Queue confidence is the business decision signal.  The intent score
        # remains useful for explanation but must not gate routing directly.
        if (
            (queue_prediction is not None and queue_prediction.confidence < self.queue_threshold)
            or (queue_prediction is None and prediction.confidence < self.threshold)
        ):
            return RoutingDecision(
                action="human_review",
                queue_id="general_human_review_queue",
                priority="normal",
                requires_human_review=True,
                reason_codes=["LOW_QUEUE_CONFIDENCE" if queue_prediction is not None else "LOW_CONFIDENCE"],
                intent=None,
                domain=None,
            )

        if (
            (queue_prediction is not None and queue_prediction.margin < self.queue_margin)
            or (queue_prediction is None and self.min_margin is not None and prediction.margin < self.min_margin)
        ):
            return RoutingDecision(
                action="human_review",
                queue_id="general_human_review_queue",
                priority="normal",
                requires_human_review=True,
                reason_codes=["AMBIGUOUS_QUEUE_MARGIN" if queue_prediction is not None else "AMBIGUOUS_MARGIN"],
                intent=None,
                domain=None,
            )

        if self.max_entropy is not None and prediction.entropy > self.max_entropy:
            return RoutingDecision(
                action="human_review",
                queue_id="general_human_review_queue",
                priority="normal",
                requires_human_review=True,
                reason_codes=["HIGH_ENTROPY"],
                intent=None,
                domain=None,
            )

        # 4. Safe Operational Auto-Routing
        queue = queue_prediction.queue if queue_prediction is not None else self.taxonomy.get_queue(prediction.intent)
        priority = self.taxonomy.get_priority(prediction.intent)
        return RoutingDecision(
            action="auto_route",
            queue_id=queue,
            priority=priority,
            requires_human_review=False,
            reason_codes=[],
            intent=prediction.intent,
            domain=prediction.domain,
        )

    def decide(
        self,
        top_intent: str,
        confidence: float,
        top_domain: str | None = None,
        top_k_candidates: (
            list[tuple[str, float]] | list[tuple[str, str, float]] | None
        ) = None,
        margin: float | None = None,
        entropy: float | None = None,
    ) -> RoutingDecision:
        """Backward-compatible decide interface supporting legacy test calls."""
        from ..data.contracts import DEFAULT_HIGH_RISK_INTENTS, get_domain_for_intent

        domain = top_domain or get_domain_for_intent(top_intent)
        resolved_margin = margin if margin is not None else 1.0
        resolved_entropy = entropy if entropy is not None else 0.0

        pred = IntentPrediction(
            intent=top_intent,
            domain=domain,
            confidence=confidence,
            margin=resolved_margin,
            entropy=resolved_entropy,
        )

        high_risk_detected = False
        high_risk_intent = None
        high_risk_score = 0.0
        reason_codes: list[str] = []

        if top_intent in self.taxonomy.get_critical_intents() or top_intent in DEFAULT_HIGH_RISK_INTENTS:
            high_risk_detected = True
            high_risk_intent = top_intent
            high_risk_score = confidence
            reason_codes.append("HIGH_RISK_INTENT")
        elif top_k_candidates:
            for cand in top_k_candidates:
                if len(cand) == 3:
                    cand_intent, _, cand_prob = cand
                else:
                    cand_intent, cand_prob = cand
                if (
                    cand_intent in DEFAULT_HIGH_RISK_INTENTS
                    and cand_prob >= self.high_risk_trigger
                ):
                    high_risk_detected = True
                    high_risk_intent = cand_intent
                    high_risk_score = cand_prob
                    reason_codes.append("HIGH_RISK_CANDIDATE")
                    break

        risk = RiskAssessment(
            high_risk_detected=high_risk_detected,
            high_risk_intent=high_risk_intent,
            high_risk_score=high_risk_score,
            ood_detected=False,
            reason_codes=reason_codes,
        )

        return self.evaluate(pred, risk)
