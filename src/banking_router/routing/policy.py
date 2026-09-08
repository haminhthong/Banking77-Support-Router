"""Policy engine dùng chung cho optimizer, API và batch inference."""

from __future__ import annotations

from .schemas import IntentPrediction, QueuePrediction, RiskAssessment, RoutingDecision
from .taxonomy import TaxonomyResolver


class RoutingPolicy:
    """Một thứ tự quyết định duy nhất: risk -> scope -> uncertainty -> route."""

    def __init__(
        self,
        threshold: float = 0.45,
        queue_threshold: float | None = None,
        queue_margin: float | None = None,
        min_margin: float | None = None,
        max_entropy: float | None = None,
        high_risk_trigger: float = 0.20,
        minimum_risk_signal: float = 0.16,
        minimum_ood_security_signal: float = 0.30,
        taxonomy_resolver: TaxonomyResolver | None = None,
    ) -> None:
        for name, value in {
            "threshold": threshold,
            "high_risk_trigger": high_risk_trigger,
        }.items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} phải nằm trong [0, 1]")
        if min_margin is not None and not 0.0 <= min_margin <= 1.0:
            raise ValueError("min_margin phải nằm trong [0, 1]")
        if queue_margin is not None and not 0.0 <= queue_margin <= 1.0:
            raise ValueError("queue_margin phải nằm trong [0, 1]")
        if max_entropy is not None and max_entropy < 0.0:
            raise ValueError("max_entropy phải không âm")
        if not 0.0 <= minimum_risk_signal <= 1.0:
            raise ValueError("minimum_risk_signal phải nằm trong [0, 1]")
        if not 0.0 <= minimum_ood_security_signal <= 1.0:
            raise ValueError("minimum_ood_security_signal phải nằm trong [0, 1]")

        self.threshold = float(threshold)
        self.queue_threshold = float(queue_threshold if queue_threshold is not None else threshold)
        self.queue_margin = float(queue_margin if queue_margin is not None else (min_margin or 0.0))
        self.min_margin = float(min_margin) if min_margin is not None else None
        self.max_entropy = float(max_entropy) if max_entropy is not None else None
        self.high_risk_trigger = float(high_risk_trigger)
        self.minimum_risk_signal = float(minimum_risk_signal)
        self.minimum_ood_security_signal = float(minimum_ood_security_signal)
        self.taxonomy = taxonomy_resolver or TaxonomyResolver()

    @staticmethod
    def _review(reason: str, priority: str = "normal") -> RoutingDecision:
        return RoutingDecision(
            action="human_review",
            queue_id="general_human_review_queue",
            priority=priority,
            requires_human_review=True,
            reason_codes=[reason],
            intent=None,
            domain=None,
        )

    def evaluate(
        self,
        prediction: IntentPrediction,
        risk: RiskAssessment,
        queue_prediction: QueuePrediction | None = None,
    ) -> RoutingDecision:
        """Đánh giá đúng cùng một policy ở cả offline và online."""
        high_risk = risk.high_risk_detected or risk.critical_probability >= self.high_risk_trigger
        # Khi scope đã bị từ chối, một mass nhỏ nhưng rải đều trên các lớp
        # security là tín hiệu model confusion, không đủ để gán fraud queue.
        oos_signal = (
            self.minimum_ood_security_signal
            if "OUT_OF_SCOPE_QUERY" in risk.reason_codes
            else self.minimum_risk_signal
        )
        security_override = not risk.ood_detected or risk.high_risk_score >= oos_signal
        if high_risk and security_override:
            return RoutingDecision(
                action="priority_human_review",
                queue_id=self.taxonomy.get_escalation_queue(risk.risk_category),
                priority="critical",
                requires_human_review=True,
                reason_codes=risk.reason_codes or ["CRITICAL_RISK_MASS"],
                intent=prediction.intent,
                domain=prediction.domain,
            )

        if risk.ood_detected:
            return self._review(risk.reason_codes[0] if risk.reason_codes else "OUT_OF_SCOPE_QUERY")

        if queue_prediction is not None:
            if queue_prediction.confidence < self.queue_threshold:
                return self._review("LOW_QUEUE_CONFIDENCE")
            if queue_prediction.margin < self.queue_margin:
                return self._review("AMBIGUOUS_QUEUE_MARGIN")
        elif prediction.confidence < self.threshold:
            return self._review("LOW_CONFIDENCE")

        if queue_prediction is None and self.min_margin is not None and prediction.margin < self.min_margin:
            return self._review("AMBIGUOUS_MARGIN")
        if self.max_entropy is not None and prediction.entropy > self.max_entropy:
            return self._review("HIGH_ENTROPY")

        queue = queue_prediction.queue if queue_prediction is not None else self.taxonomy.get_queue(prediction.intent)
        return RoutingDecision(
            action="auto_route",
            queue_id=queue,
            priority=self.taxonomy.get_priority(prediction.intent),
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
        top_k_candidates: list[tuple[str, float]] | list[tuple[str, str, float]] | None = None,
        margin: float | None = None,
        entropy: float | None = None,
    ) -> RoutingDecision:
        """Facade cho client cũ; vẫn dùng taxonomy làm nguồn risk."""
        from ..data.contracts import get_domain_for_intent
        from .schemas import RiskAssessment

        domain = top_domain or get_domain_for_intent(top_intent)
        reason_codes: list[str] = []
        high_risk_intent = top_intent if self.taxonomy.is_priority_escalation(top_intent) else None
        high_risk_score = confidence if high_risk_intent else 0.0
        if high_risk_intent:
            reason_codes.append("HIGH_RISK_INTENT")
        if top_k_candidates:
            for candidate in top_k_candidates:
                intent = candidate[0]
                score = float(candidate[-1])
                if self.taxonomy.is_priority_escalation(intent) and score >= self.high_risk_trigger:
                    high_risk_intent = intent
                    high_risk_score = score
                    reason_codes.append("HIGH_RISK_CANDIDATE")
                    break

        risk = RiskAssessment(
            high_risk_detected=high_risk_intent is not None,
            high_risk_intent=high_risk_intent,
            high_risk_score=high_risk_score,
            ood_detected=False,
            reason_codes=reason_codes,
            critical_probability=high_risk_score,
            risk_category=self.taxonomy.get_risk_category(high_risk_intent) if high_risk_intent else None,
        )
        return self.evaluate(
            IntentPrediction(
                intent=top_intent,
                domain=domain,
                confidence=confidence,
                margin=margin if margin is not None else 1.0,
                entropy=entropy if entropy is not None else 0.0,
            ),
            risk,
        )


# Tên rõ nghĩa cho code mới, giữ RoutingPolicy để không gãy API cũ.
RoutingPolicyEngine = RoutingPolicy
