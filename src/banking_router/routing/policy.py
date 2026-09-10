"""Routing policy dùng chung cho train, đánh giá và FastAPI."""

from __future__ import annotations

from .schemas import IntentPrediction, QueuePrediction, RoutingDecision, SensitiveCaseAssessment
from .taxonomy import TaxonomyResolver


class RoutingPolicy:
    """Thứ tự duy nhất: ca nhạy cảm -> phạm vi -> độ không chắc chắn -> route."""

    def __init__(
        self,
        queue_threshold: float = 0.45,
        queue_margin: float = 0.0,
        min_margin: float | None = None,
        max_entropy: float | None = None,
        sensitive_trigger: float = 0.20,
        minimum_sensitive_signal: float = 0.16,
        minimum_scope_sensitive_signal: float = 0.30,
        taxonomy_resolver: TaxonomyResolver | None = None,
    ) -> None:
        for name, value in {
            "queue_threshold": queue_threshold,
            "queue_margin": queue_margin,
            "sensitive_trigger": sensitive_trigger,
            "minimum_sensitive_signal": minimum_sensitive_signal,
            "minimum_scope_sensitive_signal": minimum_scope_sensitive_signal,
        }.items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} phải nằm trong [0, 1]")
        if min_margin is not None and not 0.0 <= min_margin <= 1.0:
            raise ValueError("min_margin phải nằm trong [0, 1]")
        if max_entropy is not None and max_entropy < 0.0:
            raise ValueError("max_entropy phải không âm")
        self.queue_threshold = float(queue_threshold)
        self.queue_margin = float(queue_margin)
        self.min_margin = float(min_margin) if min_margin is not None else None
        self.max_entropy = float(max_entropy) if max_entropy is not None else None
        self.sensitive_trigger = float(sensitive_trigger)
        self.minimum_sensitive_signal = float(minimum_sensitive_signal)
        self.minimum_scope_sensitive_signal = float(minimum_scope_sensitive_signal)
        self.taxonomy = taxonomy_resolver or TaxonomyResolver()

    def _review(self, reason: str, priority: str = "normal") -> RoutingDecision:
        return RoutingDecision(
            action="human_review",
            queue_id=self.taxonomy.get_scope_review_queue(),
            priority=priority,
            requires_human_review=True,
            reason_codes=[reason],
        )

    def evaluate(
        self,
        prediction: IntentPrediction,
        sensitive_case: SensitiveCaseAssessment,
        scope_detected: bool = False,
        scope_reasons: list[str] | None = None,
        queue_prediction: QueuePrediction | None = None,
    ) -> RoutingDecision:
        """Áp dụng policy theo đúng thứ tự đã thống nhất của dự án."""
        scope_reasons = scope_reasons or []
        sensitive_signal = (
            sensitive_case.sensitive_score >= self.minimum_scope_sensitive_signal
            if scope_detected
            else sensitive_case.sensitive_score >= self.minimum_sensitive_signal
        )
        if sensitive_case.requires_priority_review and sensitive_signal:
            return RoutingDecision(
                action="priority_human_review",
                queue_id=self.taxonomy.get_escalation_queue(sensitive_case.sensitive_category),
                priority="critical",
                requires_human_review=True,
                reason_codes=sensitive_case.reason_codes or ["SENSITIVE_INTENT_MASS"],
                intent=prediction.intent,
                domain=prediction.domain,
            )

        if scope_detected:
            return self._review(scope_reasons[0] if scope_reasons else "OUT_OF_SCOPE_QUERY")

        if queue_prediction is not None:
            if queue_prediction.confidence < self.queue_threshold:
                return self._review("LOW_QUEUE_CONFIDENCE")
            if queue_prediction.margin < self.queue_margin:
                return self._review("AMBIGUOUS_QUEUE_MARGIN")
        elif prediction.confidence < self.queue_threshold:
            return self._review("LOW_CONFIDENCE")

        if self.min_margin is not None and prediction.margin < self.min_margin:
            return self._review("AMBIGUOUS_MARGIN")
        if self.max_entropy is not None and prediction.entropy > self.max_entropy:
            return self._review("HIGH_ENTROPY")

        queue = queue_prediction.queue if queue_prediction is not None else self.taxonomy.get_queue(prediction.intent)
        return RoutingDecision(
            action="auto_route",
            queue_id=queue,
            priority=self.taxonomy.get_priority(prediction.intent),
            requires_human_review=False,
            intent=prediction.intent,
            domain=prediction.domain,
        )
