"""Schema tách prediction, security risk và quyết định routing."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class IntentAlternative:
    """Alternative candidate intent presentation."""
    intent: str
    domain: str
    confidence: float


@dataclass(frozen=True)
class IntentPrediction:
    """Pure machine learning prediction representation.

    Decoupled from downstream operational routing and security decisions.
    """
    intent: str
    domain: str
    confidence: float
    margin: float
    entropy: float
    alternatives: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class QueuePrediction:
    """Business queue prediction projected from all intent probabilities."""
    queue: str
    confidence: float
    margin: float
    probabilities: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class RiskAssessment:
    """Đánh giá risk trên toàn bộ phân phối intent và tín hiệu scope."""
    high_risk_detected: bool
    high_risk_intent: str | None
    high_risk_score: float
    ood_detected: bool
    reason_codes: list[str] = field(default_factory=list)
    critical_probability: float = 0.0
    risk_category: str | None = None
    risk_group_mass: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class RoutingDecision:
    """Operational triage decision produced by the policy engine."""
    action: str  # "auto_route" | "human_review" | "priority_human_review"
    queue_id: str
    priority: str  # "normal" | "high" | "critical"
    requires_human_review: bool
    reason_codes: list[str] = field(default_factory=list)

    # Invariants and backward-compatibility aliases:
    intent: str | None = None
    domain: str | None = None

    @property
    def decision(self) -> str:
        """Alias for action ('auto_route', 'priority_escalation', 'abstain')."""
        if self.action == "priority_human_review":
            return "priority_escalation"
        if self.action == "human_review":
            return "abstain"
        return "auto_route"

    @property
    def abstained(self) -> bool:
        """True if the policy abstained from auto-routing."""
        return self.action == "human_review"

    @property
    def is_unknown(self) -> bool:
        """True if the policy abstained due to ambiguity or OOD."""
        return self.action == "human_review"

    @property
    def route(self) -> str:
        """Legacy route name or operational queue."""
        if self.action == "priority_human_review":
            return "priority_human_review"
        if self.action == "human_review":
            return "human"
        return self.intent or self.queue_id

    @property
    def review_reason(self) -> str | None:
        """Primary review reason code if human review is needed."""
        return self.reason_codes[0] if self.reason_codes else None


@dataclass(frozen=True)
class RoutingResult:
    """Complete canonical result returned by RoutingService."""
    request_id: str
    prediction: IntentPrediction
    risk: RiskAssessment
    decision: RoutingDecision
    queue_prediction: QueuePrediction | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
