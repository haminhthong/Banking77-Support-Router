"""Kiểu dữ liệu cho prediction, scope, sensitive-case và routing."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class IntentPrediction:
    """Prediction thuần ML, tách khỏi quyết định routing."""

    intent: str
    domain: str
    confidence: float
    margin: float
    entropy: float
    alternatives: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class QueuePrediction:
    """Xác suất queue được cộng từ toàn bộ 77 intent."""

    queue: str
    confidence: float
    margin: float
    probabilities: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class SensitiveCaseAssessment:
    """Đánh giá tín hiệu intent nhạy cảm trên toàn bộ phân phối xác suất."""

    requires_priority_review: bool
    sensitive_intent: str | None
    sensitive_score: float
    scope_detected: bool
    reason_codes: list[str] = field(default_factory=list)
    sensitive_probability_mass: float = 0.0
    sensitive_category: str | None = None
    sensitive_group_mass: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class RoutingDecision:
    """Quyết định cuối: auto_route hoặc human_review."""

    action: str  # Một trong: tự route, review thường hoặc review ưu tiên.
    queue_id: str
    priority: str  # Một trong: bình thường, cao hoặc khẩn cấp.
    requires_human_review: bool
    reason_codes: list[str] = field(default_factory=list)
    intent: str | None = None
    domain: str | None = None


@dataclass(frozen=True)
class RoutingResult:
    """Kết quả đầy đủ của một lần route ticket."""

    request_id: str
    prediction: IntentPrediction
    queue_prediction: QueuePrediction | None
    sensitive_case: SensitiveCaseAssessment
    scope_detected: bool
    scope_reasons: list[str]
    decision: RoutingDecision
    metadata: dict[str, Any] = field(default_factory=dict)
