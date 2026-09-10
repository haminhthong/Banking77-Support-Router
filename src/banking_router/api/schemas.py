"""Pydantic schemas cho API route ticket."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class RouteRequest(BaseModel):
    text: str = Field(..., min_length=2, max_length=1000, description="Nội dung ticket hỗ trợ")
    top_k: int = Field(default=3, ge=1, le=5, description="Số intent thay thế được hiển thị")
    request_id: str | None = Field(default=None, description="Mã request do client cung cấp")


class BatchRouteRequest(BaseModel):
    tickets: list[RouteRequest] = Field(..., min_length=1, max_length=100)


class IntentAlternativeResponse(BaseModel):
    intent: str
    domain: str
    confidence: float


class PredictionResponse(BaseModel):
    intent: str
    domain: str
    confidence: float
    margin: float
    entropy: float
    alternatives: list[IntentAlternativeResponse] = Field(default_factory=list)


class QueuePredictionResponse(BaseModel):
    queue: str
    confidence: float
    margin: float
    probabilities: dict[str, float] = Field(default_factory=dict)


class SensitiveCaseResponse(BaseModel):
    requires_priority_review: bool
    sensitive_intent: str | None = None
    sensitive_score: float
    sensitive_probability_mass: float
    category: str | None = None
    group_mass: dict[str, float] = Field(default_factory=dict)
    signals: list[str] = Field(default_factory=list)


class ScopeResponse(BaseModel):
    supported: bool
    signals: list[str] = Field(default_factory=list)


class DecisionResponse(BaseModel):
    action: str
    queue: str
    priority: str
    requires_human_review: bool
    reason_codes: list[str] = Field(default_factory=list)


class RouteResponse(BaseModel):
    request_id: str
    prediction: PredictionResponse
    queue_prediction: QueuePredictionResponse | None = None
    scope: ScopeResponse
    sensitive_case: SensitiveCaseResponse
    decision: DecisionResponse
    metadata: dict[str, Any] = Field(default_factory=dict)


class FeedbackRequest(BaseModel):
    request_id: str
    reviewed_intent: str
    reviewed_queue: str | None = None
    resolution: str = "reviewed"
    reviewer_id: str = "human_agent"
    notes: str | None = None
    reason_code: str | None = None
