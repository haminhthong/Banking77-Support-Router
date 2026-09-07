"""Pydantic schemas for REST API endpoints."""

from __future__ import annotations

from typing import Any
from pydantic import BaseModel, Field


class RouteRequest(BaseModel):
    """Customer support ticket input request."""
    text: str = Field(
        ...,
        min_length=2,
        max_length=1000,
        description="Nội dung thắc mắc hoặc yêu cầu hỗ trợ từ khách hàng",
        examples=["Why has my cash withdrawal been declined?"],
    )
    top_k: int = Field(
        default=3,
        ge=1,
        le=5,
        description="Số lượng dự đoán thay thế top_k hiển thị (không ảnh hưởng tới safety engine)",
    )
    request_id: str | None = Field(
        default=None,
        description="Optional client request tracking identifier",
    )


class BatchRouteRequest(BaseModel):
    """Batch routing request schema."""
    tickets: list[RouteRequest] = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Danh sách các câu hỏi ticket cần xử lý phân luồng",
    )


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


class RiskResponse(BaseModel):
    high_risk_detected: bool
    high_risk_intent: str | None
    high_risk_score: float
    ood_detected: bool


class DecisionResponse(BaseModel):
    action: str
    queue: str
    priority: str
    requires_human_review: bool
    reason_codes: list[str] = Field(default_factory=list)


class RouteResponse(BaseModel):
    """Canonical production response format for triage decisions."""
    request_id: str
    prediction: PredictionResponse
    risk: RiskResponse
    decision: DecisionResponse
    metadata: dict[str, Any] = Field(default_factory=dict)

    # Legacy flat fields for backward compatibility with existing tests
    decision_legacy: str | None = None
    intent: str | None = None
    domain: str | None = None
    route: str | None = None


class FeedbackRequest(BaseModel):
    """Human agent correction or feedback submission schema."""
    request_id: str = Field(..., description="Mã request_id của ticket cần hiệu chỉnh")
    reviewed_intent: str = Field(..., description="Nhãn ý định chính xác được nhân viên xác nhận")
    reviewer_id: str = Field(default="human_agent", description="ID định danh chuyên viên xử lý")
    notes: str | None = Field(default=None, description="Ghi chú nghiệp vụ (được lọc PII tự động)")
