"""FastAPI service cho inference và review feedback."""

from __future__ import annotations

import time
from typing import Any

from fastapi import FastAPI, HTTPException

from .schemas import (
    BatchRouteRequest,
    DecisionResponse,
    FeedbackRequest,
    IntentAlternativeResponse,
    PredictionResponse,
    QueuePredictionResponse,
    RouteRequest,
    RouteResponse,
    ScopeResponse,
    SensitiveCaseResponse,
)
from ..config import ARTIFACTS_DIR, REPORTS_DIR
from ..modeling.artifact import ModelArtifacts, load_artifacts
from ..routing.escalation import SensitiveIntentGuard
from ..routing.policy import RoutingPolicy
from ..routing.scope import ScopeGuard
from ..routing.service import RoutingService
from ..routing.taxonomy import TaxonomyResolver
from ..storage.repositories import TicketRepository
from ..telemetry.events import record_human_feedback, record_routing_event
from ..telemetry.privacy import redact_pii
from ..utils import LOGGER

app = FastAPI(
    title="Banking77 Support Ticket Router API",
    description="Intent classification and uncertainty-aware support-ticket routing on Banking77.",
    version="1.0.0",
)

_service: RoutingService | None = None
_artifacts: ModelArtifacts | None = None
_ticket_repository = TicketRepository(REPORTS_DIR / "tickets.sqlite3")


def get_routing_service() -> RoutingService:
    """Tải bộ artifact duy nhất và khởi tạo service dùng chung."""
    global _service, _artifacts
    if _service is not None:
        return _service
    try:
        _artifacts = load_artifacts(ARTIFACTS_DIR)
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Chưa sẵn sàng: {exc}. Hãy chạy 'python -m src.train' trước.",
        ) from exc

    policy_cfg = _artifacts.routing_policy
    runtime = policy_cfg.get("runtime_thresholds", {})
    sensitive_cfg = policy_cfg.get("sensitive_case", {})
    scope_cfg = policy_cfg.get("scope", {})
    taxonomy = TaxonomyResolver(_artifacts.taxonomy)
    policy = RoutingPolicy(
        queue_threshold=float(runtime.get("queue_probability", 0.45)),
        queue_margin=float(runtime.get("queue_margin", 0.0)),
        min_margin=float(runtime.get("min_margin", 0.02)),
        max_entropy=float(runtime.get("max_entropy", 3.80)),
        sensitive_trigger=float(runtime.get("sensitive_probability", 0.20)),
        minimum_sensitive_signal=float(sensitive_cfg.get("minimum_signal_probability", 0.16)),
        minimum_scope_sensitive_signal=float(sensitive_cfg.get("minimum_scope_signal", 0.30)),
        taxonomy_resolver=taxonomy,
    )
    sensitive_guard = SensitiveIntentGuard(
        classes=_artifacts.intent_model.classes_,
        taxonomy=taxonomy,
        sensitive_trigger=float(runtime.get("sensitive_probability", 0.20)),
    )
    scope_guard = ScopeGuard(
        min_chars=int(scope_cfg.get("min_chars", 4)),
        min_tokens=int(scope_cfg.get("min_tokens", 2)),
        lexical_similarity_threshold=float(scope_cfg.get("lexical_similarity_threshold", 0.08)),
        low_confidence_threshold=float(scope_cfg.get("low_confidence_threshold", 0.22)),
        unsupported_threshold=float(scope_cfg.get("unsupported_threshold", 0.80)),
    )
    _service = RoutingService(
        model=_artifacts.intent_model,
        policy=policy,
        sensitive_guard=sensitive_guard,
        scope_guard=scope_guard,
        scope_model=_artifacts.scope_model,
        taxonomy=taxonomy,
        metadata=dict(_artifacts.metadata),
    )
    return _service


@app.get("/health/live", summary="Liveness")
def liveness() -> dict[str, str]:
    return {"status": "alive"}


@app.get("/health/ready", summary="Readiness")
def readiness() -> dict[str, Any]:
    try:
        service = get_routing_service()
        return {
            "status": "ready",
            "model_ready": True,
            "class_count": len(service.model.classes_),
            "sensitive_intent_count": len(service.sensitive_guard.sensitive_intents),
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Service chưa sẵn sàng: {exc}") from exc


def _prediction_response(result: Any) -> PredictionResponse:
    return PredictionResponse(
        intent=result.prediction.intent,
        domain=result.prediction.domain,
        confidence=result.prediction.confidence,
        margin=result.prediction.margin,
        entropy=result.prediction.entropy,
        alternatives=[IntentAlternativeResponse(**item) for item in result.prediction.alternatives],
    )


def _route_response(result: Any) -> RouteResponse:
    queue = result.queue_prediction
    sensitive = result.sensitive_case
    return RouteResponse(
        request_id=result.request_id,
        prediction=_prediction_response(result),
        queue_prediction=(
            QueuePredictionResponse(
                queue=queue.queue,
                confidence=queue.confidence,
                margin=queue.margin,
                probabilities=queue.probabilities,
            )
            if queue else None
        ),
        scope=ScopeResponse(supported=not result.scope_detected, signals=result.scope_reasons),
        sensitive_case=SensitiveCaseResponse(
            requires_priority_review=sensitive.requires_priority_review,
            sensitive_intent=sensitive.sensitive_intent,
            sensitive_score=sensitive.sensitive_score,
            sensitive_probability_mass=sensitive.sensitive_probability_mass,
            category=sensitive.sensitive_category,
            group_mass=sensitive.sensitive_group_mass,
            signals=sensitive.reason_codes,
        ),
        decision=DecisionResponse(
            action=result.decision.action,
            queue=result.decision.queue_id,
            priority=result.decision.priority,
            requires_human_review=result.decision.requires_human_review,
            reason_codes=result.decision.reason_codes,
        ),
        metadata=result.metadata,
    )


@app.post("/v1/route", response_model=RouteResponse, summary="Route một ticket")
def route_ticket(req: RouteRequest) -> RouteResponse:
    started = time.perf_counter()
    service = get_routing_service()
    result = service.route(req.text, top_k=req.top_k, request_id=req.request_id)
    latency_ms = (time.perf_counter() - started) * 1000.0
    record_routing_event(REPORTS_DIR / "routing_events.jsonl", result.request_id, req.text, result, latency_ms)
    _ticket_repository.save_routing_result(req.text, result)
    LOGGER.info(
        "Route [id=%s]: text='%s' | action=%s | queue=%s | confidence=%.4f | sensitive=%s | scope=%s",
        result.request_id,
        redact_pii(req.text),
        result.decision.action,
        result.decision.queue_id,
        result.prediction.confidence,
        result.sensitive_case.requires_priority_review,
        result.scope_detected,
    )
    return _route_response(result)


@app.post("/v1/route/batch", summary="Route nhiều ticket")
def route_batch(batch: BatchRouteRequest) -> dict[str, Any]:
    service = get_routing_service()
    results = service.route_batch(
        [ticket.text for ticket in batch.tickets],
        top_ks=[ticket.top_k for ticket in batch.tickets],
    )
    counts: dict[str, int] = {}
    for result in results:
        counts[result.decision.action] = counts.get(result.decision.action, 0) + 1
    return {
        "total_tickets": len(results),
        "decision_counts": counts,
        "results": [_route_response(result).model_dump() for result in results],
    }


@app.post("/v1/feedback", summary="Lưu feedback review")
def submit_feedback(feedback: FeedbackRequest) -> dict[str, Any]:
    service = get_routing_service()
    ticket = _ticket_repository.get_ticket(feedback.request_id)
    if ticket is not None:
        review = _ticket_repository.add_review(
            request_id=feedback.request_id,
            reviewer_id=feedback.reviewer_id,
            final_intent=feedback.reviewed_intent,
            final_queue=feedback.reviewed_queue,
            resolution=feedback.resolution,
            notes=feedback.notes,
            reason_code=feedback.reason_code,
        )
        return {"status": "recorded", "feedback": review}
    record = record_human_feedback(
        REPORTS_DIR / "feedback_events.jsonl",
        request_id=feedback.request_id,
        model_name=str(service.metadata.get("model", "banking77-router")),
        predicted_intent="",
        reviewed_intent=feedback.reviewed_intent,
        reviewed_queue=feedback.reviewed_queue,
        resolution=feedback.resolution,
        reviewer_id=feedback.reviewer_id,
        notes=feedback.notes,
        reason_code=feedback.reason_code,
    )
    return {"status": "recorded", "feedback": record}


@app.post("/v1/tickets/{request_id}/review", summary="Review ticket")
def review_ticket(request_id: str, feedback: FeedbackRequest) -> dict[str, Any]:
    if feedback.request_id != request_id:
        raise HTTPException(status_code=400, detail="request_id trong path và payload phải giống nhau")
    try:
        review = _ticket_repository.add_review(
            request_id=request_id,
            reviewer_id=feedback.reviewer_id,
            final_intent=feedback.reviewed_intent,
            final_queue=feedback.reviewed_queue,
            resolution=feedback.resolution,
            notes=feedback.notes,
            reason_code=feedback.reason_code,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Không tìm thấy ticket: {request_id}") from exc
    return {"status": "recorded", "review": review}
