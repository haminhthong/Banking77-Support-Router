"""FastAPI HTTP REST Service for Risk-Aware Banking Support Triage."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .schemas import (
    BatchRouteRequest,
    DecisionResponse,
    FeedbackRequest,
    IntentAlternativeResponse,
    PredictionResponse,
    QueuePredictionResponse,
    RiskResponse,
    RouteRequest,
    RouteResponse,
    ScopeResponse,
)
from ..config import MODELS_DIR, REPORTS_DIR, resolve_models_dir
from ..modeling.artifact import ModelBundle, load_and_validate_bundle
from ..routing.ood import OODGuard
from ..routing.policy import RoutingPolicy
from ..routing.risk import RiskAssessor
from ..routing.service import RoutingService
from ..routing.taxonomy import TaxonomyResolver
from ..telemetry.events import record_human_feedback, record_routing_event
from ..telemetry.privacy import redact_pii
from ..storage.repositories import TicketRepository
from ..utils import LOGGER

app = FastAPI(
    title="Risk-Aware Banking Support Triage System API",
    description="Multi-tiered Customer Support Triage System with Decoupled Predictions, Security Risk Scanner, OOD Detection, and Operational Queue Routing.",
    version="3.0.0",
)

_service: RoutingService | None = None
_bundle: ModelBundle | None = None
_ticket_repository = TicketRepository(REPORTS_DIR / "tickets.sqlite3")


def get_routing_service() -> RoutingService:
    """Lazy-load and initialize the singleton RoutingService with bundle contract verification."""
    global _service, _bundle
    if _service is None:
        try:
            _bundle = load_and_validate_bundle(resolve_models_dir(MODELS_DIR), verify_checksum=True)
        except Exception as exc:
            raise HTTPException(
                status_code=503,
                detail=f"Routing service initialization failed: {exc}. Please run 'python -m src.train' first.",
            ) from exc

        cfg = _bundle.config
        policy_cfg = _bundle.policy_config
        taxonomy = TaxonomyResolver(_bundle.taxonomy)
        auto_cfg = policy_cfg.get("auto_route", {})
        critical_cfg = policy_cfg.get("critical_risk", {})
        scope_cfg = policy_cfg.get("scope", {})
        policy = RoutingPolicy(
            threshold=float(auto_cfg.get("min_queue_probability", policy_cfg.get("threshold", cfg.get("threshold", 0.45)))),
            queue_threshold=float(auto_cfg.get("min_queue_probability", policy_cfg.get("threshold", cfg.get("threshold", 0.45)))),
            queue_margin=float(auto_cfg.get("min_queue_margin", policy_cfg.get("min_margin", cfg.get("min_margin", 0.02)))),
            min_margin=policy_cfg.get("min_margin", cfg.get("min_margin")),
            max_entropy=policy_cfg.get("max_entropy", cfg.get("max_entropy")),
            high_risk_trigger=float(critical_cfg.get("minimum_probability", policy_cfg.get("high_risk_trigger", cfg.get("high_risk_trigger", 0.20)))),
            minimum_risk_signal=float(critical_cfg.get("minimum_signal_probability", 0.16)),
            minimum_ood_security_signal=float(critical_cfg.get("minimum_ood_security_signal", 0.30)),
            taxonomy_resolver=taxonomy,
        )
        risk_assessor = RiskAssessor(
            classes=_bundle.model.classes_,
            taxonomy=taxonomy,
            high_risk_trigger=float(critical_cfg.get("minimum_probability", policy_cfg.get("high_risk_trigger", cfg.get("high_risk_trigger", 0.20)))),
        )
        ood_guard = OODGuard(
            min_chars=int(scope_cfg.get("min_chars", 4)),
            min_tokens=int(scope_cfg.get("min_tokens", 2)),
            lexical_similarity_threshold=float(scope_cfg.get("lexical_novelty_threshold", 0.08)),
            low_confidence_ood_threshold=float(scope_cfg.get("low_confidence_threshold", 0.22)),
            unsupported_threshold=float(scope_cfg.get("unsupported_threshold", 0.80)),
        )
        _service = RoutingService(
            model=_bundle.model,
            policy=policy,
            risk_assessor=risk_assessor,
            ood_guard=ood_guard,
            scope_model=_bundle.scope_model,
            taxonomy=taxonomy,
            metadata={
                "model_version": _bundle.manifest.get("model_version", cfg.get("version", "unknown")),
                "policy_version": _bundle.manifest.get("policy_version", policy_cfg.get("policy_version", "unknown")),
                "normalization_version": _bundle.manifest.get("normalization_version", "legacy"),
                "scope_model_version": _bundle.manifest.get("scope_model_version") or "heuristic-only",
            },
        )
    return _service


@app.get("/health/live", summary="Liveness Probe")
def liveness() -> dict[str, str]:
    """Check if the service process is alive."""
    return {"status": "alive"}


@app.get("/health/ready", summary="Readiness Probe")
def readiness() -> dict[str, Any]:
    """Validate model bundle readiness, class count, and taxonomy integrity."""
    try:
        service = get_routing_service()
        release_dir = resolve_models_dir(MODELS_DIR)
        bundle = _bundle or load_and_validate_bundle(release_dir, verify_checksum=True)
        gate_path = release_dir / "release_gate.json"
        gate = json.loads(gate_path.read_text(encoding="utf-8")) if gate_path.exists() else {"status": "UNKNOWN"}
        return {
            "status": "ready",
            "model_ready": True,
            "model_version": bundle.config.get("version", "unknown"),
            "policy_version": bundle.config.get("policy_version", "unknown"),
            "class_count": len(bundle.model.classes_),
            "high_risk_classes_count": len(TaxonomyResolver(bundle.taxonomy).get_critical_intents()),
            "release_gate": gate,
        }
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Service not ready: {exc}",
        )


@app.get("/health", summary="Legacy Health Check")
def legacy_health() -> dict[str, Any]:
    """Backward-compatible health check endpoint."""
    try:
        ready_info = readiness()
        return {
            "status": "ok",
            "model_ready": True,
            "model_version": ready_info["model_version"],
            "policy_version": ready_info["policy_version"],
            "class_count": ready_info["class_count"],
        }
    except HTTPException:
        return {
            "status": "degraded",
            "model_ready": False,
            "model_version": "not_trained",
            "policy_version": "not_trained",
            "class_count": 0,
        }


@app.post("/v1/route", response_model=RouteResponse, summary="Route customer ticket")
def route_ticket(req: RouteRequest) -> RouteResponse:
    """Canonical triage endpoint: produces decoupled prediction, risk assessment, and queue decision."""
    start_time = time.perf_counter()
    service = get_routing_service()
    res = service.route(text=req.text, top_k=req.top_k, request_id=req.request_id)
    latency_ms = (time.perf_counter() - start_time) * 1000.0

    # Record telemetry event (PII redacted)
    record_routing_event(
        event_file=REPORTS_DIR / "routing_events.jsonl",
        request_id=res.request_id,
        raw_text=req.text,
        result=res,
        latency_ms=latency_ms,
    )
    _ticket_repository.save_routing_result(req.text, res)

    LOGGER.info(
        "Triage [id=%s]: text='%s' | action=%s | queue=%s | conf=%.4f | high_risk=%s | ood=%s",
        res.request_id,
        redact_pii(req.text),
        res.decision.action,
        res.decision.queue_id,
        res.prediction.confidence,
        res.risk.high_risk_detected,
        res.risk.ood_detected,
    )

    return RouteResponse(
        request_id=res.request_id,
        prediction=PredictionResponse(
            intent=res.prediction.intent,
            domain=res.prediction.domain,
            confidence=res.prediction.confidence,
            margin=res.prediction.margin,
            entropy=res.prediction.entropy,
            alternatives=[
                IntentAlternativeResponse(**alt) for alt in res.prediction.alternatives
            ],
        ),
        risk=RiskResponse(
            tier="critical" if res.risk.high_risk_detected else "normal",
            high_risk_detected=res.risk.high_risk_detected,
            high_risk_intent=res.risk.high_risk_intent,
            high_risk_score=res.risk.high_risk_score,
            critical_probability=res.risk.critical_probability,
            ood_detected=res.risk.ood_detected,
            risk_category=res.risk.risk_category,
            risk_group_mass=res.risk.risk_group_mass,
        ),
        scope=ScopeResponse(
            supported=not res.risk.ood_detected,
            signals=list(res.risk.reason_codes) if res.risk.ood_detected else [],
        ),
        versions={
            "model": str(res.metadata.get("model_version", "unknown")),
            "policy": str(res.metadata.get("policy_version", "unknown")),
            "scope_model": str(res.metadata.get("scope_model_version", "heuristic-only")),
        },
        decision=DecisionResponse(
            action=res.decision.action,
            queue=res.decision.queue_id,
            priority=res.decision.priority,
            requires_human_review=res.decision.requires_human_review,
            reason_codes=res.decision.reason_codes,
        ),
        metadata=res.metadata,
        intent_prediction=PredictionResponse(
            intent=res.prediction.intent,
            domain=res.prediction.domain,
            confidence=res.prediction.confidence,
            margin=res.prediction.margin,
            entropy=res.prediction.entropy,
            alternatives=[IntentAlternativeResponse(**alt) for alt in res.prediction.alternatives],
        ),
        queue_prediction=(
            QueuePredictionResponse(
                queue=res.queue_prediction.queue,
                confidence=res.queue_prediction.confidence,
                margin=res.queue_prediction.margin,
                probabilities=res.queue_prediction.probabilities,
            )
            if res.queue_prediction is not None
            else None
        ),
        # Legacy compatibility
        decision_legacy=res.decision.decision,
        intent=res.decision.intent,
        domain=res.decision.domain,
        route=res.decision.route,
    )


@app.post("/v1/route/batch", summary="Vectorized Batch Routing")
def route_batch(batch: BatchRouteRequest) -> dict[str, Any]:
    """Process high-volume ticket batches using vectorized matrix inference."""
    service = get_routing_service()
    texts = [t.text for t in batch.tickets]
    results = service.route_batch(
        texts,
        top_ks=[ticket.top_k for ticket in batch.tickets],
    )

    decision_counts: dict[str, int] = {}
    response_items = []
    for r in results:
        action = r.decision.action
        decision_counts[action] = decision_counts.get(action, 0) + 1
        response_items.append({
            "request_id": r.request_id,
            "prediction": {
                "intent": r.prediction.intent,
                "domain": r.prediction.domain,
                "confidence": r.prediction.confidence,
                "alternatives": r.prediction.alternatives,
            },
            "risk": {
                "high_risk_detected": r.risk.high_risk_detected,
                "ood_detected": r.risk.ood_detected,
            },
            "decision": {
                "action": r.decision.action,
                "queue": r.decision.queue_id,
                "priority": r.decision.priority,
                "requires_human_review": r.decision.requires_human_review,
            },
        })

    return {
        "total_tickets": len(results),
        "decision_counts": decision_counts,
        "results": response_items,
    }


@app.post("/v1/feedback", summary="Submit Human Reviewer Feedback")
def submit_feedback(fb: FeedbackRequest) -> dict[str, Any]:
    """Submit reviewer correction for quality tracking and continuous improvement loop."""
    service = get_routing_service()
    existing_ticket = _ticket_repository.get_ticket(fb.request_id)
    if existing_ticket is not None:
        review = _ticket_repository.add_review(
            request_id=fb.request_id,
            reviewer_id=fb.reviewer_id,
            final_intent=fb.reviewed_intent,
            final_queue=fb.reviewed_queue,
            resolution=fb.resolution,
            notes=fb.notes,
            reason_code=fb.reason_code,
        )
        return {"status": "recorded", "feedback": review}

    # Legacy compatibility for clients that sent feedback before ticket
    # persistence existed.  New integrations should use the review endpoint.
    record = record_human_feedback(
        feedback_file=REPORTS_DIR / "feedback_events.jsonl",
        request_id=fb.request_id,
        model_version=service.metadata.get("model_version", "v3"),
        predicted_intent="",  # Filled or tracked via request_id lookup
        reviewed_intent=fb.reviewed_intent,
        reviewed_queue=fb.reviewed_queue,
        resolution=fb.resolution,
        reviewer_id=fb.reviewer_id,
        notes=fb.notes,
        reason_code=fb.reason_code,
    )
    return {"status": "recorded", "feedback": record}


@app.post("/v1/tickets/{request_id}/review", summary="Review a routed ticket")
def review_ticket(request_id: str, fb: FeedbackRequest) -> dict[str, Any]:
    """Persist a human resolution against the original prediction by request_id."""
    if fb.request_id != request_id:
        raise HTTPException(status_code=400, detail="request_id in path and payload must match")
    try:
        review = _ticket_repository.add_review(
            request_id=request_id,
            reviewer_id=fb.reviewer_id,
            final_intent=fb.reviewed_intent,
            final_queue=fb.reviewed_queue,
            resolution=fb.resolution,
            notes=fb.notes,
            reason_code=fb.reason_code,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown ticket: {request_id}") from exc
    return {"status": "recorded", "review": review}


# ─── Legacy API Compatibility Endpoints (/predict and /predict/batch) ─────────

@app.post("/predict", summary="Legacy Predict Endpoint")
def legacy_predict(req: RouteRequest) -> dict[str, Any]:
    """Legacy prediction endpoint preserved for backward compatibility."""
    resp = route_ticket(req)
    # Match the legacy response dictionary structure
    return {
        "decision": resp.decision_legacy or ("auto_route" if resp.decision.action == "auto_route" else ("priority_escalation" if resp.decision.action == "priority_human_review" else "abstain")),
        "abstained": resp.decision.action == "human_review",
        "is_unknown": resp.decision.action == "human_review",
        "intent": resp.prediction.intent if resp.decision.action == "auto_route" else (resp.risk.high_risk_intent if resp.risk.high_risk_detected else None),
        "domain": resp.prediction.domain if resp.decision.action == "auto_route" else None,
        "top_intent": resp.prediction.intent,
        "top_domain": resp.prediction.domain,
        "confidence": resp.prediction.confidence,
        "margin": resp.prediction.margin,
        "entropy": resp.prediction.entropy,
        "alternatives": [alt.model_dump() for alt in resp.prediction.alternatives],
        "route": resp.decision.queue if resp.decision.action == "auto_route" else resp.decision.action,
        "requires_human_review": resp.decision.requires_human_review,
        "review_reason": resp.decision.reason_codes[0] if resp.decision.reason_codes else None,
        "model_version": resp.metadata.get("model_version", "v3"),
        "policy_version": resp.metadata.get("policy_version", "v3"),
    }


class LegacyBatchQuery(BaseModel):
    queries: list[RouteRequest]


@app.post("/predict/batch", summary="Legacy Batch Predict Endpoint")
def legacy_predict_batch(batch: LegacyBatchQuery) -> dict[str, Any]:
    """Legacy batch predict endpoint preserved for backward compatibility."""
    results = [legacy_predict(q) for q in batch.queries]
    decision_counts: dict[str, int] = {}
    for r in results:
        dec = str(r["decision"])
        decision_counts[dec] = decision_counts.get(dec, 0) + 1

    return {
        "total_queries": len(results),
        "decision_counts": decision_counts,
        "results": results,
    }
