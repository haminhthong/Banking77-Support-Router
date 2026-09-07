"""FastAPI HTTP REST Service for Risk-Aware Banking Support Triage."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from .schemas import (
    BatchRouteRequest,
    DecisionResponse,
    FeedbackRequest,
    IntentAlternativeResponse,
    PredictionResponse,
    RiskResponse,
    RouteRequest,
    RouteResponse,
)
from ..config import MODELS_DIR, REPORTS_DIR, get_taxonomy_config
from ..data.contracts import DEFAULT_HIGH_RISK_INTENTS
from ..modeling.artifact import ModelBundle, load_and_validate_bundle
from ..routing.ood import OODGuard
from ..routing.policy import RoutingPolicy
from ..routing.risk import RiskAssessor
from ..routing.service import RoutingService
from ..routing.taxonomy import TaxonomyResolver
from ..telemetry.events import record_human_feedback, record_routing_event
from ..telemetry.privacy import redact_pii
from ..utils import LOGGER

app = FastAPI(
    title="Risk-Aware Banking Support Triage System API",
    description="Multi-tiered Customer Support Triage System with Decoupled Predictions, Security Risk Scanner, OOD Detection, and Operational Queue Routing.",
    version="3.0.0",
)

_service: RoutingService | None = None
_bundle: ModelBundle | None = None


def get_routing_service() -> RoutingService:
    """Lazy-load and initialize the singleton RoutingService with bundle contract verification."""
    global _service, _bundle
    if _service is None:
        try:
            _bundle = load_and_validate_bundle(MODELS_DIR, verify_checksum=False)
        except Exception as exc:
            raise HTTPException(
                status_code=503,
                detail=f"Routing service initialization failed: {exc}. Please run 'python -m src.train' first.",
            ) from exc

        cfg = _bundle.config
        taxonomy = TaxonomyResolver(_bundle.taxonomy)
        policy = RoutingPolicy(
            threshold=float(cfg.get("threshold", 0.45)),
            high_risk_trigger=float(cfg.get("high_risk_trigger", 0.20)),
            min_margin=cfg.get("min_margin"),
            max_entropy=cfg.get("max_entropy"),
            taxonomy_resolver=taxonomy,
        )
        risk_assessor = RiskAssessor(
            classes=_bundle.model.classes_,
            high_risk_intents=DEFAULT_HIGH_RISK_INTENTS,
            high_risk_trigger=float(cfg.get("high_risk_trigger", 0.20)),
        )
        ood_guard = OODGuard()
        _service = RoutingService(
            model=_bundle.model,
            policy=policy,
            risk_assessor=risk_assessor,
            ood_guard=ood_guard,
            taxonomy=taxonomy,
            metadata={
                "model_version": cfg.get("version", "v3"),
                "policy_version": cfg.get("policy_version", "v3"),
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
        bundle = _bundle or load_and_validate_bundle(MODELS_DIR, verify_checksum=False)
        return {
            "status": "ready",
            "model_ready": True,
            "model_version": bundle.config.get("version", "unknown"),
            "policy_version": bundle.config.get("policy_version", "unknown"),
            "class_count": len(bundle.model.classes_),
            "high_risk_classes_count": len(bundle.manifest.get("high_risk_intents", [])),
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
            high_risk_detected=res.risk.high_risk_detected,
            high_risk_intent=res.risk.high_risk_intent,
            high_risk_score=res.risk.high_risk_score,
            ood_detected=res.risk.ood_detected,
        ),
        decision=DecisionResponse(
            action=res.decision.action,
            queue=res.decision.queue_id,
            priority=res.decision.priority,
            requires_human_review=res.decision.requires_human_review,
            reason_codes=res.decision.reason_codes,
        ),
        metadata=res.metadata,
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
    results = service.route_batch(texts, top_k=batch.tickets[0].top_k if batch.tickets else 3)

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
    record = record_human_feedback(
        feedback_file=REPORTS_DIR / "feedback_events.jsonl",
        request_id=fb.request_id,
        model_version=service.metadata.get("model_version", "v3"),
        predicted_intent="",  # Filled or tracked via request_id lookup
        reviewed_intent=fb.reviewed_intent,
        reviewer_id=fb.reviewer_id,
        notes=fb.notes,
    )
    return {"status": "recorded", "feedback": record}


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
