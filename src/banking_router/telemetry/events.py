"""Structured event recording and human feedback persistence."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from .privacy import redact_pii


def record_routing_event(
    event_file: Path | str,
    request_id: str,
    raw_text: str,
    result: Any,
    latency_ms: float = 0.0,
) -> None:
    """Record anonymized routing event for audit and telemetry."""
    path = Path(event_file)
    path.parent.mkdir(parents=True, exist_ok=True)

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "request_id": request_id,
        "text_redacted": redact_pii(raw_text),
        "predicted_intent": result.prediction.intent,
        "predicted_domain": result.prediction.domain,
        "confidence": result.prediction.confidence,
        "margin": result.prediction.margin,
        "entropy": result.prediction.entropy,
        "action": result.decision.action,
        "queue_id": result.decision.queue_id,
        "queue_confidence": result.queue_prediction.confidence if result.queue_prediction else None,
        "queue_margin": result.queue_prediction.margin if result.queue_prediction else None,
        "high_risk_detected": result.risk.high_risk_detected,
        "critical_probability": result.risk.critical_probability,
        "ood_detected": result.risk.ood_detected,
        "reason_codes": result.decision.reason_codes,
        "latency_ms": round(latency_ms, 2),
    }

    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def record_human_feedback(
    feedback_file: Path | str,
    request_id: str,
    model_version: str,
    predicted_intent: str,
    reviewed_intent: str,
    reviewed_queue: str | None = None,
    resolution: str = "reviewed",
    reviewer_id: str = "human_agent",
    notes: str | None = None,
) -> dict[str, Any]:
    """Record human reviewer corrections for continuous quality tracking and future retraining."""
    path = Path(feedback_file)
    path.parent.mkdir(parents=True, exist_ok=True)

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "request_id": request_id,
        "model_version": model_version,
        "predicted_intent": predicted_intent,
        "reviewed_intent": reviewed_intent,
        "reviewed_queue": reviewed_queue,
        "resolution": resolution,
        "is_correction": predicted_intent != reviewed_intent,
        "reviewer_id": reviewer_id,
        "notes": redact_pii(notes) if notes else None,
    }

    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return record
