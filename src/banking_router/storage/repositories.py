"""Repository operations for the ticket and reviewed-feedback lifecycle."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .database import connect_database
from ..telemetry.privacy import redact_pii


class TicketRepository:
    def __init__(self, database_path: Path | str) -> None:
        self.database_path = Path(database_path)

    def save_routing_result(self, raw_text: str, result: Any) -> None:
        queue_prediction = result.queue_prediction
        now = datetime.now(timezone.utc).isoformat()
        connection = connect_database(self.database_path)
        try:
            connection.execute(
                """
                INSERT INTO tickets (
                    request_id, created_at, text_redacted, model_name,
                    predicted_intent, intent_confidence, predicted_queue,
                    queue_confidence, sensitive_probability_mass,
                    decision, queue_id, priority, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(request_id) DO UPDATE SET
                    text_redacted=excluded.text_redacted,
                    decision=excluded.decision,
                    queue_id=excluded.queue_id,
                    priority=excluded.priority
                """,
                (
                    result.request_id,
                    now,
                    redact_pii(raw_text),
                    result.metadata.get("model", "banking77-router"),
                    result.prediction.intent,
                    result.prediction.confidence,
                    queue_prediction.queue if queue_prediction else result.decision.queue_id,
                    queue_prediction.confidence if queue_prediction else result.prediction.confidence,
                    result.sensitive_case.sensitive_probability_mass,
                    result.decision.action,
                    result.decision.queue_id,
                    result.decision.priority,
                    "PENDING_PRIORITY_REVIEW" if result.decision.action == "priority_human_review" else (
                        "PENDING_REVIEW" if result.decision.requires_human_review else "AUTO_ROUTED"
                    ),
                ),
            )
            connection.commit()
        finally:
            connection.close()

    def get_ticket(self, request_id: str) -> dict[str, Any] | None:
        connection = connect_database(self.database_path)
        try:
            row = connection.execute("SELECT * FROM tickets WHERE request_id = ?", (request_id,)).fetchone()
            return dict(row) if row else None
        finally:
            connection.close()

    def add_review(
        self,
        request_id: str,
        reviewer_id: str,
        final_intent: str,
        final_queue: str | None,
        resolution: str,
        notes: str | None,
        reason_code: str | None = None,
    ) -> dict[str, Any]:
        connection = connect_database(self.database_path)
        try:
            ticket = connection.execute("SELECT * FROM tickets WHERE request_id = ?", (request_id,)).fetchone()
            if ticket is None:
                raise KeyError(request_id)
            now = datetime.now(timezone.utc).isoformat()
            cursor = connection.execute(
                """
                INSERT INTO reviews (request_id, reviewer_id, final_intent, final_queue, resolution, created_at, notes_redacted, reason_code)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (request_id, reviewer_id, final_intent, final_queue, resolution, now, redact_pii(notes) if notes else None, reason_code),
            )
            connection.execute(
                "UPDATE tickets SET status = 'REVIEWED' WHERE request_id = ?",
                (request_id,),
            )
            connection.commit()
            return {
                "review_id": cursor.lastrowid,
                "request_id": request_id,
                "predicted_intent": ticket["predicted_intent"],
                "predicted_queue": ticket["predicted_queue"],
                "reviewed_intent": final_intent,
                "reviewed_queue": final_queue,
                "intent_corrected": ticket["predicted_intent"] != final_intent,
                "queue_corrected": final_queue is not None and ticket["predicted_queue"] != final_queue,
                "resolution": resolution,
                "reviewer_id": reviewer_id,
                "reason_code": reason_code,
            }
        finally:
            connection.close()
