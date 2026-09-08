"""SQLite schema and connection helpers for ticket lifecycle state."""

from __future__ import annotations

import sqlite3
from pathlib import Path


def connect_database(path: Path | str) -> sqlite3.Connection:
    database_path = Path(path)
    database_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(database_path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS tickets (
            request_id TEXT PRIMARY KEY,
            created_at TEXT NOT NULL,
            text_redacted TEXT NOT NULL,
            model_version TEXT NOT NULL,
            policy_version TEXT NOT NULL,
            predicted_intent TEXT NOT NULL,
            intent_confidence REAL NOT NULL,
            predicted_queue TEXT NOT NULL,
            queue_confidence REAL NOT NULL,
            critical_risk_probability REAL NOT NULL,
            decision TEXT NOT NULL,
            queue_id TEXT NOT NULL,
            priority TEXT NOT NULL,
            status TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS reviews (
            review_id INTEGER PRIMARY KEY AUTOINCREMENT,
            request_id TEXT NOT NULL REFERENCES tickets(request_id),
            reviewer_id TEXT NOT NULL,
            final_intent TEXT NOT NULL,
            final_queue TEXT,
            resolution TEXT NOT NULL,
            created_at TEXT NOT NULL,
            notes_redacted TEXT,
            reason_code TEXT
        );
        """
    )
    columns = {row[1] for row in connection.execute("PRAGMA table_info(reviews)").fetchall()}
    if "reason_code" not in columns:
        connection.execute("ALTER TABLE reviews ADD COLUMN reason_code TEXT")
    connection.commit()
    return connection
