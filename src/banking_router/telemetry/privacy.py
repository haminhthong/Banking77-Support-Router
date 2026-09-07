"""Privacy protection and PII redaction layer for telemetry and logs."""

from __future__ import annotations

import re

_CARD_REGEX = re.compile(r"\b(?:\d[ -]*?){13,19}\b")
_EMAIL_REGEX = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")
_PHONE_REGEX = re.compile(r"(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b")
_ACCOUNT_REGEX = re.compile(r"\b(?:acc|account|ibans?)\s*[:#]?\s*([a-zA-Z0-9]{8,24})\b", re.IGNORECASE)


def redact_pii(text: str) -> str:
    """Mask sensitive banking and personal customer identifiers."""
    redacted = _CARD_REGEX.sub("[CARD_NUMBER]", text)
    redacted = _EMAIL_REGEX.sub("[EMAIL]", redacted)
    redacted = _PHONE_REGEX.sub("[PHONE_NUMBER]", redacted)
    redacted = _ACCOUNT_REGEX.sub("account [ACCOUNT_ID]", redacted)
    return redacted
