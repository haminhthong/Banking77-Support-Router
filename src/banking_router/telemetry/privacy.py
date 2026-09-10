"""Che PII trước khi ghi log hoặc lưu sự kiện."""

from __future__ import annotations

import re

from ..data.normalization import normalize_pii_semantically

_CARD_REGEX = re.compile(r"\b(?:\d[ -]*?){13,19}\b")
_EMAIL_REGEX = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")
_PHONE_REGEX = re.compile(r"(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b")
_ACCOUNT_REGEX = re.compile(r"\b(?:acc|account|ibans?)\s*[:#]?\s*([a-zA-Z0-9]{8,24})\b", re.IGNORECASE)


def redact_pii(text: str) -> str:
    """Che các định danh ngân hàng và thông tin cá nhân của khách hàng."""
    return normalize_pii_semantically(text)
