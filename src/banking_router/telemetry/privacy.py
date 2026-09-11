"""Che PII trước khi ghi log hoặc lưu sự kiện."""

from __future__ import annotations

from ..data.normalization import normalize_pii_semantically


def redact_pii(text: str) -> str:
    """Che các định danh ngân hàng và thông tin cá nhân của khách hàng."""
    return normalize_pii_semantically(text)
