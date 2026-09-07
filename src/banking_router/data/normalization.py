"""Text normalization and cryptographic hashing utilities for dataset contracts."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from pathlib import Path


NORMALIZATION_VERSION = "semantic-pii-v1"

_CARD_REGEX = re.compile(r"\b(?:\d[ -]*?){13,19}\b")
_EMAIL_REGEX = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")
_PHONE_REGEX = re.compile(r"(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b")
_ACCOUNT_REGEX = re.compile(
    r"\b(?P<label>acc|account|iban|ibans?)\s*[:#]?\s*(?P<value>[a-zA-Z0-9]{8,24})\b",
    re.IGNORECASE,
)


def normalize_whitespace(text: str) -> str:
    """Normalize basic whitespace without mutating characters or punctuation."""
    return re.sub(r"\s+", " ", str(text)).strip()


def normalize_pii_semantically(text: str) -> str:
    """Normalize PII to semantic placeholders before model inference.

    The noun introducing an identifier is intentionally retained (``account``
    rather than only ``[REDACTED]``), so the transformation removes leakage
    without removing banking meaning.  This function is shared by training and
    serving to prevent normalization skew.
    """
    normalized = unicodedata.normalize("NFKC", str(text))
    normalized = _CARD_REGEX.sub("[CARD_NUMBER]", normalized)
    normalized = _EMAIL_REGEX.sub("[EMAIL]", normalized)
    normalized = _PHONE_REGEX.sub("[PHONE_NUMBER]", normalized)
    normalized = _ACCOUNT_REGEX.sub(
        lambda match: f"{match.group('label').lower()} [ACCOUNT_ID]", normalized
    )
    return normalize_whitespace(normalized)


def normalize_text_for_audit(text: str) -> str:
    """NFKC-normalize, lowercase, remove punctuation and collapse whitespace.

    Used specifically for semantic fingerprinting in cross-split leakage
    and conflicting-label audits.
    """
    text = unicodedata.normalize("NFKC", str(text).lower().strip())
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def compute_file_sha256(path: Path | str) -> str:
    """Compute SHA-256 hash of a file for reproducibility and manifest verification."""
    p = Path(path)
    if not p.exists():
        return "missing"
    h = hashlib.sha256()
    h.update(p.read_bytes())
    return h.hexdigest()
