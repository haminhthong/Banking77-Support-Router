"""Text normalization and cryptographic hashing utilities for dataset contracts."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from pathlib import Path


def normalize_whitespace(text: str) -> str:
    """Normalize basic whitespace without mutating characters or punctuation."""
    return re.sub(r"\s+", " ", str(text)).strip()


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
