"""Chuẩn hóa text và PII trước khi train hoặc inference."""

from __future__ import annotations

import re
import unicodedata


NORMALIZATION_VERSION = "semantic-pii-v1"

_CARD_REGEX = re.compile(r"\b(?:\d[ -]*?){13,19}\b")
_EMAIL_REGEX = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")
_PHONE_REGEX = re.compile(r"(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b")
_ACCOUNT_REGEX = re.compile(
    r"\b(?P<label>acc|account|iban|ibans?)\s*[:#]?\s*(?P<value>[a-zA-Z0-9]{8,24})\b",
    re.IGNORECASE,
)


def normalize_whitespace(text: str) -> str:
    """Chuẩn hóa khoảng trắng mà không đổi ký tự hoặc dấu câu."""
    return re.sub(r"\s+", " ", str(text)).strip()


def normalize_pii_semantically(text: str) -> str:
    """Đổi PII thành placeholder ngữ nghĩa trước khi model suy luận.

    Giữ lại danh từ đứng trước định danh (``account`` thay vì chỉ dùng
    ``[REDACTED]``) để loại rò rỉ nhưng không làm mất nghĩa ngân hàng. Hàm này
    được dùng chung cho train và serving để tránh lệch chuẩn hóa.
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
    """Chuẩn hóa NFKC, viết thường, bỏ dấu câu và gom khoảng trắng.

    Hàm dùng để tạo fingerprint ngữ nghĩa khi kiểm tra rò rỉ giữa split và
    xung đột nhãn.
    """
    text = unicodedata.normalize("NFKC", str(text).lower().strip())
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()
