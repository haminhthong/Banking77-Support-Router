"""Các hàm tiện ích hệ thống cho dự án AI Customer Support Router.

Cung cấp các công cụ cấu hình logging, đặt seed ngẫu nhiên, lưu file JSON,
đo lường Expected Calibration Error (ECE), tính toán Entropy độ mập mờ,
và tầng làm sạch thông tin nhạy cảm PII (PII Redaction) cho Telemetry Logging.
"""

from __future__ import annotations

import json
import logging
import os
import random
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

# Khởi tạo logger chung cho dự án
LOGGER = logging.getLogger("banking77_router")


def setup_logging() -> None:
    """Cấu hình định dạng và cấp độ hiển thị log hệ thống với UTF-8 encoding."""
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except (OSError, ValueError) as exc:
            LOGGER.debug("Không thể đổi encoding console sang UTF-8: %s", exc)

    level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def set_seed(seed: int = 42) -> None:
    """Đảm bảo tính tái lập (reproducibility) bằng cách cố định seed ngẫu nhiên."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        LOGGER.debug("Không tìm thấy PyTorch; chỉ đặt seed cho random và NumPy")


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    """Lưu dữ liệu kiểu dict thành file JSON định dạng UTF-8 đẹp mắt."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def calculate_ece(
    confidences: np.ndarray,
    predictions: np.ndarray,
    targets: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Tính toán Expected Calibration Error (ECE) đo lường độ lệch xác suất.

    ECE chia dải xác suất [0, 1] thành n_bins khoảng bằng nhau, tính chênh lệch
    tuyệt đối giữa độ chính xác thực tế (accuracy) và độ tin cậy trung bình (confidence)
    trong từng khoảng, sau đó lấy trung bình có trọng số theo số lượng mẫu.
    """
    if len(confidences) == 0:
        return 0.0

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    total_samples = len(confidences)

    correct_mask = (predictions == targets).astype(float)

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        if i == n_bins - 1:
            in_bin = (confidences >= bin_lower) & (confidences <= bin_upper)
        else:
            in_bin = (confidences >= bin_lower) & (confidences < bin_upper)

        bin_size = np.sum(in_bin)
        if bin_size > 0:
            bin_acc = np.mean(correct_mask[in_bin])
            bin_conf = np.mean(confidences[in_bin])
            ece += (bin_size / total_samples) * abs(bin_acc - bin_conf)

    return float(ece)


def calculate_entropy(probabilities: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Tính toán Entropy Shannon H(p) = -sum(p * log(p)) để đo độ phân tán xác suất.

    Args:
        probabilities (np.ndarray): Mảng xác suất 1D (shape: C,) hoặc 2D (shape: N, C).
        eps (float): Hệ số chống log(0).

    Returns:
        np.ndarray: Giá trị entropy tương ứng từng mẫu.
    """
    p = np.clip(probabilities, eps, 1.0)
    if p.ndim == 1:
        return float(-np.sum(p * np.log(p)))
    return -np.sum(p * np.log(p), axis=1)


# Biểu thức chính quy cho PII Redaction
_CARD_REGEX = re.compile(r"\b(?:\d[ -]*?){13,19}\b")
_EMAIL_REGEX = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")
_PHONE_REGEX = re.compile(r"(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b")


def redact_pii(text: str) -> str:
    """Che giấu thông tin nhạy cảm của khách hàng trước khi log telemetry.

    Bảo vệ các dữ liệu ngân hàng nhạy cảm:
    - Số thẻ tín dụng / thẻ ghi nợ (13-19 chữ số).
    - Địa chỉ email cá nhân.
    - Số điện thoại liên lạc.
    """
    redacted = _CARD_REGEX.sub("[CARD_NUMBER]", text)
    redacted = _EMAIL_REGEX.sub("[EMAIL]", redacted)
    redacted = _PHONE_REGEX.sub("[PHONE_NUMBER]", redacted)
    return redacted
