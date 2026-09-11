"""Tiện ích dùng chung cho log, seed và ghi JSON."""

from __future__ import annotations

import logging
import os
import random
import sys
from contextlib import suppress

import numpy as np

LOGGER = logging.getLogger("banking_router")


def setup_logging() -> None:
    """Thiết lập stdout UTF-8 và một định dạng log thống nhất."""
    if hasattr(sys.stdout, "reconfigure"):
        with suppress(OSError, ValueError):
            sys.stdout.reconfigure(encoding="utf-8")

    level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def set_seed(seed: int) -> None:
    """Cố định seed để các lần chạy có thể tái lập."""
    random.seed(seed)
    np.random.seed(seed)
