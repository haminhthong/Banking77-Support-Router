"""Tách dữ liệu scope theo split cố định, không dùng locked set để tune."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_scope_splits(path: str | Path = "data/evaluation/ood.jsonl") -> dict[str, list[dict[str, Any]]]:
    """Đọc OOS benchmark và tạo train/dev/locked theo thứ tự ổn định.

    Quy tắc tách deterministic giúp mọi lần train dùng cùng locked examples.
    Dev có thể dùng để chọn threshold; locked chỉ dùng báo cáo cuối.
    """
    records = [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not records:
        raise ValueError(f"Scope dataset rỗng: {path}")
    return {
        "train": [record for index, record in enumerate(records) if index % 5 not in (0, 1)],
        "dev": [record for index, record in enumerate(records) if index % 5 == 1],
        "locked": [record for index, record in enumerate(records) if index % 5 == 0],
    }
