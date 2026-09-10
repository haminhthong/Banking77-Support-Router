"""Chuẩn bị dataset và xác nhận artifact duy nhất cho CI."""

from __future__ import annotations

import os
from pathlib import Path

from scripts.download_data import main as download_data
from src.banking_router.config import ARTIFACTS_DIR
from src.banking_router.modeling.artifact import load_artifacts

ROOT = Path(__file__).resolve().parents[1]


def ensure_dataset() -> None:
    raw_dir = ROOT / "data" / "raw"
    if not all((raw_dir / filename).exists() for filename in ("train.csv", "test.csv")):
        download_data()


def ensure_artifacts() -> None:
    load_artifacts(ARTIFACTS_DIR)


def main() -> None:
    os.chdir(ROOT)
    ensure_dataset()
    ensure_artifacts()
    print(f"CI inputs ready: {ARTIFACTS_DIR}")


if __name__ == "__main__":
    main()
