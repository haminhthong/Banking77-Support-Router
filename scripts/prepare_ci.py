"""Chuẩn bị dataset và model facade cho CI trên checkout sạch."""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from download_data import EXPECTED_SHA256, compute_sha256, main as download_data
from src.banking_router.modeling.artifact import load_and_validate_bundle
from src.banking_router.modeling.training import train_and_optimize


MODELS_DIR = ROOT / "models"
RELEASE_NAME = "banking-router-v5"
CI_RELEASE_NAME = "banking-router-ci"

BUNDLE_FILES = (
    "router.joblib",
    "scope_model.joblib",
    "manifest.json",
    "model_manifest.json",
    "model_config.json",
    "config.json",
    "taxonomy.json",
    "routing_policy.json",
    "policy.json",
)


def _has_bundle(release_dir: Path) -> bool:
    return all((release_dir / name).exists() for name in ("router.joblib", "manifest.json", "model_config.json"))


def _ensure_dataset() -> None:
    raw_dir = ROOT / "data" / "raw"
    valid = all(
        (raw_dir / name).exists()
        and compute_sha256((raw_dir / name).read_bytes()) == EXPECTED_SHA256[name]
        for name in ("train.csv", "test.csv")
    )
    if not valid:
        download_data()


def _ensure_release() -> Path:
    release_dir = MODELS_DIR / "releases" / RELEASE_NAME
    if _has_bundle(release_dir):
        return release_dir

    _ensure_dataset()
    ci_release_dir = MODELS_DIR / "releases" / CI_RELEASE_NAME
    train_and_optimize(
        models_dir=ci_release_dir,
        reports_dir=ROOT / "reports" / "ci-training",
    )
    load_and_validate_bundle(ci_release_dir, verify_checksum=True)

    pointer_path = MODELS_DIR / "production.json"
    pointer_path.write_text(
        json.dumps(
            {
                "release": f"releases/{CI_RELEASE_NAME}",
                "model_version": "banking77-tfidf-char-lr-v5",
                "policy_version": "queue-policy-v5",
                "promotion_status": "CI_BOOTSTRAPPED",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return ci_release_dir


def _create_legacy_model_view(release_dir: Path) -> None:
    """Đồng bộ alias models/ cho các test/client legacy còn đọc đường dẫn cũ."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    required = ("router.joblib", "model_manifest.json", "model_config.json")
    if all((MODELS_DIR / name).exists() for name in required):
        print("Using existing legacy model view")
        return
    for name in BUNDLE_FILES:
        source = release_dir / name
        if source.exists():
            shutil.copy2(source, MODELS_DIR / name)


def main() -> None:
    os.chdir(ROOT)
    _ensure_dataset()
    release_dir = _ensure_release()
    load_and_validate_bundle(release_dir, verify_checksum=True)
    _create_legacy_model_view(release_dir)
    print(f"CI fixtures ready: {release_dir}")


if __name__ == "__main__":
    main()
