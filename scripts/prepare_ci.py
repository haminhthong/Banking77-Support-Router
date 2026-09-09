"""Chuẩn bị dataset và model fixture cho CI trên checkout sạch."""

from __future__ import annotations

import json
import os
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


def _has_bundle(release_dir: Path) -> bool:
    required_files = ("router.joblib", "manifest.json", "model_config.json")
    if not all((release_dir / name).exists() for name in required_files):
        return False

    try:
        manifest_path = release_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False

    artifact_files = {
        "model": "router.joblib",
        "model_config": "model_config.json",
        "taxonomy": "taxonomy.json",
        "routing_policy": "routing_policy.json",
        "scope_model": "scope_model.joblib",
    }
    protected_artifacts = manifest.get("artifact_hashes", {})
    if not isinstance(protected_artifacts, dict):
        return False
    if any(name not in artifact_files for name in protected_artifacts):
        return False
    return all(
        (release_dir / artifact_files[name]).exists()
        for name in protected_artifacts
    )


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


def main() -> None:
    os.chdir(ROOT)
    _ensure_dataset()
    release_dir = _ensure_release()
    load_and_validate_bundle(release_dir, verify_checksum=True)
    print(f"CI fixtures ready: {release_dir}")


if __name__ == "__main__":
    main()
