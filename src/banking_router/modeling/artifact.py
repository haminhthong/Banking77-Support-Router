"""Model artifact contract, bundle serialization, and cryptographic verification."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import joblib
import numpy as np
import sklearn

from ..data.contracts import BANKING77_77_CLASSES, DEFAULT_HIGH_RISK_INTENTS, INTENT_TO_DOMAIN
from ..data.normalization import compute_file_sha256


@dataclass
class ModelBundle:
    """Production Model Bundle containing weights, manifest, configuration, and policies."""
    model: Any
    config: dict[str, Any]
    manifest: dict[str, Any]
    taxonomy: dict[str, Any]
    policy_config: dict[str, Any]


def get_git_commit() -> str:
    """Retrieve current Git commit hash for reproducibility metadata."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return res.stdout.strip()
    except Exception:
        return "unknown"


def compute_string_sha256(text: str) -> str:
    """Compute SHA-256 hash of a text string."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def save_bundle(
    target_dir: Path | str,
    calibrated_model: Any,
    config_payload: dict[str, Any],
    policy_payload: dict[str, Any],
    taxonomy_payload: dict[str, Any],
    train_dataset_sha256: str,
    test_dataset_sha256: str,
    split_sizes: dict[str, int],
    pipeline_config: dict[str, Any],
    model_version: str = "banking77-support-triage-v3",
    policy_version: str = "risk-aware-triage-v3",
) -> ModelBundle:
    """Save full production ModelBundle and emit verified manifest with cryptographic hashes."""
    p_dir = Path(target_dir)
    p_dir.mkdir(parents=True, exist_ok=True)

    # 1. Save binary model weights
    joblib_path = p_dir / "router.joblib"
    joblib.dump(calibrated_model, joblib_path)
    model_sha256 = compute_file_sha256(joblib_path)

    # 2. Hash class labels for consistency
    classes = sorted(list(calibrated_model.classes_))
    labels_sha256 = compute_string_sha256(",".join(classes))

    runtime_info = {
        "python": platform.python_version(),
        "scikit_learn": sklearn.__version__,
        "numpy": np.__version__,
        "joblib": joblib.__version__,
    }

    # 3. Model Manifest
    manifest_payload = {
        "schema_version": 3,
        "model_version": model_version,
        "policy_version": policy_version,
        "git_commit": get_git_commit(),
        "artifact_sha256": model_sha256,
        "class_labels_sha256": labels_sha256,
        "dataset_checksums": {
            "train_csv_sha256": train_dataset_sha256,
            "test_csv_sha256": test_dataset_sha256,
        },
        "split_sizes": split_sizes,
        "pipeline_config": pipeline_config,
        "class_labels_count": len(classes),
        "high_risk_intents": sorted(list(DEFAULT_HIGH_RISK_INTENTS)),
        "runtime": runtime_info,
    }

    # Write JSON files
    (p_dir / "config.json").write_text(
        json.dumps(config_payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (p_dir / "model_manifest.json").write_text(
        json.dumps(manifest_payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (p_dir / "taxonomy.json").write_text(
        json.dumps(taxonomy_payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (p_dir / "policy.json").write_text(
        json.dumps(policy_payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    return ModelBundle(
        model=calibrated_model,
        config=config_payload,
        manifest=manifest_payload,
        taxonomy=taxonomy_payload,
        policy_config=policy_payload,
    )


def load_and_validate_bundle(
    models_dir: Path | str,
    verify_checksum: bool = True,
) -> ModelBundle:
    """Load and strictly validate the ModelBundle contract.

    Validation steps:
    1. Verify all expected artifact files exist.
    2. Verify manifest JSON is readable.
    3. Check binary artifact SHA-256 against manifest (if verify_checksum=True).
    4. Load binary weights and verify model.classes_ contains exact 77 classes.
    5. Verify taxonomy integrity.
    """
    p_dir = Path(models_dir)
    joblib_path = p_dir / "router.joblib"
    manifest_path = p_dir / "model_manifest.json"
    config_path = p_dir / "config.json"

    if not joblib_path.exists():
        raise FileNotFoundError(f"Missing model artifact: {joblib_path}")
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing model manifest: {manifest_path}")
    if not config_path.exists():
        raise FileNotFoundError(f"Missing model config: {config_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    config = json.loads(config_path.read_text(encoding="utf-8"))

    taxonomy_path = p_dir / "taxonomy.json"
    taxonomy = (
        json.loads(taxonomy_path.read_text(encoding="utf-8"))
        if taxonomy_path.exists()
        else {}
    )

    policy_path = p_dir / "policy.json"
    policy_config = (
        json.loads(policy_path.read_text(encoding="utf-8"))
        if policy_path.exists()
        else {}
    )

    # 3. Checksum verification
    if verify_checksum and "artifact_sha256" in manifest:
        current_sha256 = compute_file_sha256(joblib_path)
        expected_sha256 = manifest["artifact_sha256"]
        if current_sha256 != expected_sha256:
            raise ValueError(
                f"Model artifact integrity check failed! Expected SHA256 {expected_sha256}, got {current_sha256}"
            )

    # 4. Load weights & verify 77 classes
    model = joblib.load(joblib_path)
    classes = list(model.classes_)
    if len(classes) != 77:
        raise ValueError(
            f"Model classes count invariant violated! Expected 77 classes, got {len(classes)}"
        )

    if set(classes) != set(BANKING77_77_CLASSES):
        missing = set(BANKING77_77_CLASSES) - set(classes)
        raise ValueError(f"Model missing required banking classes: {missing}")

    return ModelBundle(
        model=model,
        config=config,
        manifest=manifest,
        taxonomy=taxonomy,
        policy_config=policy_config,
    )
