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

from ..data.contracts import BANKING77_77_CLASSES
from ..data.normalization import compute_file_sha256


@dataclass
class ModelBundle:
    """Bundle bất biến gồm model intent, scope model và policy executable."""
    model: Any
    config: dict[str, Any]
    manifest: dict[str, Any]
    taxonomy: dict[str, Any]
    policy_config: dict[str, Any]
    scope_model: Any | None = None


def get_git_commit() -> str:
    """Lấy commit Git hiện tại để tái lập artifact."""
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
    """Tính SHA-256 cho chuỗi text."""
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
    model_version: str = "banking77-tfidf-char-lr-v5",
    policy_version: str = "queue-policy-v5",
    scope_model: Any | None = None,
) -> ModelBundle:
    """Lưu bundle và manifest có hash cho mọi artifact executable."""
    p_dir = Path(target_dir)
    p_dir.mkdir(parents=True, exist_ok=True)

    # Lưu model intent.
    joblib_path = p_dir / "router.joblib"
    joblib.dump(calibrated_model, joblib_path)
    model_sha256 = compute_file_sha256(joblib_path)
    scope_model_sha256: str | None = None
    if scope_model is not None:
        scope_path = p_dir / "scope_model.joblib"
        joblib.dump(scope_model, scope_path)
        scope_model_sha256 = compute_file_sha256(scope_path)

    # Hash nhãn để phát hiện model và taxonomy lệch nhau.
    classes = sorted(list(calibrated_model.classes_))
    labels_sha256 = compute_string_sha256(",".join(classes))

    runtime_info = {
        "python": platform.python_version(),
        "scikit_learn": sklearn.__version__,
        "numpy": np.__version__,
        "joblib": joblib.__version__,
    }

    # File canonical; alias cũ chỉ để client offline còn đọc được.
    config_text = json.dumps(config_payload, indent=2, ensure_ascii=False)
    taxonomy_text = json.dumps(taxonomy_payload, indent=2, ensure_ascii=False)
    policy_text = json.dumps(policy_payload, indent=2, ensure_ascii=False)
    (p_dir / "model_config.json").write_text(config_text, encoding="utf-8")
    (p_dir / "config.json").write_text(config_text, encoding="utf-8")
    (p_dir / "taxonomy.json").write_text(taxonomy_text, encoding="utf-8")
    (p_dir / "routing_policy.json").write_text(policy_text, encoding="utf-8")
    (p_dir / "policy.json").write_text(policy_text, encoding="utf-8")

    # Policy và taxonomy là cấu hình executable nên phải được hash cùng model.
    artifact_hashes = {
        "model": model_sha256,
        "model_config": compute_file_sha256(p_dir / "model_config.json"),
        "taxonomy": compute_file_sha256(p_dir / "taxonomy.json"),
        "routing_policy": compute_file_sha256(p_dir / "routing_policy.json"),
    }
    if scope_model_sha256:
        artifact_hashes["scope_model"] = scope_model_sha256
    manifest_payload = {
        "schema_version": 4,
        "model_version": model_version,
        "policy_version": policy_version,
        "git_commit": get_git_commit(),
        "artifact_sha256": model_sha256,
        "artifact_hashes": artifact_hashes,
        "class_labels_sha256": labels_sha256,
        "dataset_checksums": {
            "train_csv_sha256": train_dataset_sha256,
            "test_csv_sha256": test_dataset_sha256,
        },
        "split_sizes": split_sizes,
        "pipeline_config": pipeline_config,
        "class_labels_count": len(classes),
        "normalization_version": "semantic-pii-v1",
        "scope_model_version": getattr(scope_model, "model_version", None),
        "runtime": runtime_info,
    }

    manifest_text = json.dumps(manifest_payload, indent=2, ensure_ascii=False)
    (p_dir / "manifest.json").write_text(manifest_text, encoding="utf-8")
    (p_dir / "model_manifest.json").write_text(manifest_text, encoding="utf-8")

    return ModelBundle(
        model=calibrated_model,
        config=config_payload,
        manifest=manifest_payload,
        taxonomy=taxonomy_payload,
        policy_config=policy_payload,
        scope_model=scope_model,
    )


def load_and_validate_bundle(
    models_dir: Path | str,
    verify_checksum: bool = True,
) -> ModelBundle:
    """Load và xác thực nghiêm ngặt hợp đồng ModelBundle.

    Validation steps:
    1. Verify all expected artifact files exist.
    2. Verify manifest JSON is readable.
    3. Check binary artifact SHA-256 against manifest (if verify_checksum=True).
    4. Load binary weights and verify model.classes_ contains exact 77 classes.
    5. Verify taxonomy integrity.
    """
    p_dir = Path(models_dir)
    joblib_path = p_dir / "router.joblib"
    manifest_path = p_dir / "manifest.json"
    if not manifest_path.exists():
        manifest_path = p_dir / "model_manifest.json"
    config_path = p_dir / "model_config.json"
    if not config_path.exists():
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

    policy_path = p_dir / "routing_policy.json"
    if not policy_path.exists():
        policy_path = p_dir / "policy.json"
    policy_config = (
        json.loads(policy_path.read_text(encoding="utf-8"))
        if policy_path.exists()
        else {}
    )
    scope_path = p_dir / "scope_model.joblib"
    scope_model = joblib.load(scope_path) if scope_path.exists() else None

    # Xác minh checksum.
    if verify_checksum:
        expected_model_sha = manifest.get("artifact_hashes", {}).get(
            "model", manifest.get("artifact_sha256")
        )
        if expected_model_sha:
            current_sha256 = compute_file_sha256(joblib_path)
            if current_sha256 != expected_model_sha:
                raise ValueError(
                    f"Model artifact integrity check failed! Expected SHA256 {expected_model_sha}, got {current_sha256}"
                )

        # Manifest mới khóa mọi artifact executable; bundle cũ chỉ để migrate offline.
        for artifact_name, expected_sha in manifest.get("artifact_hashes", {}).items():
            if artifact_name == "model":
                continue
            artifact_file = {
                "model_config": config_path,
                "taxonomy": taxonomy_path,
                "routing_policy": policy_path,
                "scope_model": scope_path,
            }.get(artifact_name)
            if artifact_file is None or not artifact_file.exists():
                raise FileNotFoundError(f"Missing integrity-protected artifact: {artifact_name}")
            current_sha = compute_file_sha256(artifact_file)
            if current_sha != expected_sha:
                raise ValueError(
                    f"{artifact_name} integrity check failed! Expected SHA256 {expected_sha}, got {current_sha}"
                )

    # Load weights và xác minh đủ 77 class.
    model = joblib.load(joblib_path)
    classes = list(model.classes_)
    if len(classes) != 77:
        raise ValueError(
            f"Model classes count invariant violated! Expected 77 classes, got {len(classes)}"
        )

    if set(classes) != set(BANKING77_77_CLASSES):
        missing = set(BANKING77_77_CLASSES) - set(classes)
        raise ValueError(f"Model missing required banking classes: {missing}")

    if taxonomy:
        taxonomy_intents = set(taxonomy.get("intents", {}))
        if taxonomy_intents and taxonomy_intents != set(BANKING77_77_CLASSES):
            raise ValueError("Taxonomy intent set does not match Banking77's 77 classes")

    return ModelBundle(
        model=model,
        config=config,
        manifest=manifest,
        taxonomy=taxonomy,
        policy_config=policy_config,
        scope_model=scope_model,
    )
