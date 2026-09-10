"""Lưu và tải các artifact dùng cho train, evaluate và inference."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib

from ..data.contracts import BANKING77_77_CLASSES


@dataclass
class ModelArtifacts:
    """Các artifact canonical của một pipeline Banking77."""

    intent_model: Any
    metadata: dict[str, Any]
    taxonomy: dict[str, Any]
    routing_policy: dict[str, Any]
    scope_model: Any | None = None


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Ghi JSON UTF-8 với LF để artifact có cùng nội dung trên mọi hệ điều hành."""
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    with path.open("w", encoding="utf-8", newline="\n") as file:
        file.write(content)


def save_artifacts(
    artifacts_dir: str | Path,
    intent_model: Any,
    scope_model: Any | None,
    metadata: dict[str, Any],
    taxonomy: dict[str, Any],
    routing_policy: dict[str, Any],
) -> ModelArtifacts:
    """Lưu đúng một bộ artifact canonical cho model đã chọn."""
    artifact_dir = Path(artifacts_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(intent_model, artifact_dir / "intent_model.joblib")
    if scope_model is not None:
        joblib.dump(scope_model, artifact_dir / "scope_model.joblib")

    _write_json(artifact_dir / "metadata.json", metadata)
    _write_json(artifact_dir / "taxonomy.json", taxonomy)
    _write_json(artifact_dir / "routing_policy.json", routing_policy)

    return ModelArtifacts(
        intent_model=intent_model,
        metadata=metadata,
        taxonomy=taxonomy,
        routing_policy=routing_policy,
        scope_model=scope_model,
    )


def load_artifacts(artifacts_dir: str | Path) -> ModelArtifacts:
    """Tải artifact canonical và kiểm tra các invariant ML cần thiết."""
    artifact_dir = Path(artifacts_dir)
    model_path = artifact_dir / "intent_model.joblib"
    metadata_path = artifact_dir / "metadata.json"
    taxonomy_path = artifact_dir / "taxonomy.json"
    policy_path = artifact_dir / "routing_policy.json"

    required_paths = (model_path, metadata_path, taxonomy_path, policy_path)
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Thiếu artifact canonical: {', '.join(missing)}")

    intent_model = joblib.load(model_path)
    scope_path = artifact_dir / "scope_model.joblib"
    scope_model = joblib.load(scope_path) if scope_path.exists() else None
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    taxonomy = json.loads(taxonomy_path.read_text(encoding="utf-8"))
    routing_policy = json.loads(policy_path.read_text(encoding="utf-8"))

    classes = list(intent_model.classes_)
    if len(classes) != len(BANKING77_77_CLASSES):
        raise ValueError(
            f"Model phải có {len(BANKING77_77_CLASSES)} intent, nhận được {len(classes)}"
        )
    if set(classes) != set(BANKING77_77_CLASSES):
        raise ValueError("Nhãn model không khớp bộ 77 intent Banking77")

    taxonomy_intents = set(taxonomy.get("intents", {}))
    if taxonomy_intents != set(BANKING77_77_CLASSES):
        raise ValueError("Taxonomy không khớp bộ 77 intent Banking77")

    return ModelArtifacts(
        intent_model=intent_model,
        metadata=metadata,
        taxonomy=taxonomy,
        routing_policy=routing_policy,
        scope_model=scope_model,
    )
