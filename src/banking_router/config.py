"""Các đường dẫn và loader cấu hình của dự án."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs"
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
REPORTS_DIR = PROJECT_ROOT / "reports"


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Đọc file YAML UTF-8."""
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file cấu hình: {config_path}")
    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def get_taxonomy_config() -> dict[str, Any]:
    return load_yaml(CONFIG_DIR / "taxonomy.yaml")


def get_routing_policy_config() -> dict[str, Any]:
    return load_yaml(CONFIG_DIR / "routing_policy.yaml")


def get_model_config() -> dict[str, Any]:
    return load_yaml(CONFIG_DIR / "model.yaml")
