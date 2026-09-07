"""Unified configuration loader for Banking77 Support Router."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs"
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load YAML file with UTF-8 encoding."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def get_taxonomy_config() -> dict[str, Any]:
    """Load taxonomy and queue configuration."""
    return load_yaml(CONFIG_DIR / "taxonomy.yaml")


def get_routing_policy_config() -> dict[str, Any]:
    """Load routing policy and SLO configuration."""
    return load_yaml(CONFIG_DIR / "routing_policy.yaml")


def get_model_config() -> dict[str, Any]:
    """Load ML pipeline and calibration configuration."""
    return load_yaml(CONFIG_DIR / "model.yaml")
