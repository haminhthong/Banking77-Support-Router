"""Release gate và promotion an toàn cho model bundle."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .artifact import load_and_validate_bundle


@dataclass(frozen=True)
class ReleaseGateResult:
    """Kết quả immutable của một lần kiểm tra release."""

    passed: bool
    checks: dict[str, dict[str, Any]]

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": "PASSED" if self.passed else "FAILED",
            "passed": self.passed,
            "checked_at": datetime.now(timezone.utc).isoformat(),
            "checks": self.checks,
        }


def evaluate_release_gate(
    test_metrics: dict[str, Any],
    ood_metrics: dict[str, Any],
    gate_config: dict[str, Any] | None = None,
) -> ReleaseGateResult:
    """Kiểm tra các SLO release trên report đã sinh ra, không tự bịa metric."""
    cfg = gate_config or {}
    checks = {
        "wrong_queue_rate": {
            "value": float(test_metrics.get("auto_route_wrong_queue_rate", 1.0)),
            "operator": "<=",
            "limit": float(cfg.get("max_wrong_queue_rate", 0.05)),
        },
        "security_recall": {
            "value": float(test_metrics.get("high_risk_escalation_recall", 0.0)),
            "operator": ">=",
            "limit": float(cfg.get("min_security_recall", 0.95)),
        },
        "unsupported_auto_route_rate": {
            "value": float(ood_metrics.get("oos_auto_route_rate", 1.0)),
            "operator": "<=",
            "limit": float(cfg.get("max_unsupported_auto_route_rate", 0.05)),
        },
        "auto_route_coverage": {
            "value": float(test_metrics.get("operational_auto_route_coverage", 0.0)),
            "operator": ">=",
            "limit": float(cfg.get("min_auto_route_coverage", 0.65)),
        },
    }
    checks["wrong_queue_rate"]["passed"] = checks["wrong_queue_rate"]["value"] <= checks["wrong_queue_rate"]["limit"]
    checks["security_recall"]["passed"] = checks["security_recall"]["value"] >= checks["security_recall"]["limit"]
    checks["unsupported_auto_route_rate"]["passed"] = checks["unsupported_auto_route_rate"]["value"] <= checks["unsupported_auto_route_rate"]["limit"]
    checks["auto_route_coverage"]["passed"] = checks["auto_route_coverage"]["value"] >= checks["auto_route_coverage"]["limit"]
    return ReleaseGateResult(passed=all(bool(item["passed"]) for item in checks.values()), checks=checks)


def write_release_gate(release_dir: str | Path, result: ReleaseGateResult) -> Path:
    """Ghi kết quả gate cạnh manifest của release candidate."""
    path = Path(release_dir) / "release_gate.json"
    path.write_text(json.dumps(result.as_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def promote_release(
    models_dir: str | Path,
    release_name: str,
    gate_result: ReleaseGateResult,
) -> Path:
    """Chỉ cập nhật production pointer sau khi gate pass và bundle hợp lệ."""
    if not gate_result.passed:
        raise ValueError("Release gate FAILED; không được promote model này")
    root = Path(models_dir)
    release_dir = root / "releases" / release_name
    load_and_validate_bundle(release_dir, verify_checksum=True)
    pointer = root / "production.json"
    payload = {
        "release": f"releases/{release_name}",
        "model_version": json.loads((release_dir / "manifest.json").read_text(encoding="utf-8")).get("model_version"),
        "policy_version": json.loads((release_dir / "manifest.json").read_text(encoding="utf-8")).get("policy_version"),
        "promotion_status": "PROMOTED",
        "promoted_at": datetime.now(timezone.utc).isoformat(),
    }
    pointer.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return pointer
