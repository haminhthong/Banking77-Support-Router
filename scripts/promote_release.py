"""Kiểm tra release gate và promote model candidate nếu đạt SLO."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.banking_router.config import get_routing_policy_config
from src.banking_router.modeling.release import evaluate_release_gate, promote_release, write_release_gate


def main() -> int:
    parser = argparse.ArgumentParser(description="Promote một Banking Router release đã đạt gate")
    parser.add_argument("--release", required=True, help="Tên thư mục trong models/releases")
    parser.add_argument("--reports", default="reports", help="Thư mục chứa test_metrics.json và ood_metrics.json")
    parser.add_argument("--models", default="models", help="Thư mục models")
    args = parser.parse_args()

    reports = Path(args.reports)
    test_metrics = json.loads((reports / "test_metrics.json").read_text(encoding="utf-8"))
    ood_metrics = json.loads((reports / "ood_metrics.json").read_text(encoding="utf-8"))
    gate = evaluate_release_gate(
        test_metrics,
        ood_metrics,
        get_routing_policy_config().get("release_gate", {}),
    )
    release_dir = Path(args.models) / "releases" / args.release
    write_release_gate(release_dir, gate)
    print(json.dumps(gate.as_dict(), indent=2, ensure_ascii=False))
    if not gate.passed:
        print("Release bi tu choi: khong cap nhat production pointer.")
        return 2
    promote_release(args.models, args.release, gate)
    print(f"Da promote: {args.release}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
