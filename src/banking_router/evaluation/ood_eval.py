"""Out-of-Distribution (OOD) benchmark evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from ..routing.service import RoutingService


def evaluate_ood_benchmark(
    ood_file: Path | str,
    routing_service: RoutingService,
) -> dict[str, Any]:
    """Evaluate Out-of-Distribution detection performance on dedicated out-of-scope dataset."""
    path = Path(ood_file)
    if not path.exists():
        return {
            "ood_evaluated": False,
            "reason": f"OOD evaluation dataset not found at {path}",
        }

    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if not records:
        return {"ood_evaluated": False, "reason": "Empty OOD evaluation dataset"}

    total_samples = len(records)
    ood_detected_count = 0
    auto_routed_count = 0
    human_reviewed_count = 0
    priority_escalated_count = 0

    category_breakdown: dict[str, dict[str, int]] = {}

    for item in records:
        text = item["text"]
        cat = item.get("category", "unknown")
        if cat not in category_breakdown:
            category_breakdown[cat] = {"total": 0, "detected": 0, "auto_routed": 0}
        category_breakdown[cat]["total"] += 1

        res = routing_service.route(text)
        if res.risk.ood_detected:
            ood_detected_count += 1
            category_breakdown[cat]["detected"] += 1

        if res.decision.action == "auto_route":
            auto_routed_count += 1
            category_breakdown[cat]["auto_routed"] += 1
        elif res.decision.action == "priority_human_review":
            priority_escalated_count += 1
        else:
            human_reviewed_count += 1

    ood_recall = float(ood_detected_count / total_samples)
    false_acceptance_rate = float(auto_routed_count / total_samples)
    safe_containment_rate = float(1.0 - false_acceptance_rate)

    return {
        "ood_evaluated": True,
        "ood_total_samples": total_samples,
        "ood_detected_count": ood_detected_count,
        "ood_recall": round(ood_recall, 4),
        "ood_false_acceptance_rate": round(false_acceptance_rate, 4),
        "safe_containment_rate": round(safe_containment_rate, 4),
        "oos_auto_route_rate": round(false_acceptance_rate, 4),
        "oos_containment_rate": round(safe_containment_rate, 4),
        "auto_routed_count": auto_routed_count,
        "human_reviewed_count": human_reviewed_count,
        "priority_escalated_count": priority_escalated_count,
        "category_breakdown": category_breakdown,
    }
