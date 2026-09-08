"""Quét security risk trên toàn bộ phân phối 77 intent."""

from __future__ import annotations

from collections import defaultdict
from typing import Sequence

import numpy as np

from .schemas import RiskAssessment
from .taxonomy import TaxonomyResolver


class RiskAssessor:
    """Tách security risk khỏi top-k và priority vận hành.

    Khi có taxonomy, chỉ các intent có ``risk.priority_escalation`` mới tạo
    priority review.  Tham số ``high_risk_intents`` chỉ giữ cho test/client cũ;
    không được dùng khi runtime đã có taxonomy.
    """

    def __init__(
        self,
        classes: Sequence[str],
        high_risk_intents: frozenset[str] | set[str] | None = None,
        high_risk_trigger: float = 0.20,
        taxonomy: TaxonomyResolver | None = None,
    ) -> None:
        if not 0.0 <= high_risk_trigger <= 1.0:
            raise ValueError("high_risk_trigger phải nằm trong [0, 1]")

        self.classes = np.asarray(classes)
        self.class_to_idx = {str(label): i for i, label in enumerate(classes)}
        self.taxonomy = taxonomy
        self.high_risk_trigger = float(high_risk_trigger)

        if taxonomy is not None:
            risk_intents = {
                intent for intent in self.class_to_idx
                if taxonomy.is_priority_escalation(intent)
            }
        else:
            # Chỉ phục vụ facade tương thích, không đi qua đường production.
            from ..data.contracts import DEFAULT_HIGH_RISK_INTENTS

            risk_intents = set(high_risk_intents or DEFAULT_HIGH_RISK_INTENTS)

        self.high_risk_intents = frozenset(risk_intents)
        self.risk_indices = [self.class_to_idx[i] for i in self.high_risk_intents if i in self.class_to_idx]

        groups: dict[str, list[int]] = defaultdict(list)
        if taxonomy is not None:
            for intent, index in self.class_to_idx.items():
                category = taxonomy.get_risk_category(intent)
                if category != "none":
                    groups[category].append(index)
        else:
            groups["security"] = list(self.risk_indices)
        self.risk_group_indices = dict(groups)

    def assess(
        self,
        probabilities: np.ndarray,
        top_intent: str,
        ood_detected: bool = False,
        ood_reasons: list[str] | None = None,
    ) -> RiskAssessment:
        """Tính mass theo nhóm risk, độc lập với top-k hiển thị."""
        del top_intent
        probs = np.asarray(probabilities, dtype=float)
        reason_codes = list(ood_reasons or [])
        group_mass = {
            category: float(probs[indices].sum())
            for category, indices in self.risk_group_indices.items()
            if indices
        }
        critical_probability = float(probs[self.risk_indices].sum()) if self.risk_indices else 0.0

        max_intent: str | None = None
        max_score = 0.0
        max_category: str | None = None
        if self.risk_indices:
            best_index = max(self.risk_indices, key=lambda index: float(probs[index]))
            max_score = float(probs[best_index])
            max_intent = str(self.classes[best_index])
            if self.taxonomy is not None:
                max_category = self.taxonomy.get_risk_category(max_intent)

        high_risk_detected = critical_probability >= self.high_risk_trigger
        if high_risk_detected:
            reason_codes.append("CRITICAL_RISK_MASS")
            if max_category:
                reason_codes.append(f"RISK_GROUP_{max_category.upper()}")

        return RiskAssessment(
            high_risk_detected=high_risk_detected,
            high_risk_intent=max_intent if max_score > 0.05 else None,
            high_risk_score=max_score,
            ood_detected=ood_detected,
            reason_codes=reason_codes,
            critical_probability=critical_probability,
            risk_category=max_category,
            risk_group_mass=group_mass,
        )
