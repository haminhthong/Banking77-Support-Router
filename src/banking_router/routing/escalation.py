"""Đánh giá tín hiệu intent nhạy cảm để ưu tiên human review."""

from __future__ import annotations

from collections import defaultdict
from typing import Sequence

import numpy as np

from .schemas import SensitiveCaseAssessment
from .taxonomy import TaxonomyResolver


class SensitiveIntentGuard:
    """Quét probability mass của nhóm intent nhạy cảm trên toàn bộ 77 lớp.

    Guard này không kết luận gian lận hay sự cố bảo mật. Nó chỉ tạo tín hiệu
    để routing policy ưu tiên ticket cho người kiểm tra.
    """

    def __init__(
        self,
        classes: Sequence[str],
        sensitive_intents: frozenset[str] | set[str] | None = None,
        sensitive_trigger: float = 0.20,
        taxonomy: TaxonomyResolver | None = None,
    ) -> None:
        if not 0.0 <= sensitive_trigger <= 1.0:
            raise ValueError("sensitive_trigger phải nằm trong [0, 1]")
        self.classes = np.asarray(classes)
        self.class_to_idx = {str(label): index for index, label in enumerate(classes)}
        self.taxonomy = taxonomy
        self.sensitive_trigger = float(sensitive_trigger)
        if taxonomy is not None:
            selected = {
                intent for intent in self.class_to_idx
                if taxonomy.requires_priority_review(intent)
            }
        else:
            selected = set(sensitive_intents or ())
        self.sensitive_intents = frozenset(selected)
        self.sensitive_indices = [self.class_to_idx[item] for item in self.sensitive_intents]

        groups: dict[str, list[int]] = defaultdict(list)
        for intent, index in self.class_to_idx.items():
            category = taxonomy.get_sensitive_category(intent) if taxonomy else "sensitive_case"
            if intent in self.sensitive_intents and category != "none":
                groups[category].append(index)
        self.sensitive_group_indices = dict(groups)

    def assess(
        self,
        probabilities: np.ndarray,
        scope_detected: bool = False,
        scope_reasons: list[str] | None = None,
    ) -> SensitiveCaseAssessment:
        probs = np.asarray(probabilities, dtype=float)
        if probs.ndim != 1 or len(probs) != len(self.classes):
            raise ValueError("probabilities phải có đúng một giá trị cho mỗi intent")
        group_mass = {
            category: float(probs[indices].sum())
            for category, indices in self.sensitive_group_indices.items()
            if indices
        }
        sensitive_mass = float(probs[self.sensitive_indices].sum()) if self.sensitive_indices else 0.0
        best_intent = None
        best_score = 0.0
        best_category = None
        if self.sensitive_indices:
            best_index = max(self.sensitive_indices, key=lambda index: float(probs[index]))
            best_score = float(probs[best_index])
            best_intent = str(self.classes[best_index])
            if self.taxonomy:
                best_category = self.taxonomy.get_sensitive_category(best_intent)

        reason_codes = list(scope_reasons or [])
        requires_review = sensitive_mass >= self.sensitive_trigger
        if requires_review:
            reason_codes.append("SENSITIVE_INTENT_MASS")
            if best_category:
                reason_codes.append(f"SENSITIVE_GROUP_{best_category.upper()}")
        return SensitiveCaseAssessment(
            requires_priority_review=requires_review,
            sensitive_intent=best_intent if best_score > 0.05 else None,
            sensitive_score=best_score,
            scope_detected=scope_detected,
            reason_codes=reason_codes,
            sensitive_probability_mass=sensitive_mass,
            sensitive_category=best_category,
            sensitive_group_mass=group_mass,
        )
