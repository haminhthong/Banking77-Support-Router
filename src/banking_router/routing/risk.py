"""Security Risk Scanner inspecting high-risk threats independently of top-k presentation."""

from __future__ import annotations

from typing import Any, Sequence
import numpy as np
from .schemas import RiskAssessment
from ..data.contracts import DEFAULT_HIGH_RISK_INTENTS


class RiskAssessor:
    """Evaluates security risk across all 77 class probabilities.

    INVARIANT:
    Safety scanning ALWAYS inspects all configured high-risk intents
    across the entire probability distribution, completely independent
    of the display `top_k` requested by the client.
    """

    def __init__(
        self,
        classes: Sequence[str],
        high_risk_intents: frozenset[str] | set[str] | None = None,
        high_risk_trigger: float = 0.20,
    ) -> None:
        self.classes = np.array(classes)
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        self.high_risk_intents = frozenset(high_risk_intents or DEFAULT_HIGH_RISK_INTENTS)
        self.high_risk_trigger = float(high_risk_trigger)

        # Precompute indices for high-risk classes present in the model
        self.risk_indices = [
            self.class_to_idx[intent]
            for intent in self.high_risk_intents
            if intent in self.class_to_idx
        ]

    def assess(
        self,
        probabilities: np.ndarray,
        top_intent: str,
        ood_detected: bool = False,
        ood_reasons: list[str] | None = None,
    ) -> RiskAssessment:
        """Scan the full probability distribution for security threats and combine with OOD signals."""
        reason_codes: list[str] = []
        if ood_reasons:
            reason_codes.extend(ood_reasons)

        # 1. Check if the top predicted intent is an explicit high-risk class
        if top_intent in self.high_risk_intents:
            top_idx = self.class_to_idx.get(top_intent)
            top_score = float(probabilities[top_idx]) if top_idx is not None else 0.0
            reason_codes.append("HIGH_RISK_INTENT")
            return RiskAssessment(
                high_risk_detected=True,
                high_risk_intent=top_intent,
                high_risk_score=top_score,
                ood_detected=ood_detected,
                reason_codes=reason_codes,
            )

        # 2. Inspect all high-risk classes across the full probability array
        if self.risk_indices:
            risk_probs = probabilities[self.risk_indices]
            argmax_pos = int(np.argmax(risk_probs))
            max_risk_idx = self.risk_indices[argmax_pos]
            max_risk_score = float(probabilities[max_risk_idx])
            max_risk_intent = str(self.classes[max_risk_idx])

            if max_risk_score >= self.high_risk_trigger:
                reason_codes.append("HIGH_RISK_CANDIDATE")
                return RiskAssessment(
                    high_risk_detected=True,
                    high_risk_intent=max_risk_intent,
                    high_risk_score=max_risk_score,
                    ood_detected=ood_detected,
                    reason_codes=reason_codes,
                )
        else:
            max_risk_score = 0.0
            max_risk_intent = None

        return RiskAssessment(
            high_risk_detected=False,
            high_risk_intent=max_risk_intent if max_risk_score > 0.05 else None,
            high_risk_score=max_risk_score,
            ood_detected=ood_detected,
            reason_codes=reason_codes,
        )
