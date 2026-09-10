"""Cộng xác suất intent chi tiết thành các queue nghiệp vụ."""

from __future__ import annotations

from typing import Sequence
import numpy as np

from .schemas import QueuePrediction
from .taxonomy import TaxonomyResolver


class QueueProjector:
    """Cộng xác suất của các intent loại trừ nhau theo từng queue."""

    def __init__(self, classes: Sequence[str], taxonomy: TaxonomyResolver) -> None:
        self.classes = tuple(str(item) for item in classes)
        self.taxonomy = taxonomy

    def project(self, probabilities: Sequence[float]) -> QueuePrediction:
        values = np.asarray(probabilities, dtype=float)
        if values.ndim != 1 or len(values) != len(self.classes):
            raise ValueError(
                f"Expected one probability per class ({len(self.classes)}), got {values.shape}"
            )
        total = float(values.sum())
        if total <= 0.0:
            raise ValueError("Intent probabilities must contain positive mass")
        values = values / total

        queue_probabilities: dict[str, float] = {}
        for intent, probability in zip(self.classes, values):
            queue = self.taxonomy.get_queue(intent)
            queue_probabilities[queue] = queue_probabilities.get(queue, 0.0) + float(probability)

        ranked = sorted(queue_probabilities.items(), key=lambda item: item[1], reverse=True)
        top_queue, top_probability = ranked[0]
        second_probability = ranked[1][1] if len(ranked) > 1 else 0.0
        return QueuePrediction(
            queue=top_queue,
            confidence=round(float(top_probability), 6),
            margin=round(float(top_probability - second_probability), 6),
            probabilities={key: round(float(value), 6) for key, value in queue_probabilities.items()},
        )
