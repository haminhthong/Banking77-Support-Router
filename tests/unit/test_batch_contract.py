"""Kiểm tra contract batch: top_k chỉ thuộc về từng ticket."""

from __future__ import annotations

import numpy as np

from src.banking_router.data.contracts import BANKING77_77_CLASSES
from src.banking_router.routing.ood import OODGuard
from src.banking_router.routing.policy import RoutingPolicy
from src.banking_router.routing.risk import RiskAssessor
from src.banking_router.routing.service import RoutingService
from src.banking_router.routing.taxonomy import TaxonomyResolver


class FakeModel:
    """Model nhỏ, deterministic để kiểm tra orchestration mà không train lại."""

    classes_ = np.asarray(BANKING77_77_CLASSES)

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        probabilities = np.zeros((len(texts), len(self.classes_)))
        probabilities[:, :5] = [0.40, 0.25, 0.15, 0.10, 0.10]
        return probabilities


def test_route_batch_preserves_each_ticket_top_k() -> None:
    """Mỗi ticket trong batch phải có số alternative đúng theo top_k riêng."""
    taxonomy = TaxonomyResolver()
    model = FakeModel()
    service = RoutingService(
        model=model,
        policy=RoutingPolicy(taxonomy_resolver=taxonomy),
        risk_assessor=RiskAssessor(classes=model.classes_, taxonomy=taxonomy),
        ood_guard=OODGuard(),
        taxonomy=taxonomy,
    )

    results = service.route_batch(
        ["Where is my card?", "How do I activate my card?"],
        top_ks=[1, 5],
    )

    assert len(results) == 2
    assert len(results[0].prediction.alternatives) == 1
    assert len(results[1].prediction.alternatives) == 5
