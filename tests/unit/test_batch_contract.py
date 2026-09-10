"""Kiểm thử hợp đồng batch: top_k được áp dụng riêng cho từng ticket."""

import numpy as np

from src.banking_router.data.contracts import BANKING77_77_CLASSES
from src.banking_router.routing.escalation import SensitiveIntentGuard
from src.banking_router.routing.policy import RoutingPolicy
from src.banking_router.routing.scope import ScopeGuard
from src.banking_router.routing.service import RoutingService
from src.banking_router.routing.taxonomy import TaxonomyResolver


class FakeModel:
    classes_ = np.asarray(BANKING77_77_CLASSES)

    def predict_proba(self, texts):
        probabilities = np.zeros((len(texts), len(self.classes_)))
        probabilities[:, :5] = [0.40, 0.25, 0.15, 0.10, 0.10]
        return probabilities


def test_route_batch_preserves_each_ticket_top_k():
    taxonomy = TaxonomyResolver()
    model = FakeModel()
    service = RoutingService(
        model=model,
        policy=RoutingPolicy(taxonomy_resolver=taxonomy),
        sensitive_guard=SensitiveIntentGuard(model.classes_, taxonomy=taxonomy),
        scope_guard=ScopeGuard(),
        taxonomy=taxonomy,
    )
    results = service.route_batch(["Where is my card?", "How do I activate my card?"], top_ks=[1, 5])
    assert len(results) == 2
    assert len(results[0].prediction.alternatives) == 1
    assert len(results[1].prediction.alternatives) == 5
