"""Integration test cho artifact canonical, service và FastAPI."""

import numpy as np
from fastapi.testclient import TestClient

from src.banking_router.api.app import app
from src.banking_router.config import ARTIFACTS_DIR
from src.banking_router.data.contracts import BANKING77_77_CLASSES
from src.banking_router.modeling.artifact import load_artifacts
from src.banking_router.routing.escalation import SensitiveIntentGuard
from src.banking_router.routing.policy import RoutingPolicy
from src.banking_router.routing.scope import ScopeGuard
from src.banking_router.routing.service import RoutingService
from src.banking_router.routing.taxonomy import TaxonomyResolver

client = TestClient(app)


def test_model_classes_match_taxonomy():
    artifacts = load_artifacts(ARTIFACTS_DIR)
    assert len(artifacts.intent_model.classes_) == 77
    assert set(artifacts.intent_model.classes_) == set(BANKING77_77_CLASSES)
    assert artifacts.metadata["dataset"] == "Banking77"


def test_canonical_artifact_layout():
    assert all((ARTIFACTS_DIR / name).exists() for name in (
        "intent_model.joblib", "metadata.json", "taxonomy.json", "routing_policy.json"
    ))
    assert not (ARTIFACTS_DIR / "manifest.json").exists()
    assert not (ARTIFACTS_DIR / "model_config.json").exists()


def test_batch_and_single_prediction_equivalent():
    artifacts = load_artifacts(ARTIFACTS_DIR)
    taxonomy = TaxonomyResolver(artifacts.taxonomy)
    service = RoutingService(
        model=artifacts.intent_model,
        policy=RoutingPolicy(queue_threshold=0.48, sensitive_trigger=0.40, taxonomy_resolver=taxonomy),
        sensitive_guard=SensitiveIntentGuard(artifacts.intent_model.classes_, taxonomy=taxonomy, sensitive_trigger=0.40),
        scope_guard=ScopeGuard(),
        scope_model=artifacts.scope_model,
        taxonomy=taxonomy,
    )
    queries = ["Where is my card?", "Why was my cash withdrawal declined?", "I lost my phone and card"]
    singles = [service.route(query) for query in queries]
    batches = service.route_batch(queries)
    for single, batch in zip(singles, batches):
        assert single.prediction.intent == batch.prediction.intent
        assert abs(single.prediction.confidence - batch.prediction.confidence) < 1e-4
        assert single.sensitive_case.requires_priority_review == batch.sensitive_case.requires_priority_review
        assert single.decision.action == batch.decision.action


def test_liveness_and_readiness_endpoints():
    assert client.get("/health/live").json() == {"status": "alive"}
    response = client.get("/health/ready")
    assert response.status_code == 200
    assert response.json()["class_count"] == 77


def test_v1_route_endpoint():
    response = client.post("/v1/route", json={"text": "Where is my card?", "top_k": 3})
    assert response.status_code == 200
    body = response.json()
    assert "prediction" in body
    assert "sensitive_case" in body
    assert "scope" in body
    assert "decision" in body
    assert body["prediction"]["domain"] == "card_services"
    assert body["decision"]["action"] in {"auto_route", "human_review", "priority_human_review"}
    assert len(body["prediction"]["alternatives"]) == 3
