"""Integration tests for RoutingService, Artifact Integrity, and FastAPI Endpoints."""

import json
from pathlib import Path
from fastapi.testclient import TestClient
import pytest

from src.banking_router.api.app import app
from src.banking_router.data.contracts import BANKING77_77_CLASSES
from src.banking_router.data.normalization import compute_file_sha256
from src.banking_router.modeling.artifact import load_and_validate_bundle
from src.banking_router.routing.service import RoutingService

client = TestClient(app)


def test_model_classes_match_taxonomy():
    """INVARIANT: Model classes must match the 77-class taxonomy exactly."""
    bundle = load_and_validate_bundle("models", verify_checksum=False)
    classes = list(bundle.model.classes_)
    assert len(classes) == 77
    assert set(classes) == set(BANKING77_77_CLASSES)


def test_model_artifact_matches_manifest():
    """INVARIANT: Artifact binary SHA-256 must match the value recorded in model_manifest.json."""
    manifest = json.loads(Path("models/model_manifest.json").read_text(encoding="utf-8"))
    recorded_sha = manifest["artifact_sha256"]
    actual_sha = compute_file_sha256("models/router.joblib")
    assert recorded_sha == actual_sha


def test_batch_and_single_prediction_equivalent():
    """INVARIANT: Vectorized batch routing must return identical predictions to sequential single routing."""
    bundle = load_and_validate_bundle("models", verify_checksum=False)
    from src.banking_router.routing.policy import RoutingPolicy
    from src.banking_router.routing.risk import RiskAssessor
    from src.banking_router.routing.ood import OODGuard
    from src.banking_router.routing.taxonomy import TaxonomyResolver

    service = RoutingService(
        model=bundle.model,
        policy=RoutingPolicy(threshold=0.48, high_risk_trigger=0.40),
        risk_assessor=RiskAssessor(bundle.model.classes_, high_risk_trigger=0.40),
        ood_guard=OODGuard(),
        taxonomy=TaxonomyResolver(),
    )

    test_queries = [
        "Where is my card?",
        "Why was my cash withdrawal declined?",
        "I lost my phone and card",
        "How do I cancel my pending transfer?",
    ]

    single_results = [service.route(q) for q in test_queries]
    batch_results = service.route_batch(test_queries)

    assert len(single_results) == len(batch_results)
    for s, b in zip(single_results, batch_results):
        assert s.prediction.intent == b.prediction.intent
        assert s.prediction.domain == b.prediction.domain
        assert abs(s.prediction.confidence - b.prediction.confidence) < 1e-4
        assert s.risk.high_risk_detected == b.risk.high_risk_detected
        assert s.decision.action == b.decision.action
        assert s.decision.queue_id == b.decision.queue_id


def test_liveness_and_readiness_endpoints():
    """Verify FastAPI /health/live and /health/ready probes."""
    live_resp = client.get("/health/live")
    assert live_resp.status_code == 200
    assert live_resp.json() == {"status": "alive"}

    ready_resp = client.get("/health/ready")
    assert ready_resp.status_code == 200
    r_json = ready_resp.json()
    assert r_json["status"] == "ready"
    assert r_json["model_ready"] is True
    assert r_json["class_count"] == 77


def test_v1_route_endpoint():
    """Verify POST /v1/route canonical response structure."""
    resp = client.post(
        "/v1/route",
        json={"text": "Where is my card?", "top_k": 3},
    )
    assert resp.status_code == 200
    body = resp.json()

    assert "request_id" in body
    assert "prediction" in body
    assert "risk" in body
    assert "decision" in body
    assert body["prediction"]["intent"] in {"card_arrival", "card_acceptance", "card_delivery_estimate"}
    assert body["prediction"]["domain"] == "card_services"
    assert body["decision"]["action"] in {"auto_route", "human_review", "priority_human_review"}
    assert len(body["prediction"]["alternatives"]) == 3


def test_feedback_loop_records_event(tmp_path):
    """Verify POST /v1/feedback persists reviewer correction."""
    resp = client.post(
        "/v1/feedback",
        json={
            "request_id": "req_test_12345",
            "reviewed_intent": "card_arrival",
            "reviewer_id": "senior_reviewer_01",
            "notes": "Verified with customer call",
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "recorded"
    assert body["feedback"]["reviewed_intent"] == "card_arrival"
    assert body["feedback"]["reviewer_id"] == "senior_reviewer_01"
