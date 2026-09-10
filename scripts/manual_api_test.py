import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from fastapi.testclient import TestClient

from src.banking_router.api.app import app

client = TestClient(app)


def test_api():
    print("=== 1. KIỂM THỬ HEALTH PROBES ===")
    live_res = client.get("/health/live")
    print("GET /health/live:", live_res.json())
    assert live_res.status_code == 200

    ready_res = client.get("/health/ready")
    print("GET /health/ready:", ready_res.json())
    assert ready_res.status_code == 200
    assert ready_res.json()["model_ready"] is True
    assert ready_res.json()["class_count"] == 77

    print("\n=== 2. KIỂM THỬ CANONICAL ROUTING (/v1/route) ===")
    test_cases = [
        ("Where is my card?", 3),
        (
            "I lost my phone and card, help me immediately!",
            3,
        ),
        (
            "Why was my cash withdrawal declined?",
            3,
        ),
        (
            "How do I apply for a 30-year fixed home mortgage loan?",
            3,
        ),
        ("asdfghjklqwerty", 3),
    ]

    first_request_id = None
    for text, top_k in test_cases:
        p_res = client.post(
            "/v1/route",
            json={"text": text, "top_k": top_k},
        )
        assert p_res.status_code == 200
        data = p_res.json()
        first_request_id = first_request_id or data["request_id"]
        print(f"\nQuery: '{text}'")
        print(
            f"  Action: {data['decision']['action']} | Queue: {data['decision']['queue']} | Priority: {data['decision']['priority']}"
        )
        print(
            f"  Predicted Intent: {data['prediction']['intent']} ({data['prediction']['domain']}) | Confidence: {data['prediction']['confidence']}"
        )
        print(
            f"  Signals: sensitive={data['sensitive_case']['requires_priority_review']} | scope={not data['scope']['supported']}"
        )

    print("\n=== 3. KIỂM THỬ BATCH ROUTING (/v1/route/batch) ===")
    b_res = client.post(
        "/v1/route/batch",
        json={
            "tickets": [
                {"text": "Where is my card?"},
                {"text": "I lost my phone and card, help!"},
                {"text": "What is the exchange rate for EUR to USD?"},
            ]
        },
    )
    assert b_res.status_code == 200
    b_data = b_res.json()
    print("Batch total:", b_data["total_tickets"])
    print("Batch decision counts:", b_data["decision_counts"])

    print("\n=== 4. KIỂM THỬ FEEDBACK LOOP (/v1/feedback) ===")
    fb_res = client.post(
        "/v1/feedback",
        json={
            "request_id": first_request_id,
            "reviewed_intent": "card_arrival",
            "reviewer_id": "senior_agent_07",
            "notes": "Verified through manual customer phone confirmation",
        },
    )
    assert fb_res.status_code == 200
    print("Feedback submission:", fb_res.json())

    print("\nAll manual API checks passed successfully!")


if __name__ == "__main__":
    test_api()
