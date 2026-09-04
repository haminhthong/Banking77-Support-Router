from fastapi.testclient import TestClient
from src.api import app

client = TestClient(app)


def test_api():
    # 1. Test Health
    h_res = client.get("/health")
    print("HEALTH RESPONSE:", h_res.json())
    assert h_res.status_code == 200
    assert h_res.json()["model_ready"] is True

    # 2. Test Single Predict
    p_res = client.post(
        "/predict",
        json={"text": "Why has my cash withdrawal been declined?", "top_k": 3},
    )
    print("PREDICT RESPONSE:", p_res.json())
    assert p_res.status_code == 200
    res_data = p_res.json()
    assert (
        res_data["top_intent"] == "pending_cash_withdrawal"
        or "cash" in res_data["top_intent"]
    )
    assert res_data["alternatives"][0]["domain"] == "atm_cash"

    # 3. Test Batch Predict
    b_res = client.post(
        "/predict/batch",
        json={
            "queries": [
                {"text": "Where is my card?"},
                {"text": "I lost my phone and card, help!"},
            ]
        },
    )
    print("BATCH PREDICT RESPONSE:", b_res.json())
    assert b_res.status_code == 200
    assert b_res.json()["total_queries"] == 2


if __name__ == "__main__":
    test_api()
    print("All API Client checks passed successfully!")
