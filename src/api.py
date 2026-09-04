"""HTTP REST API phân luồng yêu cầu hỗ trợ khách hàng bằng FastAPI.

Cung cấp các endpoint:
- `/health`: Kiểm tra trạng thái sẵn sàng của dịch vụ và mô hình.
- `/predict`: Phân loại ý định, miền nghiệp vụ và đưa ra quyết định routing cho 1 câu hỏi.
- `/predict/batch`: Phân loại hàng loạt danh sách nhiều câu hỏi cùng lúc.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .data import get_domain_for_intent
from .policy import RoutingPolicy

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models/router.joblib"
CONFIG_PATH = PROJECT_ROOT / "models/config.json"

app = FastAPI(
    title="AI Customer Support Router (BANKING77)",
    description="Dịch vụ phân luồng ticket hỗ trợ khách hàng ngân hàng kèm cơ chế từ chối an toàn (Selective Classification)",
    version="1.1.0",
)

_model: Any | None = None
_config: dict[str, Any] | None = None


class Query(BaseModel):
    """Schema dữ liệu đầu vào cho một câu hỏi khách hàng."""

    text: str = Field(
        ...,
        min_length=2,
        max_length=1000,
        description="Nội dung thắc mắc hoặc yêu cầu hỗ trợ từ khách hàng",
        examples=["Why has my cash withdrawal been declined?"],
    )
    top_k: int = Field(
        default=3,
        ge=1,
        le=5,
        description="Số lượng dự đoán thay thế top_k trả về",
    )


class BatchQuery(BaseModel):
    """Schema dữ liệu đầu vào cho xử lý hàng loạt nhiều câu hỏi."""

    queries: list[Query] = Field(
        ...,
        min_length=1,
        max_length=50,
        description="Danh sách các câu hỏi cần xử lý phân luồng",
    )


def load_model() -> tuple[Any, dict[str, Any]]:
    """Lazy-load mô hình và file cấu hình một lần duy nhất vào bộ nhớ RAM."""
    global _model, _config
    if _model is None or _config is None:
        if not MODEL_PATH.exists() or not CONFIG_PATH.exists():
            raise FileNotFoundError(
                "Mô hình chưa được huấn luyện. Vui lòng chạy train.py trước."
            )
        _model = joblib.load(MODEL_PATH)
        _config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    return _model, _config


@app.get("/health", summary="Kiểm tra sức khỏe dịch vụ (Health Check)")
def health() -> dict[str, Any]:
    """Kiểm tra trạng thái sẵn sàng của service và mô hình AI."""
    ready = MODEL_PATH.exists() and CONFIG_PATH.exists()
    version = "not_trained"
    class_count = 0
    if CONFIG_PATH.exists():
        try:
            cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
            version = cfg.get("version", "unknown")
            class_count = cfg.get("class_count", 0)
        except (OSError, json.JSONDecodeError):
            ready = False

    return {
        "status": "ok" if ready else "degraded",
        "model_ready": ready,
        "model_version": version,
        "class_count": class_count,
    }


def _get_ready_model() -> tuple[Any, dict[str, Any]]:
    """Tải mô hình và chuyển đổi ngoại lệ thành HTTP 503 nếu dịch vụ chưa sẵn sàng."""
    try:
        return load_model()
    except (OSError, ValueError, KeyError, FileNotFoundError) as exc:
        raise HTTPException(
            status_code=503,
            detail="Mô hình chưa sẵn sàng hoặc file cấu hình không hợp lệ",
        ) from exc


def _process_single_query(
    query: Query, model: Any, config: dict[str, Any]
) -> dict[str, Any]:
    """Hàm bổ trợ xử lý phân loại và routing cho 1 query."""
    probabilities = model.predict_proba([query.text])[0]
    ranked_indices = probabilities.argsort()[::-1][: query.top_k]

    domain_map = config.get("domain_map", {})

    alternatives = []
    for idx in ranked_indices:
        intent_name = str(model.classes_[idx])
        alternatives.append(
            {
                "intent": intent_name,
                "domain": domain_map.get(
                    intent_name, get_domain_for_intent(intent_name)
                ),
                "confidence": round(float(probabilities[idx]), 4),
            }
        )

    top_intent = alternatives[0]["intent"]
    top_domain = alternatives[0]["domain"]
    confidence = alternatives[0]["confidence"]

    policy = RoutingPolicy(threshold=float(config["threshold"]))
    decision = policy.decide(
        top_intent=top_intent, confidence=confidence, top_domain=top_domain
    )

    return {
        "intent": decision.intent,
        "domain": decision.domain,
        "top_intent": top_intent,
        "confidence": confidence,
        "alternatives": alternatives,
        "is_unknown": decision.is_unknown,
        "route": decision.route,
        "requires_human_review": decision.requires_human_review,
        "review_reason": decision.review_reason,
        "model_version": config.get("version", "v1"),
    }


@app.post("/predict", summary="Dự đoán ý định và phân luồng ticket")
def predict(query: Query) -> dict[str, Any]:
    """Phân loại ý định, xác định miền nghiệp vụ và ra quyết định routing cho 1 ticket."""
    model, config = _get_ready_model()
    return _process_single_query(query, model, config)


@app.post("/predict/batch", summary="Dự đoán và phân luồng hàng loạt ticket")
def predict_batch(batch: BatchQuery) -> dict[str, Any]:
    """Xử lý phân loại và routing hàng loạt cho danh sách nhiều ticket cùng lúc."""
    model, config = _get_ready_model()
    results = [_process_single_query(q, model, config) for q in batch.queries]
    return {
        "total_queries": len(results),
        "results": results,
    }
