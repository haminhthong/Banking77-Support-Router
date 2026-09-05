"""HTTP REST API phân luồng yêu cầu hỗ trợ khách hàng bằng FastAPI.

Cung cấp các endpoint:
- `/health`: Kiểm tra trạng thái sẵn sàng của dịch vụ và mô hình.
- `/predict`: Phân loại ý định, xác định miền nghiệp vụ và đưa ra quyết định routing cho 1 câu hỏi.
- `/predict/batch`: Phân loại hàng loạt với xử lý vector hóa (Vectorized Batch Inference).
- Telemetry & Logging an toàn thông tin (PII-Safe Logging).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .data import get_domain_for_intent
from .policy import RoutingPolicy
from .utils import LOGGER, calculate_entropy, redact_pii

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models/router.joblib"
CONFIG_PATH = PROJECT_ROOT / "models/config.json"

app = FastAPI(
    title="Banking77 Support Triage Platform API",
    description="Dịch vụ phân luồng ticket hỗ trợ khách hàng ngân hàng kèm cơ chế từ chối an toàn (Selective Classification) và ưu tiên rủi ro cao",
    version="2.0.0",
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
        max_length=100,
        description="Danh sách các câu hỏi cần xử lý phân luồng",
    )


def load_model() -> tuple[Any, dict[str, Any]]:
    """Lazy-load mô hình và file cấu hình một lần duy nhất vào bộ nhớ RAM."""
    global _model, _config
    if _model is None or _config is None:
        if not MODEL_PATH.exists() or not CONFIG_PATH.exists():
            raise FileNotFoundError(
                "Mô hình chưa được huấn luyện. Vui lòng chạy python -m src.train trước."
            )
        _model = joblib.load(MODEL_PATH)
        _config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    return _model, _config


@app.get("/health", summary="Kiểm tra sức khỏe dịch vụ (Health Check)")
def health() -> dict[str, Any]:
    """Kiểm tra trạng thái sẵn sàng của service và mô hình AI."""
    ready = MODEL_PATH.exists() and CONFIG_PATH.exists()
    version = "not_trained"
    policy_version = "not_trained"
    class_count = 0
    if CONFIG_PATH.exists():
        try:
            cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
            version = cfg.get("version", "unknown")
            policy_version = cfg.get("policy_version", "unknown")
            class_count = cfg.get("class_count", 0)
        except (OSError, json.JSONDecodeError):
            ready = False

    return {
        "status": "ok" if ready else "degraded",
        "model_ready": ready,
        "model_version": version,
        "policy_version": policy_version,
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


def _build_policy(config: dict[str, Any]) -> RoutingPolicy:
    """Khởi tạo RoutingPolicy từ file cấu hình."""
    return RoutingPolicy(
        threshold=float(config["threshold"]),
        high_risk_trigger=float(config.get("high_risk_trigger", 0.20)),
        min_margin=config.get("min_margin"),
        max_entropy=config.get("max_entropy"),
    )


def _format_prediction_item(
    probabilities: np.ndarray,
    model: Any,
    config: dict[str, Any],
    top_k: int,
    policy: RoutingPolicy,
    raw_query_text: str,
) -> dict[str, Any]:
    """Bổ trợ tính toán metrics độ mập mờ, áp dụng policy với raw probability và format JSON."""
    ranked_indices = probabilities.argsort()[::-1]
    domain_map = config.get("domain_map", {})

    raw_confidence = float(probabilities[ranked_indices[0]])
    raw_margin = (
        float(probabilities[ranked_indices[0]] - probabilities[ranked_indices[1]])
        if len(ranked_indices) > 1
        else 1.0
    )
    entropy = float(calculate_entropy(probabilities))

    top_intent = str(model.classes_[ranked_indices[0]])
    top_domain = domain_map.get(top_intent, get_domain_for_intent(top_intent))

    # Xây dựng danh sách top_k candidates cho policy
    top_k_candidates = [
        (str(model.classes_[idx]), float(probabilities[idx]))
        for idx in ranked_indices[:top_k]
    ]

    # Quyết định Routing Policy trên raw confidence chưa làm tròn
    decision = policy.decide(
        top_intent=top_intent,
        confidence=raw_confidence,
        top_domain=top_domain,
        top_k_candidates=top_k_candidates,
        margin=raw_margin,
        entropy=entropy,
    )

    # Telemetry logging an toàn thông tin (PII-safe)
    LOGGER.info(
        "Triage: text='%s' | route='%s' | decision='%s' | conf=%.4f | margin=%.4f",
        redact_pii(raw_query_text),
        decision.route,
        decision.decision,
        raw_confidence,
        raw_margin,
    )

    # Format alternatives (làm tròn hiển thị)
    alternatives = []
    for idx in ranked_indices[:top_k]:
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

    return {
        "decision": decision.decision,
        "abstained": decision.abstained,
        "is_unknown": decision.is_unknown,  # Tương thích ngược
        "intent": decision.intent,
        "domain": decision.domain,
        "top_intent": top_intent,
        "confidence": round(raw_confidence, 4),
        "margin": round(raw_margin, 4),
        "entropy": round(entropy, 4),
        "alternatives": alternatives,
        "route": decision.route,
        "requires_human_review": decision.requires_human_review,
        "review_reason": decision.review_reason,
        "model_version": config.get("version", "v2"),
        "policy_version": config.get("policy_version", "v2"),
    }


@app.post("/predict", summary="Dự đoán ý định và phân luồng ticket")
def predict(query: Query) -> dict[str, Any]:
    """Phân loại ý định, xác định miền nghiệp vụ và ra quyết định routing cho 1 ticket."""
    model, config = _get_ready_model()
    policy = _build_policy(config)
    probabilities = model.predict_proba([query.text])[0]
    return _format_prediction_item(
        probabilities=probabilities,
        model=model,
        config=config,
        top_k=query.top_k,
        policy=policy,
        raw_query_text=query.text,
    )


@app.post("/predict/batch", summary="Dự đoán và phân luồng hàng loạt ticket (Vectorized Batch)")
def predict_batch(batch: BatchQuery) -> dict[str, Any]:
    """Xử lý phân loại và routing hàng loạt thông qua một lần vectorize và predict_proba duy nhất."""
    model, config = _get_ready_model()
    policy = _build_policy(config)

    texts = [q.text for q in batch.queries]
    # Thực hiện batch inference một lần duy nhất
    proba_matrix = model.predict_proba(texts)

    results = [
        _format_prediction_item(
            probabilities=proba_matrix[i],
            model=model,
            config=config,
            top_k=batch.queries[i].top_k,
            policy=policy,
            raw_query_text=texts[i],
        )
        for i in range(len(texts))
    ]

    return {
        "total_queries": len(results),
        "results": results,
    }
