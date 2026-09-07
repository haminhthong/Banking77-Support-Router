"""HTTP REST API phân luồng yêu cầu hỗ trợ khách hàng bằng FastAPI.

Cung cấp facade tương thích ngược cho toàn bộ API và các unit test cũ,
kết nối trực tiếp với package banking_router.api.
"""

from __future__ import annotations

from typing import Any
import numpy as np

from src.banking_router.api.app import app, get_routing_service
from src.banking_router.api.schemas import (
    BatchRouteRequest,
    FeedbackRequest,
    RouteRequest,
    RouteResponse,
)
from src.banking_router.data.contracts import get_domain_for_intent
from src.banking_router.routing.policy import RoutingPolicy
from src.banking_router.utils import calculate_entropy

# Legacy globals for monkeypatching support in test_smoke.py
_model: Any | None = None
_config: dict[str, Any] | None = None

Query = RouteRequest


class BatchQuery:
    def __init__(self, queries: list[Any]) -> None:
        self.queries = queries


def predict(query: Query) -> dict[str, Any]:
    """Legacy predict function supporting both test monkeypatch and production service."""
    global _model, _config
    if _model is not None:
        probabilities = _model.predict_proba([query.text])[0]
        ranked_indices = probabilities.argsort()[::-1]
        domain_map = _config.get("domain_map", {}) if _config else {}

        raw_confidence = float(probabilities[ranked_indices[0]])
        raw_margin = (
            float(probabilities[ranked_indices[0]] - probabilities[ranked_indices[1]])
            if len(ranked_indices) > 1
            else 1.0
        )
        entropy = float(calculate_entropy(probabilities))
        top_intent = str(_model.classes_[ranked_indices[0]])
        top_domain = domain_map.get(top_intent, get_domain_for_intent(top_intent))

        top_k_candidates = [
            (
                str(_model.classes_[idx]),
                domain_map.get(
                    str(_model.classes_[idx]),
                    get_domain_for_intent(str(_model.classes_[idx])),
                ),
                float(probabilities[idx]),
            )
            for idx in ranked_indices[:query.top_k]
        ]

        threshold = float(_config.get("threshold", 0.45)) if _config else 0.45
        high_risk_trigger = (
            float(_config.get("high_risk_trigger", 0.20)) if _config else 0.20
        )
        policy = RoutingPolicy(
            threshold=threshold,
            high_risk_trigger=high_risk_trigger,
            min_margin=_config.get("min_margin") if _config else None,
            max_entropy=_config.get("max_entropy") if _config else None,
        )

        decision = policy.decide(
            top_intent=top_intent,
            confidence=raw_confidence,
            top_domain=top_domain,
            top_k_candidates=top_k_candidates,
            margin=raw_margin,
            entropy=entropy,
        )

        alternatives = [
            {
                "intent": str(_model.classes_[idx]),
                "domain": domain_map.get(
                    str(_model.classes_[idx]),
                    get_domain_for_intent(str(_model.classes_[idx])),
                ),
                "confidence": round(float(probabilities[idx]), 4),
            }
            for idx in ranked_indices[:query.top_k]
        ]

        return {
            "decision": decision.decision,
            "abstained": decision.abstained,
            "is_unknown": decision.is_unknown,
            "intent": decision.intent,
            "domain": decision.domain,
            "top_intent": top_intent,
            "top_domain": top_domain,
            "confidence": round(raw_confidence, 4),
            "margin": round(raw_margin, 4),
            "entropy": round(entropy, 4),
            "alternatives": alternatives,
            "route": decision.route,
            "requires_human_review": decision.requires_human_review,
            "review_reason": decision.review_reason,
            "model_version": _config.get("version", "v2") if _config else "v2",
            "policy_version": _config.get("policy_version", "v2") if _config else "v2",
        }

    # Production service delegation
    service = get_routing_service()
    res = service.route(query.text, top_k=query.top_k)
    return {
        "decision": res.decision.decision,
        "abstained": res.decision.abstained,
        "is_unknown": res.decision.is_unknown,
        "intent": res.decision.intent,
        "domain": res.decision.domain,
        "top_intent": res.prediction.intent,
        "top_domain": res.prediction.domain,
        "confidence": res.prediction.confidence,
        "margin": res.prediction.margin,
        "entropy": res.prediction.entropy,
        "alternatives": res.prediction.alternatives,
        "route": res.decision.route,
        "requires_human_review": res.decision.requires_human_review,
        "review_reason": res.decision.review_reason,
        "model_version": res.metadata.get("model_version", "v3"),
        "policy_version": res.metadata.get("policy_version", "v3"),
    }


def predict_batch(batch: BatchQuery) -> dict[str, Any]:
    """Legacy predict_batch function supporting both test monkeypatch and production service."""
    global _model, _config
    if _model is not None:
        texts = [q.text for q in batch.queries]
        proba_matrix = _model.predict_proba(texts)
        results = []
        for i, q in enumerate(batch.queries):
            prob = proba_matrix[i]
            ranked_indices = prob.argsort()[::-1]
            domain_map = _config.get("domain_map", {}) if _config else {}
            raw_conf = float(prob[ranked_indices[0]])
            top_intent = str(_model.classes_[ranked_indices[0]])
            top_domain = domain_map.get(top_intent, get_domain_for_intent(top_intent))
            threshold = float(_config.get("threshold", 0.45)) if _config else 0.45
            high_risk_trigger = (
                float(_config.get("high_risk_trigger", 0.20)) if _config else 0.20
            )
            policy = RoutingPolicy(
                threshold=threshold,
                high_risk_trigger=high_risk_trigger,
            )
            decision = policy.decide(top_intent=top_intent, confidence=raw_conf, top_domain=top_domain)
            results.append({
                "decision": decision.decision,
                "confidence": round(raw_conf, 4),
                "top_intent": top_intent,
                "domain": top_domain,
            })
        decision_counts: dict[str, int] = {}
        for r in results:
            d = str(r["decision"])
            decision_counts[d] = decision_counts.get(d, 0) + 1
        return {
            "total_queries": len(results),
            "decision_counts": decision_counts,
            "results": results,
        }

    service = get_routing_service()
    texts = [q.text for q in batch.queries]
    res_list = service.route_batch(texts, top_k=batch.queries[0].top_k if batch.queries else 3)
    results = [
        {
            "decision": r.decision.decision,
            "confidence": r.prediction.confidence,
            "top_intent": r.prediction.intent,
            "domain": r.prediction.domain,
        }
        for r in res_list
    ]
    decision_counts = {}
    for r in results:
        d = str(r["decision"])
        decision_counts[d] = decision_counts.get(d, 0) + 1
    return {
        "total_queries": len(results),
        "decision_counts": decision_counts,
        "results": results,
    }


__all__ = [
    "app",
    "get_routing_service",
    "RouteRequest",
    "RouteResponse",
    "BatchRouteRequest",
    "FeedbackRequest",
    "Query",
    "BatchQuery",
    "predict",
    "predict_batch",
    "_model",
    "_config",
]
