"""Điều phối pipeline inference của support-ticket router."""

from __future__ import annotations

import time
import uuid
from typing import Any, Sequence

import numpy as np

from ..data.normalization import normalize_pii_semantically
from .escalation import SensitiveIntentGuard
from .policy import RoutingPolicy
from .queue_projector import QueueProjector
from .schemas import IntentPrediction, RoutingResult
from .scope import ScopeGuard
from .taxonomy import TaxonomyResolver


class RoutingService:
    """Chạy normalize -> model -> kiểm tra queue/scope/nhạy cảm -> policy."""

    def __init__(
        self,
        model: Any,
        policy: RoutingPolicy,
        sensitive_guard: SensitiveIntentGuard,
        scope_guard: ScopeGuard | None = None,
        scope_model: Any | None = None,
        taxonomy: TaxonomyResolver | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.model = model
        self.policy = policy
        self.sensitive_guard = sensitive_guard
        self.scope_guard = scope_guard or ScopeGuard()
        self.scope_guard.set_scope_classifier(scope_model)
        self.taxonomy = taxonomy or TaxonomyResolver()
        self.metadata = metadata or {}
        self.queue_projector = QueueProjector(self.model.classes_, self.taxonomy)
        self._load_vocabulary()

    def _load_vocabulary(self) -> None:
        model = self.model
        if hasattr(model, "estimator"):
            model = model.estimator
        elif hasattr(model, "calibrated_classifiers_") and model.calibrated_classifiers_:
            model = model.calibrated_classifiers_[0].estimator
        if hasattr(model, "estimator"):
            model = model.estimator
        if not hasattr(model, "named_steps"):
            return
        features = model.named_steps.get("features")
        if features is not None and hasattr(features, "transformer_list"):
            for name, transformer in features.transformer_list:
                if name == "word_tfidf" and hasattr(transformer, "vocabulary_"):
                    self.scope_guard.set_vocabulary(set(transformer.vocabulary_))
                    return
        tfidf = model.named_steps.get("tfidf")
        if tfidf is not None and hasattr(tfidf, "vocabulary_"):
            self.scope_guard.set_vocabulary(set(tfidf.vocabulary_))

    @staticmethod
    def _entropy(probabilities: np.ndarray, eps: float = 1e-12) -> float:
        values = np.clip(probabilities, eps, 1.0)
        return float(-np.sum(values * np.log(values)))

    def _route_one(self, text: str, probabilities: np.ndarray, top_k: int, request_id: str) -> RoutingResult:
        classes = self.model.classes_
        ranked_indices = probabilities.argsort()[::-1]
        top_intent = str(classes[ranked_indices[0]])
        confidence = float(probabilities[ranked_indices[0]])
        margin = float(probabilities[ranked_indices[0]] - probabilities[ranked_indices[1]]) if len(ranked_indices) > 1 else 1.0
        entropy = self._entropy(probabilities)
        queue_prediction = self.queue_projector.project(probabilities)
        scope_detected, scope_reasons = self.scope_guard.detect(text, confidence, margin)
        sensitive_case = self.sensitive_guard.assess(probabilities, scope_detected, scope_reasons)
        safe_top_k = max(1, min(int(top_k), len(classes)))
        alternatives = [
            {
                "intent": str(classes[index]),
                "domain": self.taxonomy.get_domain(str(classes[index])),
                "confidence": round(float(probabilities[index]), 4),
            }
            for index in ranked_indices[:safe_top_k]
        ]
        prediction = IntentPrediction(
            intent=top_intent,
            domain=self.taxonomy.get_domain(top_intent),
            confidence=round(confidence, 4),
            margin=round(margin, 4),
            entropy=round(entropy, 4),
            alternatives=alternatives,
        )
        decision = self.policy.evaluate(
            prediction,
            sensitive_case,
            scope_detected=scope_detected,
            scope_reasons=scope_reasons,
            queue_prediction=queue_prediction,
        )
        return RoutingResult(
            request_id=request_id,
            prediction=prediction,
            queue_prediction=queue_prediction,
            sensitive_case=sensitive_case,
            scope_detected=scope_detected,
            scope_reasons=scope_reasons,
            decision=decision,
            metadata=dict(self.metadata),
        )

    def route(self, text: str, top_k: int = 3, request_id: str | None = None) -> RoutingResult:
        """Route một ticket sau khi áp dụng cùng chuẩn hóa như lúc train."""
        started = time.perf_counter()
        normalized = normalize_pii_semantically(text)
        probabilities = self.model.predict_proba([normalized])[0]
        result = self._route_one(normalized, probabilities, top_k, request_id or f"req_{uuid.uuid4().hex[:12]}")
        result.metadata["latency_ms"] = round((time.perf_counter() - started) * 1000, 3)
        return result

    def route_batch(self, texts: list[str], top_k: int = 3, top_ks: Sequence[int] | None = None) -> list[RoutingResult]:
        """Route nhiều ticket bằng một lần gọi predict_proba."""
        if not texts:
            return []
        if top_ks is not None and len(top_ks) != len(texts):
            raise ValueError("top_ks phải có cùng số phần tử với texts")
        normalized = [normalize_pii_semantically(text) for text in texts]
        probabilities = self.model.predict_proba(normalized)
        return [
            self._route_one(
                text,
                probabilities[index],
                top_ks[index] if top_ks is not None else top_k,
                f"req_{uuid.uuid4().hex[:12]}",
            )
            for index, text in enumerate(normalized)
        ]
