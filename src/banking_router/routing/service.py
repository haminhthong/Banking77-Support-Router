"""Routing Service orchestrating Intent Classifier, Risk Assessor, OOD Guard, and Policy Engine."""

from __future__ import annotations

import time
import uuid
from typing import Any, Sequence
import numpy as np

from .ood import OODGuard
from .policy import RoutingPolicy
from .queue_projector import QueueProjector
from .risk import RiskAssessor
from .schemas import IntentPrediction, RiskAssessment, RoutingDecision, RoutingResult
from .taxonomy import TaxonomyResolver
from ..data.normalization import normalize_pii_semantically
from ..telemetry.privacy import redact_pii


class RoutingService:
    """Canonical service orchestrator for the Banking Support Triage System."""

    def __init__(
        self,
        model: Any,
        policy: RoutingPolicy,
        risk_assessor: RiskAssessor,
        ood_guard: OODGuard | None = None,
        scope_model: Any | None = None,
        taxonomy: TaxonomyResolver | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.model = model
        self.policy = policy
        self.risk_assessor = risk_assessor
        self.ood_guard = ood_guard or OODGuard()
        self.ood_guard.set_scope_classifier(scope_model)
        self.taxonomy = taxonomy or TaxonomyResolver()
        self.metadata = metadata or {}
        self.queue_projector = QueueProjector(self.model.classes_, self.taxonomy)

        # Extract vocabulary from model (unwrapping CalibratedClassifierCV / FrozenEstimator)
        curr = model
        if hasattr(curr, "estimator"):
            curr = curr.estimator
        elif hasattr(curr, "calibrated_classifiers_") and curr.calibrated_classifiers_:
            curr = curr.calibrated_classifiers_[0].estimator
        if hasattr(curr, "estimator"):
            curr = curr.estimator

        if hasattr(curr, "named_steps"):
            if "features" in curr.named_steps:
                feats = curr.named_steps["features"]
                if hasattr(feats, "transformer_list"):
                    for name, trans in feats.transformer_list:
                        if name == "word_tfidf" and hasattr(trans, "vocabulary_"):
                            self.ood_guard.set_vocabulary(set(trans.vocabulary_.keys()))
                            break
            elif "tfidf" in curr.named_steps and hasattr(curr.named_steps["tfidf"], "vocabulary_"):
                self.ood_guard.set_vocabulary(set(curr.named_steps["tfidf"].vocabulary_.keys()))

    def _calculate_entropy(self, probabilities: np.ndarray, eps: float = 1e-12) -> float:
        """Shannon entropy for 1D probability distribution."""
        p = np.clip(probabilities, eps, 1.0)
        return float(-np.sum(p * np.log(p)))

    def route(
        self,
        text: str,
        top_k: int = 3,
        request_id: str | None = None,
    ) -> RoutingResult:
        """Execute end-to-end triage on a customer ticket."""
        req_id = request_id or f"req_{uuid.uuid4().hex[:12]}"
        classes = self.model.classes_

        # 1. Normalize identifiers before inference using the same contract as
        # offline training.  The raw text is only used for request-local audit.
        model_text = normalize_pii_semantically(text)
        proba = self.model.predict_proba([model_text])[0]
        ranked_indices = proba.argsort()[::-1]

        top_intent = str(classes[ranked_indices[0]])
        raw_confidence = float(proba[ranked_indices[0]])
        raw_margin = (
            float(proba[ranked_indices[0]] - proba[ranked_indices[1]])
            if len(ranked_indices) > 1
            else 1.0
        )
        entropy = self._calculate_entropy(proba)
        top_domain = self.taxonomy.get_domain(top_intent)

        queue_prediction = self.queue_projector.project(proba)

        # 2. OOD / Out-of-Scope Detection
        ood_detected, ood_reasons = self.ood_guard.detect(
            text=model_text,
            confidence=raw_confidence,
            margin=raw_margin,
        )

        # 3. Security Risk Assessment (ALWAYS scans all 77 classes, independent of top_k)
        risk = self.risk_assessor.assess(
            probabilities=proba,
            top_intent=top_intent,
            ood_detected=ood_detected,
            ood_reasons=ood_reasons,
        )

        # 4. Format alternatives strictly for client display presentation
        safe_top_k = max(1, min(top_k, len(classes)))
        alternatives = [
            {
                "intent": str(classes[idx]),
                "domain": self.taxonomy.get_domain(str(classes[idx])),
                "confidence": round(float(proba[idx]), 4),
            }
            for idx in ranked_indices[:safe_top_k]
        ]

        # 5. ML Intent Prediction representation
        prediction = IntentPrediction(
            intent=top_intent,
            domain=top_domain,
            confidence=round(raw_confidence, 4),
            margin=round(raw_margin, 4),
            entropy=round(entropy, 4),
            alternatives=alternatives,
        )

        # 6. Evaluate Operational Routing Policy
        decision = self.policy.evaluate(prediction, risk, queue_prediction)

        return RoutingResult(
            request_id=req_id,
            prediction=prediction,
            risk=risk,
            decision=decision,
            queue_prediction=queue_prediction,
            metadata=self.metadata,
        )

    def route_batch(
        self,
        texts: list[str],
        top_k: int = 3,
        top_ks: Sequence[int] | None = None,
    ) -> list[RoutingResult]:
        """Vectorized batch routing for high-throughput batch ticket pipelines."""
        if not texts:
            return []
        if top_ks is not None and len(top_ks) != len(texts):
            raise ValueError("top_ks must have the same length as texts")

        classes = self.model.classes_
        normalized_texts = [normalize_pii_semantically(text) for text in texts]
        proba_matrix = self.model.predict_proba(normalized_texts)
        results: list[RoutingResult] = []

        for i, model_text in enumerate(normalized_texts):
            proba = proba_matrix[i]
            ranked_indices = proba.argsort()[::-1]

            top_intent = str(classes[ranked_indices[0]])
            raw_conf = float(proba[ranked_indices[0]])
            raw_margin = (
                float(proba[ranked_indices[0]] - proba[ranked_indices[1]])
                if len(ranked_indices) > 1
                else 1.0
            )
            entropy = self._calculate_entropy(proba)
            top_domain = self.taxonomy.get_domain(top_intent)
            queue_prediction = self.queue_projector.project(proba)

            ood_detected, ood_reasons = self.ood_guard.detect(
                text=model_text,
                confidence=raw_conf,
                margin=raw_margin,
            )

            risk = self.risk_assessor.assess(
                probabilities=proba,
                top_intent=top_intent,
                ood_detected=ood_detected,
                ood_reasons=ood_reasons,
            )

            requested_top_k = top_ks[i] if top_ks is not None else top_k
            safe_top_k = max(1, min(int(requested_top_k), len(classes)))
            alternatives = [
                {
                    "intent": str(classes[idx]),
                    "domain": self.taxonomy.get_domain(str(classes[idx])),
                    "confidence": round(float(proba[idx]), 4),
                }
                for idx in ranked_indices[:safe_top_k]
            ]

            prediction = IntentPrediction(
                intent=top_intent,
                domain=top_domain,
                confidence=round(raw_conf, 4),
                margin=round(raw_margin, 4),
                entropy=round(entropy, 4),
                alternatives=alternatives,
            )

            decision = self.policy.evaluate(prediction, risk, queue_prediction)

            results.append(
                RoutingResult(
                    request_id=f"req_{uuid.uuid4().hex[:12]}",
                    prediction=prediction,
                    risk=risk,
                    decision=decision,
                    queue_prediction=queue_prediction,
                    metadata=self.metadata,
                )
            )

        return results
