"""Mô hình phân biệt câu hỏi thuộc phạm vi Banking77 và câu hỏi ngoài phạm vi."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

from ..data.normalization import normalize_pii_semantically

SUPPORTED = "SUPPORTED"
UNSUPPORTED = "UNSUPPORTED"


@dataclass
class ScopeClassifier:
    """Wrapper nhỏ để runtime chỉ cần gọi ``predict_proba``."""

    pipeline: Pipeline
    model_name: str = "scope-word-char-logistic-regression"

    @property
    def classes_(self) -> np.ndarray:
        return self.pipeline.classes_

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        normalized = [normalize_pii_semantically(text) for text in texts]
        return self.pipeline.predict_proba(normalized)

    def unsupported_probability(self, text: str) -> float:
        probabilities = self.predict_proba([text])[0]
        index = list(self.classes_).index(UNSUPPORTED)
        return float(probabilities[index])


def train_scope_classifier(
    supported_texts: Sequence[str],
    unsupported_texts: Sequence[str],
    seed: int = 42,
) -> ScopeClassifier:
    """Huấn luyện model phạm vi từ dữ liệu Banking77 và tập ngoài phạm vi có nhãn."""
    texts = [normalize_pii_semantically(text) for text in supported_texts]
    texts.extend(normalize_pii_semantically(text) for text in unsupported_texts)
    labels = np.array(
        [SUPPORTED] * len(supported_texts) + [UNSUPPORTED] * len(unsupported_texts)
    )
    if len(set(labels)) != 2:
        raise ValueError("Scope classifier cần cả mẫu SUPPORTED và UNSUPPORTED")

    features = FeatureUnion(
        [
            (
                "word",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(1, 2),
                    min_df=1,
                    sublinear_tf=True,
                    max_features=30000,
                ),
            ),
            (
                "char",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    min_df=1,
                    sublinear_tf=True,
                    max_features=40000,
                ),
            ),
        ]
    )
    pipeline = Pipeline(
        [
            ("features", features),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=seed,
                ),
            ),
        ]
    )
    pipeline.fit(texts, labels)
    return ScopeClassifier(pipeline=pipeline)
