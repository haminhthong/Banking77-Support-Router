"""Pipeline building blocks for TF-IDF feature extraction and linear classification."""

from __future__ import annotations

from typing import Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline


def build_pipeline(
    word_ngram_range: tuple[int, int] = (1, 2),
    char_ngram_range: tuple[int, int] | None = (3, 5),
    use_char_features: bool = True,
    c_param: float = 4.0,
    max_iter: int = 1200,
) -> tuple[Pipeline, dict[str, Any]]:
    """Build canonical classification pipeline with word and character n-gram robustness.

    Character n-grams ('char_wb') provide resilience against banking typos,
    abbreviations, and slight phrasing variations while maintaining sub-millisecond CPU latency.
    """
    word_vec_params = {
        "ngram_range": word_ngram_range,
        "min_df": 2,
        "max_df": 0.98,
        "sublinear_tf": True,
        "max_features": 40000,
    }

    lr_params = {
        "C": c_param,
        "max_iter": max_iter,
        "class_weight": "balanced",
        "solver": "lbfgs",
        "n_jobs": None,
    }

    if use_char_features and char_ngram_range is not None:
        char_vec_params = {
            "analyzer": "char_wb",
            "ngram_range": char_ngram_range,
            "min_df": 3,
            "max_features": 25000,
            "sublinear_tf": True,
        }
        features = FeatureUnion([
            ("word_tfidf", TfidfVectorizer(**word_vec_params)),
            ("char_tfidf", TfidfVectorizer(**char_vec_params)),
        ])
        pipeline = Pipeline([
            ("features", features),
            ("clf", LogisticRegression(**lr_params)),
        ])
        config = {
            "features": {"word": word_vec_params, "char": char_vec_params},
            "classifier": lr_params,
        }
    else:
        pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(**word_vec_params)),
            ("clf", LogisticRegression(**lr_params)),
        ])
        config = {
            "features": {"word": word_vec_params},
            "classifier": lr_params,
        }

    return pipeline, config
