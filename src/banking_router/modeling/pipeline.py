"""Các khối xây dựng TF-IDF và classifier tuyến tính."""

from __future__ import annotations

from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline


def build_pipeline(
    model_config: dict[str, Any] | None = None,
    word_ngram_range: tuple[int, int] = (1, 2),
    char_ngram_range: tuple[int, int] | None = (3, 5),
    use_char_features: bool = True,
    c_param: float = 4.0,
    max_iter: int = 1200,
) -> tuple[Pipeline, dict[str, Any]]:
    """Tạo pipeline phân loại chuẩn với word n-gram và character n-gram.

    Character n-gram ``char_wb`` giúp mô hình ít nhạy với lỗi gõ, viết tắt
    và khác biệt nhỏ trong cách diễn đạt.
    """
    # Nếu không có YAML, dùng các giá trị mặc định của hàm.
    cfg = model_config or {}
    features_cfg = cfg.get("features", {})
    word_cfg = dict(features_cfg.get("word", features_cfg.get("tfidf", {})))
    char_cfg = dict(features_cfg.get("char", {}))
    classifier_cfg = dict(
        cfg.get("classifier", {}).get("params", cfg.get("classifier", {}))
    )

    word_vec_params = {
        "ngram_range": tuple(word_cfg.get("ngram_range", word_ngram_range)),
        "min_df": word_cfg.get("min_df", 2),
        "max_df": word_cfg.get("max_df", 0.98),
        "sublinear_tf": word_cfg.get("sublinear_tf", True),
        "max_features": word_cfg.get("max_features", 50000),
    }

    lr_params = {
        "C": classifier_cfg.get("C", c_param),
        "max_iter": classifier_cfg.get("max_iter", max_iter),
        "class_weight": classifier_cfg.get("class_weight", "balanced"),
        "solver": classifier_cfg.get("solver", "lbfgs"),
        "n_jobs": classifier_cfg.get("n_jobs"),
    }

    if use_char_features and char_ngram_range is not None:
        char_vec_params = {
            "analyzer": char_cfg.get("analyzer", "char_wb"),
            "ngram_range": tuple(char_cfg.get("ngram_range", char_ngram_range)),
            "min_df": char_cfg.get("min_df", 3),
            "max_features": char_cfg.get("max_features", 25000),
            "sublinear_tf": char_cfg.get("sublinear_tf", True),
        }
        features = FeatureUnion(
            [
                ("word_tfidf", TfidfVectorizer(**word_vec_params)),
                ("char_tfidf", TfidfVectorizer(**char_vec_params)),
            ]
        )
        pipeline = Pipeline(
            [
                ("features", features),
                ("clf", LogisticRegression(**lr_params)),
            ]
        )
        config = {
            "features": {"word": word_vec_params, "char": char_vec_params},
            "classifier": lr_params,
        }
    else:
        pipeline = Pipeline(
            [
                ("tfidf", TfidfVectorizer(**word_vec_params)),
                ("clf", LogisticRegression(**lr_params)),
            ]
        )
        config = {
            "features": {"word": word_vec_params},
            "classifier": lr_params,
        }

    return pipeline, config
