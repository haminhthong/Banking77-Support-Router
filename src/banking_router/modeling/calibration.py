"""Probability calibration via Platt Scaling (Sigmoid Calibrator)."""

from __future__ import annotations

from typing import Any
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline


def build_calibrated_model(base_model: Pipeline) -> CalibratedClassifierCV:
    """Instantiate CalibratedClassifierCV with prefit base estimator across scikit-learn versions."""
    try:
        # scikit-learn >= 1.5
        from sklearn.frozen import FrozenEstimator

        return CalibratedClassifierCV(
            estimator=FrozenEstimator(base_model),
            method="sigmoid",
        )
    except (ImportError, ModuleNotFoundError):
        # scikit-learn < 1.5
        return CalibratedClassifierCV(
            estimator=base_model,
            cv="prefit",
            method="sigmoid",
        )
