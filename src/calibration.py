"""Reusable calibrated-probability wrapper for the Phase 3.5 artifact."""

from __future__ import annotations

import numpy as np


class ProbabilityCalibratedModel:
    """Wrap a fitted probability model with a fitted one-dimensional calibrator."""

    def __init__(self, base_model, calibrator, method: str):
        self.base_model = base_model
        self.calibrator = calibrator
        self.method = method

    def predict_raw_probability(self, X) -> np.ndarray:
        probabilities = self.base_model.predict_proba(X)
        classes = self.base_model.named_steps["model"].classes_
        positive_index = int(np.where(classes == "yes")[0][0])
        return probabilities[:, positive_index]

    def predict_proba(self, X) -> np.ndarray:
        raw_probability = self.predict_raw_probability(X)
        if self.method == "sigmoid":
            calibrated_probability = self.calibrator.predict_proba(raw_probability.reshape(-1, 1))[:, 1]
        else:
            calibrated_probability = self.calibrator.predict(raw_probability)
        calibrated_probability = np.asarray(calibrated_probability, dtype=float)
        calibrated_probability = np.clip(calibrated_probability, 0.0, 1.0)
        return np.column_stack([1.0 - calibrated_probability, calibrated_probability])

    def predict(self, X) -> np.ndarray:
        calibrated_probability = self.predict_proba(X)[:, 1]
        return np.where(calibrated_probability >= 0.5, "yes", "no")

