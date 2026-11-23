"""Model loading and inference logic for the banking campaign API."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


class BankingModelService:
    """Loads model artifacts once and serves inference without retraining."""

    FEATURE_COLUMNS = [
        "age", "job", "marital", "education", "default", "balance", "housing", "loan",
        "contact", "day", "month", "campaign", "pdays", "previous", "poutcome",
    ]

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.model_path = project_root / "models" / "calibrated_best_model.joblib"
        self.metadata_path = project_root / "models" / "calibrated_model_metadata.json"
        if not self.model_path.exists():
            raise FileNotFoundError(f"Calibrated model not found: {self.model_path}")
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"Calibrated model metadata not found: {self.metadata_path}")

        import json

        self.model = joblib.load(self.model_path)
        self.metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        if self.metadata.get("selected_calibration_method") != "sigmoid":
            raise ValueError("The API requires the selected sigmoid-calibrated Phase 3.5 artifact.")
        if not hasattr(self.model, "base_model") or not hasattr(self.model, "predict_proba"):
            raise TypeError("Calibrated model artifact does not expose the expected inference interface.")

    @property
    def model_name(self) -> str:
        return "Tuned Gradient Boosting"

    @property
    def calibration_method(self) -> str:
        return "sigmoid"

    def _frame(self, customer: Any) -> pd.DataFrame:
        values = customer.model_dump() if hasattr(customer, "model_dump") else dict(customer)
        values.pop("customer_id", None)
        missing = [column for column in self.FEATURE_COLUMNS if column not in values]
        if missing:
            raise ValueError(f"Missing pre-call features: {missing}")
        return pd.DataFrame([{column: values[column] for column in self.FEATURE_COLUMNS}], columns=self.FEATURE_COLUMNS)

    def _probabilities(self, frame: pd.DataFrame) -> tuple[float, float]:
        # Explicitly use the underlying base Gradient Boosting pipeline for the
        # raw ranking score. Never substitute this value for calibrated output.
        raw_probability = float(self.model.predict_raw_probability(frame)[0])
        # The calibrated artifact is sigmoid-selected. This is the only value
        # used for response probability and expected responder estimates.
        calibrated_probability = float(self.model.predict_proba(frame)[0, 1])
        return raw_probability, calibrated_probability

    @staticmethod
    def priority_category(probability: float) -> str:
        if probability >= 0.75:
            return "Very High"
        if probability >= 0.50:
            return "High"
        if probability >= 0.25:
            return "Medium"
        return "Low"

    def predict(self, customer: Any) -> dict[str, Any]:
        frame = self._frame(customer)
        raw_probability, calibrated_probability = self._probabilities(frame)
        return {
            "customer_id": customer.customer_id,
            "predicted_response_class": "yes" if calibrated_probability >= 0.5 else "no",
            "calibrated_response_probability": calibrated_probability,
            "raw_ranking_score": raw_probability,
            "priority_category": self.priority_category(calibrated_probability),
            "probability_note": "Calibrated estimate, not a guarantee of subscription.",
        }

    def prioritize(self, customers: list[Any]) -> list[dict[str, Any]]:
        rows = []
        for customer in customers:
            result = self.predict(customer)
            rows.append(result)
        rows.sort(key=lambda row: (-row["raw_ranking_score"], row["customer_id"]))
        for rank, row in enumerate(rows, start=1):
            row["priority_rank"] = rank
        return rows

    def capacity_analysis(self, customers: list[Any], capacity_percentage: float) -> dict[str, Any]:
        ranked = self.prioritize(customers)
        available = len(ranked)
        # Floor ensures the selected count never exceeds requested capacity;
        # 100% explicitly selects all available customers.
        selected_count = available if capacity_percentage == 100 else max(1, int(np.floor(available * capacity_percentage / 100)))
        selected = ranked[:selected_count]
        average_probability = float(np.mean([row["calibrated_response_probability"] for row in selected]))
        expected_responders = float(sum(row["calibrated_response_probability"] for row in selected))
        return {
            "capacity_percentage": capacity_percentage,
            "rounding_method": "floor(N × capacity_percentage / 100), with 100% equal to N",
            "customers_available": available,
            "customers_selected": selected_count,
            "average_calibrated_response_probability": average_probability,
            "expected_responders_estimate": expected_responders,
            "customers": selected,
            "probability_note": "Expected responders are the sum of calibrated probabilities, not guaranteed subscriptions. Actual outcomes are unknown for future customers.",
        }

