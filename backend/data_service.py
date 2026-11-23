"""Read-only project statistics for summary and model-information endpoints."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve


class ProjectDataService:
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.dataset = pd.read_csv(project_root / "bank-full.csv", sep=";", quotechar='"')
        self.phase2 = json.loads((project_root / "reports" / "phase2_model_results.json").read_text(encoding="utf-8"))
        self.phase2_metadata = json.loads((project_root / "models" / "best_model_metadata.json").read_text(encoding="utf-8"))
        self.calibration = json.loads((project_root / "reports" / "phase3_5_calibration_results.json").read_text(encoding="utf-8"))
        self.calibrated_test_predictions = pd.read_csv(
            project_root / "reports" / "phase3_5_test_calibrated_predictions.csv"
        )

    def summary(self) -> dict:
        responders = int(self.dataset["y"].eq("yes").sum())
        return {
            "total_customers": int(len(self.dataset)),
            "total_historical_responders": responders,
            "historical_response_rate": responders / len(self.dataset),
            "selected_model": self.phase2.get("best_model", "Tuned Gradient Boosting"),
            "roc_auc": self.phase2["test_results"]["Tuned Gradient Boosting"]["roc_auc"],
            "pr_auc": self.phase2["test_results"]["Tuned Gradient Boosting"]["pr_auc"],
            "calibration_method": self.calibration.get("selected_method", "sigmoid"),
        }

    def model_info(self, feature_columns: list[str]) -> dict:
        metrics = self.phase2["test_results"]["Tuned Gradient Boosting"]
        return {
            "model_name": "Tuned Gradient Boosting",
            "features_used": feature_columns,
            "excluded_feature": "duration",
            "evaluation_metrics": metrics,
            "calibration": {
                "method": self.calibration.get("selected_method", "sigmoid"),
                "brier_score_outer_test": self.calibration["outer_test_metrics"]["sigmoid"]["brier_score"],
                "calibrated_expected_responders_outer_test": self.calibration["sigmoid_expected_responders"],
                "calibration_fit": "5-fold development out-of-fold predictions",
            },
            "ranking_score_definition": "Raw positive-class probability from the underlying base Gradient Boosting pipeline.",
            "probability_definition": "Sigmoid-calibrated positive-class probability from calibrated_best_model.joblib.",
        }

    def roc_curve(self) -> dict:
        required = {"actual_y_retrospective", "selected_calibrated_probability"}
        missing = required.difference(self.calibrated_test_predictions.columns)
        if missing:
            raise ValueError(f"ROC artifact is missing required columns: {sorted(missing)}")

        labels = (self.calibrated_test_predictions["actual_y_retrospective"] == "yes").astype(int)
        probabilities = self.calibrated_test_predictions["selected_calibrated_probability"].astype(float)
        fpr, tpr, _ = roc_curve(labels, probabilities)
        roc_auc = float(roc_auc_score(labels, probabilities))
        expected_auc = float(self.phase2["test_results"]["Tuned Gradient Boosting"]["roc_auc"])
        if abs(roc_auc - expected_auc) > 1e-12:
            raise ValueError(f"Regenerated ROC-AUC {roc_auc} does not match saved ROC-AUC {expected_auc}")

        return {
            "model_name": "Tuned Gradient Boosting",
            "calibration_method": self.calibration.get("selected_method", "sigmoid"),
            "evaluation_set": "outer held-out test set",
            "roc_auc": roc_auc,
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "reference_fpr": [0.0, 1.0],
            "reference_tpr": [0.0, 1.0],
        }

