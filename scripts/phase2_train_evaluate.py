"""Phase 2 model training, evaluation, and limited hyperparameter tuning.

The outer test set is held out until final evaluation. Model selection and
tuning use only the development partition and an inner validation/CV process.
The saved best_model.joblib is a complete preprocessing-plus-estimator
Pipeline for future inference.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_sample_weight

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing import (  # noqa: E402
    LEAKAGE_COLUMNS,
    TARGET_COLUMN,
    build_preprocessor,
    infer_feature_types,
    prepare_features_target,
)

DATA_PATH = PROJECT_ROOT / "bank-full.csv"
MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"
FIGURE_DIR = REPORT_DIR / "figures"
RESULTS_JSON = REPORT_DIR / "phase2_model_results.json"
RESULTS_CSV = REPORT_DIR / "phase2_model_comparison.csv"
TEST_PREDICTIONS_CSV = REPORT_DIR / "phase2_test_predictions.csv"
REPORT_MD = REPORT_DIR / "phase2_model_evaluation.md"
BEST_MODEL_PATH = MODEL_DIR / "best_model.joblib"
BEST_METADATA_PATH = MODEL_DIR / "best_model_metadata.json"
RANDOM_STATE = 42
POSITIVE_LABEL = "yes"


def make_pipeline(estimator: Any, numeric_columns: list[str], categorical_columns: list[str]) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(numeric_columns, categorical_columns)),
            ("model", estimator),
        ]
    )


def fit_pipeline(pipeline: Pipeline, X: pd.DataFrame, y: pd.Series) -> Pipeline:
    """Fit a pipeline, applying balanced sample weights only to Gradient Boosting."""
    model = pipeline.named_steps["model"]
    if isinstance(model, GradientBoostingClassifier):
        weights = compute_sample_weight(class_weight="balanced", y=y)
        pipeline.fit(X, y, model__sample_weight=weights)
    else:
        pipeline.fit(X, y)
    return pipeline


def predict_probabilities(pipeline: Pipeline, X: pd.DataFrame) -> np.ndarray:
    probabilities = pipeline.predict_proba(X)
    classes = pipeline.named_steps["model"].classes_
    positive_index = int(np.where(classes == POSITIVE_LABEL)[0][0])
    return probabilities[:, positive_index]


def evaluate_predictions(y_true: pd.Series, probabilities: np.ndarray) -> dict[str, Any]:
    predictions = np.where(probabilities >= 0.5, POSITIVE_LABEL, "no")
    matrix = confusion_matrix(y_true, predictions, labels=["no", POSITIVE_LABEL])
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, pos_label=POSITIVE_LABEL, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, pos_label=POSITIVE_LABEL, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, pos_label=POSITIVE_LABEL, zero_division=0)),
        "roc_auc": float(roc_auc_score((y_true == POSITIVE_LABEL).astype(int), probabilities)),
        "pr_auc": float(average_precision_score((y_true == POSITIVE_LABEL).astype(int), probabilities)),
        "confusion_matrix": matrix.astype(int).tolist(),
        "threshold": 0.5,
    }


def save_confusion_matrix(model_name: str, metrics: dict[str, Any]) -> str:
    matrix = np.asarray(metrics["confusion_matrix"])
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = model_name.lower().replace(" ", "_").replace("/", "_")
    path = FIGURE_DIR / f"phase2_confusion_matrix_{safe_name}.png"
    fig, ax = plt.subplots(figsize=(5, 4))
    image = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks([0, 1], labels=["Predicted no", "Predicted yes"])
    ax.set_yticks([0, 1], labels=["Actual no", "Actual yes"])
    ax.set_title(f"Confusion matrix: {model_name}")
    for row in range(2):
        for col in range(2):
            ax.text(col, row, f"{matrix[row, col]:,}", ha="center", va="center")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(path.relative_to(PROJECT_ROOT)).replace("\\", "/")


def save_feature_importance(model_name: str, pipeline: Pipeline) -> tuple[str, list[dict[str, Any]]]:
    """Create native importance/coefficient analysis from the development fit."""
    names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    model = pipeline.named_steps["model"]
    if hasattr(model, "feature_importances_"):
        values = np.asarray(model.feature_importances_)
        importance_kind = "feature_importance"
    elif hasattr(model, "coef_"):
        values = np.abs(np.asarray(model.coef_).ravel())
        importance_kind = "absolute_coefficient"
    else:
        raise ValueError(f"No native importance representation for {model_name}")

    order = np.argsort(values)[::-1]
    top_indices = order[:20]
    top_items = [
        {"feature": str(names[index]), "importance": float(values[index]), "importance_kind": importance_kind}
        for index in top_indices
    ]
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = model_name.lower().replace(" ", "_").replace("/", "_")
    path = FIGURE_DIR / f"phase2_feature_importance_{safe_name}.png"
    plot_items = list(reversed(top_items))
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh([item["feature"] for item in plot_items], [item["importance"] for item in plot_items], color="#2f855a")
    ax.set_title(f"Top feature importance: {model_name}")
    ax.set_xlabel(importance_kind.replace("_", " ").title())
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"), top_items


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    frame = pd.read_csv(DATA_PATH, sep=";", quotechar='"')
    features, target = prepare_features_target(frame)
    numeric_columns, categorical_columns = infer_feature_types(features)
    if TARGET_COLUMN in features.columns:
        raise AssertionError("Target y was retained in the feature matrix.")
    if set(LEAKAGE_COLUMNS).intersection(features.columns):
        raise AssertionError("duration was retained in the feature matrix.")

    X_dev, X_test, y_dev, y_test = train_test_split(
        features, target, test_size=0.20, random_state=RANDOM_STATE, stratify=target
    )
    X_fit, X_validation, y_fit, y_validation = train_test_split(
        X_dev, y_dev, test_size=0.20, random_state=RANDOM_STATE, stratify=y_dev
    )

    estimators = {
        "Logistic Regression": LogisticRegression(
            max_iter=2000, class_weight="balanced", solver="liblinear", random_state=RANDOM_STATE
        ),
        "Decision Tree": DecisionTreeClassifier(
            class_weight="balanced", max_depth=8, min_samples_leaf=10, random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, class_weight="balanced_subsample", max_features="sqrt",
            n_jobs=1, random_state=RANDOM_STATE
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100, learning_rate=0.08, max_depth=3, min_samples_leaf=10,
            random_state=RANDOM_STATE
        ),
    }

    validation_scores: dict[str, dict[str, Any]] = {}
    for name, estimator in estimators.items():
        pipeline = make_pipeline(estimator, numeric_columns, categorical_columns)
        fit_pipeline(pipeline, X_fit, y_fit)
        validation_scores[name] = evaluate_predictions(
            y_validation, predict_probabilities(pipeline, X_validation)
        )

    selected_baseline = max(
        validation_scores,
        key=lambda name: (
            validation_scores[name]["pr_auc"], validation_scores[name]["roc_auc"], validation_scores[name]["f1"]
        ),
    )
    parameter_grids: dict[str, dict[str, list[Any]]] = {
        "Logistic Regression": {"model__C": [0.1, 1.0, 3.0]},
        "Decision Tree": {"model__max_depth": [4, 8, None], "model__min_samples_leaf": [10, 30]},
        "Random Forest": {"model__n_estimators": [75], "model__max_depth": [8, None], "model__min_samples_leaf": [5, 15]},
        "Gradient Boosting": {"model__n_estimators": [75, 100], "model__learning_rate": [0.05, 0.1], "model__max_depth": [2, 3]},
    }

    cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=RANDOM_STATE)
    tuning_estimator = make_pipeline(estimators[selected_baseline], numeric_columns, categorical_columns)
    search = GridSearchCV(
        estimator=tuning_estimator,
        param_grid=parameter_grids[selected_baseline],
        scoring=make_scorer(
            average_precision_score,
            response_method="predict_proba",
            pos_label=POSITIVE_LABEL,
        ),
        cv=cv, n_jobs=1, refit=True, return_train_score=False
    )
    if selected_baseline == "Gradient Boosting":
        search.fit(
            X_dev,
            y_dev,
            model__sample_weight=compute_sample_weight(class_weight="balanced", y=y_dev),
        )
    else:
        search.fit(X_dev, y_dev)
    tuned_pipeline = search.best_estimator_

    test_results: dict[str, dict[str, Any]] = {}
    final_pipelines: dict[str, Pipeline] = {}
    test_probability_rows: list[pd.DataFrame] = []
    for name, estimator in estimators.items():
        pipeline = make_pipeline(estimator, numeric_columns, categorical_columns)
        fit_pipeline(pipeline, X_dev, y_dev)
        final_pipelines[name] = pipeline
        probabilities = predict_probabilities(pipeline, X_test)
        test_results[name] = evaluate_predictions(y_test, probabilities)
        test_probability_rows.append(pd.DataFrame({
            "row_id": X_test.index,
            "actual_y": y_test.to_numpy(),
            "predicted_probability_yes": probabilities,
            "predicted_y_at_0_5": np.where(probabilities >= 0.5, "yes", "no"),
            "model": name,
        }))

    tuned_probabilities = predict_probabilities(tuned_pipeline, X_test)
    tuned_name = f"Tuned {selected_baseline}"
    test_results[tuned_name] = evaluate_predictions(y_test, tuned_probabilities)
    final_pipelines[tuned_name] = tuned_pipeline
    test_probability_rows.append(pd.DataFrame({
        "row_id": X_test.index,
        "actual_y": y_test.to_numpy(),
        "predicted_probability_yes": tuned_probabilities,
        "predicted_y_at_0_5": np.where(tuned_probabilities >= 0.5, "yes", "no"),
        "model": tuned_name,
    }))

    best_model_name = tuned_name
    best_pipeline = tuned_pipeline
    best_metrics = test_results[best_model_name]
    model_rows = []
    for name, metrics in test_results.items():
        model_rows.append({
            "model": name, "accuracy": metrics["accuracy"], "precision": metrics["precision"],
            "recall": metrics["recall"], "f1": metrics["f1"], "roc_auc": metrics["roc_auc"],
            "pr_auc": metrics["pr_auc"], "tn": metrics["confusion_matrix"][0][0],
            "fp": metrics["confusion_matrix"][0][1], "fn": metrics["confusion_matrix"][1][0],
            "tp": metrics["confusion_matrix"][1][1], "evaluation_split": "untouched outer test set",
        })
    comparison = pd.DataFrame(model_rows)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(RESULTS_CSV, index=False)
    pd.concat(test_probability_rows, ignore_index=True).sort_values(["model", "row_id"]).to_csv(
        TEST_PREDICTIONS_CSV, index=False
    )
    joblib.dump(best_pipeline, BEST_MODEL_PATH)
    importance_figure, top_importance = save_feature_importance(best_model_name, best_pipeline)
    confusion_figures = {name: save_confusion_matrix(name, metrics) for name, metrics in test_results.items()}
    tuning_results = [
        {"params": params, "mean_cv_pr_auc": float(mean_score), "std_cv_pr_auc": float(std_score)}
        for params, mean_score, std_score in zip(
            search.cv_results_["params"], search.cv_results_["mean_test_score"], search.cv_results_["std_test_score"]
        )
    ]
    tuning_results.sort(key=lambda item: item["mean_cv_pr_auc"], reverse=True)

    profile = {
        "dataset": str(DATA_PATH.relative_to(PROJECT_ROOT)), "random_state": RANDOM_STATE,
        "target_column": TARGET_COLUMN, "excluded_leakage_columns": list(LEAKAGE_COLUMNS),
        "feature_columns": features.columns.tolist(), "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "split_sizes": {"development": int(len(X_dev)), "test": int(len(X_test)), "inner_fit": int(len(X_fit)), "inner_validation": int(len(X_validation))},
        "selection_rule": "highest inner-validation PR-AUC, then ROC-AUC, then F1; test metrics are not used for selection",
        "class_imbalance": {
            "logistic_regression": "class_weight=balanced", "decision_tree": "class_weight=balanced",
            "random_forest": "class_weight=balanced_subsample", "gradient_boosting": "balanced sample weights during fit",
        },
        "validation_scores_for_selection": validation_scores, "tuned_model": selected_baseline,
        "best_params": search.best_params_, "best_cv_pr_auc": float(search.best_score_),
        "tuning_results": tuning_results, "test_results": test_results, "best_model": best_model_name,
        "best_model_path": str(BEST_MODEL_PATH.relative_to(PROJECT_ROOT)),
        "test_predictions_path": str(TEST_PREDICTIONS_CSV.relative_to(PROJECT_ROOT)),
        "feature_importance_figure": importance_figure, "top_feature_importance": top_importance,
        "confusion_matrix_figures": confusion_figures,
    }
    RESULTS_JSON.write_text(json.dumps(profile, indent=2, default=str), encoding="utf-8")
    BEST_METADATA_PATH.write_text(json.dumps({
        "best_model": best_model_name, "target_column": TARGET_COLUMN,
        "excluded_leakage_columns": list(LEAKAGE_COLUMNS), "random_state": RANDOM_STATE,
        "feature_columns": features.columns.tolist(), "best_params": search.best_params_,
        "selection_rule": profile["selection_rule"],
    }, indent=2), encoding="utf-8")

    def pct(value: float) -> str:
        return f"{value:.4f}"

    report = f"""# Phase 2: Model Training, Evaluation, and Practical Tuning

## Scope and evaluation protocol

- Dataset: `{DATA_PATH.name}`
- Target: `{TARGET_COLUMN}`
- Excluded leakage column: `duration`
- Random state: `{RANDOM_STATE}`
- Development rows: **{len(X_dev):,}**
- Untouched test rows: **{len(X_test):,}**
- Preprocessing was fitted inside each training/CV fit only.
- ROC-AUC and PR-AUC use predicted probability scores for `y=yes`.
- The test set was used once for final evaluation and not for tuning or model selection.

## Class-imbalance handling

- Logistic Regression: `class_weight="balanced"`
- Decision Tree: `class_weight="balanced"`
- Random Forest: `class_weight="balanced_subsample"`
- Gradient Boosting: balanced per-row sample weights during fitting
- PR-AUC is emphasized because the positive class represents a minority of customers.

## Model comparison on the untouched test set

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | TN | FP | FN | TP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
"""
    for row in model_rows:
        report += f"| {row['model']} | {pct(row['accuracy'])} | {pct(row['precision'])} | {pct(row['recall'])} | {pct(row['f1'])} | {pct(row['roc_auc'])} | {pct(row['pr_auc'])} | {row['tn']:,} | {row['fp']:,} | {row['fn']:,} | {row['tp']:,} |\n"
    report += f"""
## Model selection

The baseline selected for tuning was **{selected_baseline}**, based on inner
validation PR-AUC, followed by ROC-AUC and F1 as tie-breakers. The tuned model
was selected without using the outer test metrics.

The final saved model is **{best_model_name}**. It is appropriate for future
customer prioritization because it outputs a response probability and its
selection emphasizes ranking quality for the minority positive class rather
than accuracy alone. The saved Joblib artifact contains the fitted
preprocessing transformer and estimator in one Pipeline.

## Hyperparameter tuning

- Search scoring: average precision, which is the probability-based PR-AUC.
- Cross-validation: 2-fold stratified CV on the development set only, keeping the tuning budget practical for this project.
- Best parameters: `{json.dumps(search.best_params_, default=str)}`
- Best mean CV PR-AUC: **{search.best_score_:.4f}**

| Rank | Parameters | Mean CV PR-AUC | Std. CV PR-AUC |
|---:|---|---:|---:|
"""
    for rank, item in enumerate(tuning_results, start=1):
        report += f"| {rank} | `{json.dumps(item['params'], default=str)}` | {item['mean_cv_pr_auc']:.4f} | {item['std_cv_pr_auc']:.4f} |\n"
    report += "\n## Feature importance findings\n\n"
    report += f"Native importance analysis for **{best_model_name}** was generated from the fitted development model. It is descriptive and does not imply causality.\n\n"
    for item in top_importance[:15]:
        report += f"- `{item['feature']}`: {item['importance']:.6f}\n"
    report += "\n## Saved artifacts\n\n"
    report += f"- `{RESULTS_CSV.relative_to(PROJECT_ROOT).as_posix()}`\n"
    report += f"- `{RESULTS_JSON.relative_to(PROJECT_ROOT).as_posix()}`\n"
    report += f"- `{TEST_PREDICTIONS_CSV.relative_to(PROJECT_ROOT).as_posix()}` containing each held-out test row, actual outcome, predicted probability, and 0.5-threshold prediction for every evaluated model\n"
    report += f"- `{BEST_MODEL_PATH.relative_to(PROJECT_ROOT).as_posix()}` containing the complete fitted preprocessing-plus-estimator pipeline\n"
    report += f"- `{BEST_METADATA_PATH.relative_to(PROJECT_ROOT).as_posix()}`\n"
    report += f"- `{importance_figure}`\n"
    report += "\n## Phase 2 boundary\n\nNo FastAPI backend, frontend, customer prioritization UI, deployment, or public URL was created.\n"
    REPORT_MD.write_text(report, encoding="utf-8")
    print(f"Completed Phase 2 on {len(frame):,} rows.")
    print(f"Selected baseline for tuning: {selected_baseline}")
    print(f"Final best model: {best_model_name}")
    print(f"Test PR-AUC: {best_metrics['pr_auc']:.4f}")
    print(f"Created report: {REPORT_MD}")


if __name__ == "__main__":
    main()

