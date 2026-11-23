"""Phase 3.5 leakage-safe probability calibration and evaluation.

The development portion is used to generate out-of-fold raw probabilities and
fit calibration mappings. The outer held-out test set is read only for final
before/after evaluation and is never used to fit or select a calibrator.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_sample_weight

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.calibration import ProbabilityCalibratedModel  # noqa: E402

DATA_PATH = PROJECT_ROOT / "bank-full.csv"
BASE_MODEL_PATH = PROJECT_ROOT / "models" / "best_model.joblib"
BASE_METADATA_PATH = PROJECT_ROOT / "models" / "best_model_metadata.json"
TEST_PREDICTIONS_PATH = PROJECT_ROOT / "reports" / "phase2_test_predictions.csv"
RANKED_PATH = PROJECT_ROOT / "reports" / "phase3_ranked_customers.csv"
REPORT_DIR = PROJECT_ROOT / "reports"
FIGURE_DIR = REPORT_DIR / "figures"
COMPARISON_CSV = REPORT_DIR / "phase3_5_calibration_comparison.csv"
RESULTS_JSON = REPORT_DIR / "phase3_5_calibration_results.json"
TEST_OUTPUT_CSV = REPORT_DIR / "phase3_5_test_calibrated_predictions.csv"
REPORT_MD = REPORT_DIR / "phase3_5_calibration_evaluation.md"
RELIABILITY_FIGURE = FIGURE_DIR / "phase3_5_reliability_diagram.png"
CALIBRATED_MODEL_PATH = PROJECT_ROOT / "models" / "calibrated_best_model.joblib"
CALIBRATED_METADATA_PATH = PROJECT_ROOT / "models" / "calibrated_model_metadata.json"

RANDOM_STATE = 42
N_SPLITS = 5
POSITIVE_LABEL = "yes"


def raw_probability(model, X: pd.DataFrame) -> np.ndarray:
    probabilities = model.predict_proba(X)
    classes = model.named_steps["model"].classes_
    positive_index = int(np.where(classes == POSITIVE_LABEL)[0][0])
    return probabilities[:, positive_index]


def fit_base_pipeline(pipeline, X: pd.DataFrame, y: pd.Series):
    model = pipeline.named_steps["model"]
    if model.__class__.__name__ == "GradientBoostingClassifier":
        weights = compute_sample_weight(class_weight="balanced", y=y)
        pipeline.fit(X, y, model__sample_weight=weights)
    else:
        pipeline.fit(X, y)
    return pipeline


def metrics(y_true: pd.Series, probabilities: np.ndarray) -> dict[str, float]:
    y_binary = (y_true == POSITIVE_LABEL).astype(int)
    return {
        "brier_score": float(brier_score_loss(y_binary, probabilities)),
        "roc_auc": float(roc_auc_score(y_binary, probabilities)),
        "pr_auc": float(average_precision_score(y_binary, probabilities)),
        "expected_responders_estimate": float(np.sum(probabilities)),
    }


def main() -> None:
    required = [DATA_PATH, BASE_MODEL_PATH, BASE_METADATA_PATH, TEST_PREDICTIONS_PATH, RANKED_PATH]
    for path in required:
        if not path.exists():
            raise FileNotFoundError(f"Required Phase 3.5 input not found: {path}")

    full_frame = pd.read_csv(DATA_PATH, sep=";", quotechar='"')
    base_model = joblib.load(BASE_MODEL_PATH)
    metadata = json.loads(BASE_METADATA_PATH.read_text(encoding="utf-8"))
    prediction_frame = pd.read_csv(TEST_PREDICTIONS_PATH)
    ranked_frame = pd.read_csv(RANKED_PATH)

    if metadata.get("best_model") != "Tuned Gradient Boosting":
        raise ValueError("best_model.joblib metadata does not identify Tuned Gradient Boosting.")
    if not isinstance(base_model, __import__("sklearn.pipeline", fromlist=["Pipeline"]).Pipeline):
        raise TypeError("best_model.joblib must contain the complete fitted sklearn Pipeline.")

    tuned_predictions = prediction_frame.loc[prediction_frame["model"].eq("Tuned Gradient Boosting")].copy()
    if len(tuned_predictions) != len(ranked_frame) or tuned_predictions["row_id"].nunique() != len(ranked_frame):
        raise AssertionError("Phase 2 tuned predictions and Phase 3 ranked rows do not contain the same test population.")
    test_ids = ranked_frame["customer_index"].astype(int).to_numpy()
    if len(np.unique(test_ids)) != len(test_ids):
        raise AssertionError("Outer test customer indexes are not unique.")
    if set(test_ids) != set(tuned_predictions["row_id"].astype(int)):
        raise AssertionError("Outer test identifiers do not match the Tuned Gradient Boosting prediction artifact.")

    features = full_frame.drop(columns=["y", "duration"])
    target = full_frame["y"].astype(str)
    X_test = features.iloc[test_ids].copy()
    y_test = target.iloc[test_ids].copy()
    development_mask = ~features.index.isin(test_ids)
    X_development = features.loc[development_mask].copy()
    y_development = target.loc[development_mask].copy()
    if len(X_test) != 9043 or len(X_development) != len(full_frame) - 9043:
        raise AssertionError("Development/test partition does not match the Phase 2 outer split sizes.")

    source_actual = full_frame.iloc[test_ids]["y"].astype(str).to_numpy()
    artifact_actual = tuned_predictions.set_index("row_id").loc[test_ids, "actual_y"].astype(str).to_numpy()
    if not np.array_equal(source_actual, artifact_actual):
        raise AssertionError("Test actual outcomes do not match bank-full.csv by original index.")

    # Generate leakage-safe development OOF probabilities. Each validation fold
    # receives predictions from a clone fitted only on that fold's training rows.
    oof_probability = np.full(len(X_development), np.nan, dtype=float)
    splitter = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    for fold, (fit_indices, validation_indices) in enumerate(splitter.split(X_development, y_development), start=1):
        fold_model = clone(base_model)
        fold_model = fit_base_pipeline(
            fold_model,
            X_development.iloc[fit_indices],
            y_development.iloc[fit_indices],
        )
        oof_probability[validation_indices] = raw_probability(
            fold_model, X_development.iloc[validation_indices]
        )
        print(f"Generated OOF probabilities for fold {fold}/{N_SPLITS}.")
    if np.isnan(oof_probability).any():
        raise AssertionError("OOF probability generation left unfilled development rows.")

    y_development_binary = (y_development == POSITIVE_LABEL).astype(int).to_numpy()
    sigmoid_calibrator = LogisticRegression(solver="lbfgs", max_iter=1000, random_state=RANDOM_STATE)
    sigmoid_calibrator.fit(oof_probability.reshape(-1, 1), y_development_binary)
    isotonic_calibrator = IsotonicRegression(out_of_bounds="clip")
    isotonic_calibrator.fit(oof_probability, y_development_binary)

    oof_sigmoid = sigmoid_calibrator.predict_proba(oof_probability.reshape(-1, 1))[:, 1]
    oof_isotonic = isotonic_calibrator.predict(oof_probability)
    oof_metrics = {
        "raw": metrics(y_development, oof_probability),
        "sigmoid": metrics(y_development, oof_sigmoid),
        "isotonic": metrics(y_development, oof_isotonic),
    }
    # Select using development OOF data only. A candidate must improve Brier
    # score and retain ranking quality within 0.005 PR-AUC/ROC-AUC of raw.
    # This prevents isotonic ties from changing the prioritization ranking.
    acceptable_methods = [
        method
        for method in ["sigmoid", "isotonic"]
        if oof_metrics[method]["brier_score"] < oof_metrics["raw"]["brier_score"]
        and oof_metrics[method]["pr_auc"] >= oof_metrics["raw"]["pr_auc"] - 0.005
        and oof_metrics[method]["roc_auc"] >= oof_metrics["raw"]["roc_auc"] - 0.005
    ]
    if not acceptable_methods:
        raise RuntimeError("No calibration method improved reliability while preserving acceptable OOF ranking quality.")
    selected_method = min(acceptable_methods, key=lambda method: oof_metrics[method]["brier_score"])
    oof_reliability_improved = oof_metrics[selected_method]["brier_score"] < oof_metrics["raw"]["brier_score"]
    if not oof_reliability_improved:
        raise RuntimeError("Neither calibration method improved OOF Brier score; no calibrated model will be saved.")

    raw_test_probability = raw_probability(base_model, X_test)
    sigmoid_test_probability = sigmoid_calibrator.predict_proba(raw_test_probability.reshape(-1, 1))[:, 1]
    isotonic_test_probability = isotonic_calibrator.predict(raw_test_probability)
    test_metrics = {
        "raw": metrics(y_test, raw_test_probability),
        "sigmoid": metrics(y_test, sigmoid_test_probability),
        "isotonic": metrics(y_test, isotonic_test_probability),
    }
    selected_test_probability = sigmoid_test_probability if selected_method == "sigmoid" else isotonic_test_probability
    selected_test_metrics = test_metrics[selected_method]
    actual_test_responders = int((y_test == POSITIVE_LABEL).sum())

    # Reliability curves are evaluation-only use of the untouched outer test set.
    from sklearn.calibration import calibration_curve

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 6))
    for label, probabilities, color in [
        ("Raw", raw_test_probability, "#718096"),
        ("Sigmoid", sigmoid_test_probability, "#2f855a"),
        ("Isotonic", isotonic_test_probability, "#d97706"),
    ]:
        observed, predicted = calibration_curve(
            (y_test == POSITIVE_LABEL).astype(int), probabilities, n_bins=10, strategy="quantile"
        )
        ax.plot(predicted, observed, marker="o", label=label, color=color)
    ax.plot([0, 1], [0, 1], linestyle="--", color="black", label="Perfect calibration")
    ax.set_title("Reliability diagram on untouched outer test set")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed response frequency")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(RELIABILITY_FIGURE, dpi=150, bbox_inches="tight")
    plt.close(fig)

    output = pd.DataFrame({
        "customer_index": test_ids,
        "actual_y_retrospective": y_test.to_numpy(),
        "raw_ranking_probability": raw_test_probability,
        "sigmoid_calibrated_probability": sigmoid_test_probability,
        "isotonic_calibrated_probability": isotonic_test_probability,
        "selected_calibrated_probability": selected_test_probability,
        "selected_calibration_method": selected_method,
    })
    output.to_csv(TEST_OUTPUT_CSV, index=False)

    comparison_rows = []
    for dataset_name, result_set in [("development_oof", oof_metrics), ("outer_test_final", test_metrics)]:
        for method, result in result_set.items():
            comparison_rows.append({"evaluation_set": dataset_name, "method": method, **result})
    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(COMPARISON_CSV, index=False)

    calibrated_model = ProbabilityCalibratedModel(
        base_model=base_model,
        calibrator=sigmoid_calibrator if selected_method == "sigmoid" else isotonic_calibrator,
        method=selected_method,
    )
    joblib.dump(calibrated_model, CALIBRATED_MODEL_PATH)
    CALIBRATED_METADATA_PATH.write_text(json.dumps({
        "base_model": str(BASE_MODEL_PATH.relative_to(PROJECT_ROOT)),
        "selected_calibration_method": selected_method,
        "calibration_fit_data": "development-set 5-fold out-of-fold raw probabilities only",
        "outer_test_used_for_fit": False,
        "outer_test_used_for_method_selection": False,
        "random_state": RANDOM_STATE,
        "n_splits": N_SPLITS,
        "intended_probability_field": "selected_calibrated_probability",
        "ranking_field": "raw_ranking_probability",
    }, indent=2), encoding="utf-8")

    result_json = {
        "dataset": str(DATA_PATH.relative_to(PROJECT_ROOT)),
        "base_model": str(BASE_MODEL_PATH.relative_to(PROJECT_ROOT)),
        "outer_test_customers": len(X_test),
        "development_customers": len(X_development),
        "actual_outer_test_responders": actual_test_responders,
        "random_state": RANDOM_STATE,
        "n_splits": N_SPLITS,
        "calibration_methodology": "Clone the fitted Phase 2 pipeline for each development fold, fit preprocessing and the base estimator on fold-train rows only, generate raw probabilities on fold-validation rows, fit sigmoid and isotonic mappings on the complete OOF probability vector, then evaluate once on the untouched outer test set.",
        "selection_rule": "Select a development-OOF method with lower Brier score and no more than 0.005 loss in PR-AUC or ROC-AUC versus raw; choose the lowest Brier score. The outer test set is not used for calibration method selection.",
        "selected_method": selected_method,
        "oof_reliability_improved": bool(oof_reliability_improved),
        "oof_metrics": oof_metrics,
        "outer_test_metrics": test_metrics,
        "actual_outer_test_responders": actual_test_responders,
        "raw_expected_responders": float(np.sum(raw_test_probability)),
        "sigmoid_expected_responders": float(np.sum(sigmoid_test_probability)),
        "isotonic_expected_responders": float(np.sum(isotonic_test_probability)),
        "selected_calibrated_expected_responders": float(np.sum(selected_test_probability)),
        "files": {
            "comparison": str(COMPARISON_CSV.relative_to(PROJECT_ROOT)),
            "test_predictions": str(TEST_OUTPUT_CSV.relative_to(PROJECT_ROOT)),
            "reliability_figure": str(RELIABILITY_FIGURE.relative_to(PROJECT_ROOT)),
            "calibrated_model": str(CALIBRATED_MODEL_PATH.relative_to(PROJECT_ROOT)),
        },
    }
    RESULTS_JSON.write_text(json.dumps(result_json, indent=2), encoding="utf-8")

    report = f"""# Phase 3.5: Probability Calibration and Evaluation

## Methodology

- Base model: the unchanged Phase 2 Tuned Gradient Boosting pipeline.
- Outer test set: **{len(X_test):,}** customers, never used for calibration fitting or method selection.
- Development set: **{len(X_development):,}** customers.
- Development calibration predictions: **{N_SPLITS}-fold stratified out-of-fold probabilities**.
- In each fold, preprocessing and the base estimator were fitted only on the fold-training rows.
- Sigmoid/Platt and isotonic mappings were fitted only on the complete development OOF probability vector.
- The outer test set was used once for final before/after evaluation.
- Lower Brier score means better probabilistic accuracy and reliability.

## Calibration comparison

| Evaluation set | Method | Brier score | ROC-AUC | PR-AUC | Expected responders |
|---|---|---:|---:|---:|---:|
"""
    for row in comparison_rows:
        report += f"| {row['evaluation_set']} | {row['method']} | {row['brier_score']:.6f} | {row['roc_auc']:.6f} | {row['pr_auc']:.6f} | {row['expected_responders_estimate']:.1f} |\n"
    report += f"""
## Outer test expected versus actual responders

- Actual responders: **{actual_test_responders:,}**
- Raw expected responders: **{np.sum(raw_test_probability):,.1f}**
- Sigmoid-calibrated expected responders: **{np.sum(sigmoid_test_probability):,.1f}**
- Isotonic-calibrated expected responders: **{np.sum(isotonic_test_probability):,.1f}**
- Selected calibrated expected responders (`{selected_method}`): **{np.sum(selected_test_probability):,.1f}**

Expected responders are sums of model probabilities, not guaranteed future
subscriptions. The actual responder count is used only for retrospective test
evaluation.

## Selected calibration method

Selected method: **{selected_method}**.

Selection required a lower development OOF Brier score and no more than 0.005
loss in development OOF PR-AUC or ROC-AUC versus raw probabilities. The
selected method improved development OOF
probability reliability relative to raw probabilities: **{oof_reliability_improved}**.

Calibration mappings are monotonic, so ranking order is expected to remain the
same or nearly the same. Ranking performance and probability reliability are
separate concerns: PR-AUC/ROC-AUC measure ordering, while Brier score and the
reliability diagram measure probability quality.

## Reliability findings

The reliability diagram was evaluated on the untouched outer test set. It
compares raw, sigmoid-calibrated, and isotonic-calibrated probabilities against
the observed response frequency. The calibrated model is intended for future
probability outputs only after this evaluation; the original raw ranking score
remains available for prioritization analysis.

## Files

- `{COMPARISON_CSV.relative_to(PROJECT_ROOT).as_posix()}`
- `{RESULTS_JSON.relative_to(PROJECT_ROOT).as_posix()}`
- `{TEST_OUTPUT_CSV.relative_to(PROJECT_ROOT).as_posix()}`
- `{RELIABILITY_FIGURE.relative_to(PROJECT_ROOT).as_posix()}`
- `{CALIBRATED_MODEL_PATH.relative_to(PROJECT_ROOT).as_posix()}`
- `{CALIBRATED_METADATA_PATH.relative_to(PROJECT_ROOT).as_posix()}`

The original `{BASE_MODEL_PATH.relative_to(PROJECT_ROOT).as_posix()}` was not modified.

## Phase 3.5 boundary

No FastAPI backend, frontend, deployment, URL generation, or Phase 4 work was performed.
"""
    REPORT_MD.write_text(report, encoding="utf-8")

    print(f"Generated {N_SPLITS}-fold development OOF probabilities.")
    print(f"Selected calibration method: {selected_method}")
    print(f"Outer test actual responders: {actual_test_responders}")
    print(f"Raw expected responders: {np.sum(raw_test_probability):.1f}")
    print(f"Selected calibrated expected responders: {np.sum(selected_test_probability):.1f}")
    print(f"Created report: {REPORT_MD}")


if __name__ == "__main__":
    main()

