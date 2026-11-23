# Phase 2: Model Training, Evaluation, and Practical Tuning

## Scope and evaluation protocol

- Dataset: `bank-full.csv`
- Target: `y`
- Excluded leakage column: `duration`
- Random state: `42`
- Development rows: **36,168**
- Untouched test rows: **9,043**
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
| Logistic Regression | 0.7549 | 0.2663 | 0.6238 | 0.3733 | 0.7722 | 0.4092 | 6,167 | 1,818 | 398 | 660 |
| Decision Tree | 0.8325 | 0.3596 | 0.5529 | 0.4358 | 0.7686 | 0.3861 | 6,943 | 1,042 | 473 | 585 |
| Random Forest | 0.8952 | 0.6486 | 0.2268 | 0.3361 | 0.7901 | 0.4246 | 7,855 | 130 | 818 | 240 |
| Gradient Boosting | 0.7985 | 0.3220 | 0.6531 | 0.4313 | 0.7980 | 0.4442 | 6,530 | 1,455 | 367 | 691 |
| Tuned Gradient Boosting | 0.8033 | 0.3268 | 0.6427 | 0.4333 | 0.7998 | 0.4466 | 6,584 | 1,401 | 378 | 680 |

## Model selection

The baseline selected for tuning was **Gradient Boosting**, based on inner
validation PR-AUC, followed by ROC-AUC and F1 as tie-breakers. The tuned model
was selected without using the outer test metrics.

The final saved model is **Tuned Gradient Boosting**. It is appropriate for future
customer prioritization because it outputs a response probability and its
selection emphasizes ranking quality for the minority positive class rather
than accuracy alone. The saved Joblib artifact contains the fitted
preprocessing transformer and estimator in one Pipeline.

## Hyperparameter tuning

- Search scoring: average precision, which is the probability-based PR-AUC.
- Cross-validation: 2-fold stratified CV on the development set only, keeping the tuning budget practical for this project.
- Best parameters: `{"model__learning_rate": 0.1, "model__max_depth": 3, "model__n_estimators": 100}`
- Best mean CV PR-AUC: **0.4276**

| Rank | Parameters | Mean CV PR-AUC | Std. CV PR-AUC |
|---:|---|---:|---:|
| 1 | `{"model__learning_rate": 0.1, "model__max_depth": 3, "model__n_estimators": 100}` | 0.4276 | 0.0044 |
| 2 | `{"model__learning_rate": 0.1, "model__max_depth": 3, "model__n_estimators": 75}` | 0.4245 | 0.0024 |
| 3 | `{"model__learning_rate": 0.1, "model__max_depth": 2, "model__n_estimators": 100}` | 0.4200 | 0.0081 |
| 4 | `{"model__learning_rate": 0.05, "model__max_depth": 3, "model__n_estimators": 100}` | 0.4172 | 0.0009 |
| 5 | `{"model__learning_rate": 0.1, "model__max_depth": 2, "model__n_estimators": 75}` | 0.4148 | 0.0072 |
| 6 | `{"model__learning_rate": 0.05, "model__max_depth": 3, "model__n_estimators": 75}` | 0.4116 | 0.0009 |
| 7 | `{"model__learning_rate": 0.05, "model__max_depth": 2, "model__n_estimators": 100}` | 0.4056 | 0.0048 |
| 8 | `{"model__learning_rate": 0.05, "model__max_depth": 2, "model__n_estimators": 75}` | 0.3971 | 0.0049 |

## Feature importance findings

Native importance analysis for **Tuned Gradient Boosting** was generated from the fitted development model. It is descriptive and does not imply causality.

- `poutcome_success`: 0.275440
- `contact_unknown`: 0.158128
- `age`: 0.065769
- `day`: 0.054266
- `balance`: 0.050738
- `housing_yes`: 0.044907
- `pdays`: 0.036697
- `campaign`: 0.032774
- `month_mar`: 0.032432
- `month_oct`: 0.032262
- `housing_no`: 0.031736
- `month_jun`: 0.023498
- `month_apr`: 0.020434
- `marital_married`: 0.018691
- `month_sep`: 0.012875

## Saved artifacts

- `reports/phase2_model_comparison.csv`
- `reports/phase2_model_results.json`
- `reports/phase2_test_predictions.csv` containing each held-out test row, actual outcome, predicted probability, and 0.5-threshold prediction for every evaluated model
- `models/best_model.joblib` containing the complete fitted preprocessing-plus-estimator pipeline
- `models/best_model_metadata.json`
- `reports/figures/phase2_feature_importance_tuned_gradient_boosting.png`

## Phase 2 boundary

No FastAPI backend, frontend, customer prioritization UI, deployment, or public URL was created.
