# Phase 3.5: Probability Calibration and Evaluation

## Methodology

- Base model: the unchanged Phase 2 Tuned Gradient Boosting pipeline.
- Outer test set: **9,043** customers, never used for calibration fitting or method selection.
- Development set: **36,168** customers.
- Development calibration predictions: **5-fold stratified out-of-fold probabilities**.
- In each fold, preprocessing and the base estimator were fitted only on the fold-training rows.
- Sigmoid/Platt and isotonic mappings were fitted only on the complete development OOF probability vector.
- The outer test set was used once for final before/after evaluation.
- Lower Brier score means better probabilistic accuracy and reliability.

## Calibration comparison

| Evaluation set | Method | Brier score | ROC-AUC | PR-AUC | Expected responders |
|---|---|---:|---:|---:|---:|
| development_oof | raw | 0.170011 | 0.792552 | 0.431630 | 14540.3 |
| development_oof | sigmoid | 0.082722 | 0.792552 | 0.431630 | 4230.0 |
| development_oof | isotonic | 0.082136 | 0.793941 | 0.424575 | 4231.0 |
| outer_test_final | raw | 0.168232 | 0.799842 | 0.446556 | 3627.9 |
| outer_test_final | sigmoid | 0.081487 | 0.799842 | 0.446556 | 1053.1 |
| outer_test_final | isotonic | 0.081563 | 0.799424 | 0.432902 | 1048.4 |

## Outer test expected versus actual responders

- Actual responders: **1,058**
- Raw expected responders: **3,627.9**
- Sigmoid-calibrated expected responders: **1,053.1**
- Isotonic-calibrated expected responders: **1,048.4**
- Selected calibrated expected responders (`sigmoid`): **1,053.1**

Expected responders are sums of model probabilities, not guaranteed future
subscriptions. The actual responder count is used only for retrospective test
evaluation.

## Selected calibration method

Selected method: **sigmoid**.

Selection required a lower development OOF Brier score and no more than 0.005
loss in development OOF PR-AUC or ROC-AUC versus raw probabilities. The
selected method improved development OOF
probability reliability relative to raw probabilities: **True**.

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

- `reports/phase3_5_calibration_comparison.csv`
- `reports/phase3_5_calibration_results.json`
- `reports/phase3_5_test_calibrated_predictions.csv`
- `reports/figures/phase3_5_reliability_diagram.png`
- `models/calibrated_best_model.joblib`
- `models/calibrated_model_metadata.json`

The original `models/best_model.joblib` was not modified.

## Phase 3.5 boundary

No FastAPI backend, frontend, deployment, URL generation, or Phase 4 work was performed.
