# Phase 3: Customer Prioritization and Contact Capacity Analysis

## Scope and data controls

- Primary model: **Tuned Gradient Boosting** only.
- Source predictions: `phase2_test_predictions.csv` filtered to the final selected model.
- Held-out customers: **9,043** unique original row indexes.
- Held-out responders: **1,058**.
- Retrospective test response rate: **11.70%**.
- Actual outcomes are stored as `actual_y_retrospective` and are used only for retrospective evaluation.
- Future use should use `predicted_probability_yes`; it does not guarantee a subscription.

The held-out prediction rows were matched back to `bank-full.csv` using their
original row identifiers. All actual outcomes matched exactly. No full-dataset
outcomes were used for capacity evaluation.

## Capacity methodology

Capacity counts use `floor(N × capacity%)` so the selected customer count never
exceeds the requested calling capacity. At 100%, the full held-out test set is
selected. The random baseline uses **100** random permutations of the
same held-out test customers with fixed seed **42**. Random results
below are means across trials with sample standard deviations in the CSV/JSON
artifacts.

Expected responders are the sum of `predicted_probability_yes` for the
selected customers. They are probability-based estimates, not guaranteed
future subscriptions.

## Full capacity analysis table

| Capacity | Customers | Model actual responders | Model response rate | Model Precision@K | Model Recall@K | Model Lift@K | Model expected responders | Random actual responders mean | Random response rate mean | Random Precision@K mean | Random Recall@K mean | Random Lift@K mean | Random expected responders mean |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 5% | 452 | 279.0 | 61.73% | 61.73% | 26.37% | 5.28 | 402.9 | 52.4 | 11.59% | 11.59% | 4.95% | 0.99 | 181.0 |
| 10% | 904 | 464.0 | 51.33% | 51.33% | 43.86% | 4.39 | 745.8 | 105.6 | 11.68% | 11.68% | 9.98% | 1.00 | 362.5 |
| 20% | 1,808 | 649.0 | 35.90% | 35.90% | 61.34% | 3.07 | 1276.3 | 210.0 | 11.62% | 11.62% | 19.85% | 0.99 | 725.8 |
| 30% | 2,712 | 743.0 | 27.40% | 27.40% | 70.23% | 2.34 | 1717.6 | 317.3 | 11.70% | 11.70% | 29.99% | 1.00 | 1088.0 |
| 50% | 4,521 | 876.0 | 19.38% | 19.38% | 82.80% | 1.66 | 2453.7 | 529.2 | 11.71% | 11.71% | 50.02% | 1.00 | 1814.8 |
| 75% | 6,782 | 993.0 | 14.64% | 14.64% | 93.86% | 1.25 | 3162.4 | 794.6 | 11.72% | 11.72% | 75.10% | 1.00 | 2722.0 |
| 100% | 9,043 | 1058.0 | 11.70% | 11.70% | 100.00% | 1.00 | 3627.9 | 1058.0 | 11.70% | 11.70% | 100.00% | 1.00 | 3627.9 |

Random-baseline standard deviations for every metric are available in `phase3_capacity_analysis.csv`.

## Prioritization insights

- At 5% capacity, the model selects 452 customers and finds 279 retrospective responders, with 5.28× lift over the overall test response rate.
- At 20% capacity, the model captures 61.34% of all held-out responders while contacting only 1,808 customers.
- At 50% capacity, the model captures 82.80% of held-out responders. This illustrates the efficiency-versus-coverage trade-off: broader coverage finds more total responders but lowers the average response rate.
- Model-based Top-K selection improves efficiency by concentrating calls on customers with the highest estimated response probabilities.
- Increasing capacity generally increases the number of responders found and recall, while response rate and lift move toward the full-population baseline.
- Operationally, a bank can set K from available calling capacity, use the ranked list as a call queue, and monitor realized outcomes separately from the model’s probability estimates.
- The model supports prioritization and resource allocation; it does not guarantee that a customer will subscribe.

## Validation checks

- `held_out_customer_count`: **True**
- `held_out_unique_index_count`: **True**
- `actual_outcomes_match_source_by_index`: **True**
- `capacity_counts`: `{"5": 452, "10": 904, "20": 1808, "30": 2712, "50": 4521, "75": 6782, "100": 9043}`
- `capacity_counts_within_requested_limits`: **True**
- `model_recall_at_100_is_100_percent`: **True**
- `model_lift_at_100_is_one`: **True**
- `random_recall_at_100_is_100_percent`: **True**
- `random_lift_at_100_is_one`: **True**
- `model_full_capacity_actual_total_matches_test`: **True**
- `random_full_capacity_actual_total_matches_test`: **True**
- `model_full_capacity_response_rate_matches_test`: **True**
- `random_full_capacity_response_rate_matches_test`: **True**

## Generated artifacts

- `reports/phase3_ranked_customers.csv`
- `reports/phase3_capacity_analysis.csv`
- `reports/phase3_capacity_analysis.json`
- `reports/figures/phase3_capacity_vs_response_rate.png`
- `reports/figures/phase3_capacity_vs_actual_responders.png`
- `reports/figures/phase3_capacity_vs_recall.png`
- `reports/figures/phase3_model_vs_random.png`
- `reports/figures/phase3_cumulative_gains_lift.png`

## Phase 3 boundary

No FastAPI backend, frontend, deployment, URL generation, or Phase 4 work was performed.
