# Phase 1: Data Validation, Preprocessing, and EDA

## Scope

This phase validates `bank-full.csv`, documents the leakage-safe pre-call
feature contract, and produces descriptive EDA. No predictive model was
trained, and no fitted model or preprocessing artifact was saved.

## Dataset profile

- Source: `bank-full.csv`
- Rows: **45,211**
- Columns: **17**
- Target: **`y`**
- Positive outcomes (`y=yes`): **5,289 (11.70%)**
- Negative outcomes (`y=no`): **39,922 (88.30%)**
- Numeric features after exclusions: `age, balance, day, campaign, pdays, previous`
- Categorical features after exclusions: `job, marital, education, default, housing, loan, contact, month, poutcome`

## Preprocessing decisions

1. `y` is kept strictly as the target and is removed before feature processing.
2. `duration` is excluded from the primary pre-call feature set because it is
   observed after a customer call and would create target leakage.
3. Numerical columns use median imputation followed by standardization.
4. Categorical columns use most-frequent imputation followed by one-hot
   encoding with `handle_unknown="ignore"` for robust future inference.
5. The transformer smoke test used an 80/20 stratified split. It was fitted on
   the training partition only and then applied to the holdout partition.
6. No predictive estimator was fitted during this phase.

## Data quality checks

- Missing values: **0** total missing cells.
- Blank-string values: **0** total blank cells.
- Literal `unknown` values: **52,124** cells. These are
  retained as an explicit category because they represent an observed source
  value rather than a blank cell.
- Rows belonging to duplicate groups: **0**.
- Duplicate records after the first occurrence: **0**.

## Leakage checks

- `y` present in feature columns: **No**.
- `duration` present in feature columns: **No**.
- Transformed feature count: **50**.
- Training transformed shape: **(36168, 50)**.
- Holdout transformed shape: **(9043, 50)**.
- Preprocessing fit scope: **training partition only**.

## EDA findings

- The target is imbalanced, with a subscription rate of **11.70%**. Later modeling should use stratification and should report PR-AUC and ranking metrics in addition to accuracy.
- Highest observed job-level subscription rates in this dataset:
  - `student`: 28.68% (269/938)
  - `retired`: 22.79% (516/2,264)
  - `unemployed`: 15.50% (202/1,303)
  - `management`: 13.76% (1,301/9,458)
  - `admin.`: 12.20% (631/5,171)

- Subscription rate by contact method:
  - `cellular`: 14.92% (4,369/29,285)
  - `telephone`: 13.42% (390/2,906)
  - `unknown`: 4.07% (530/13,020)

- Subscription rate by campaign month:
  - `mar`: 51.99% (248/477)
  - `dec`: 46.73% (100/214)
  - `sep`: 46.46% (269/579)
  - `oct`: 43.77% (323/738)
  - `apr`: 19.68% (577/2,932)
  - `feb`: 16.65% (441/2,649)
  - `aug`: 11.01% (688/6,247)
  - `jun`: 10.22% (546/5,341)
  - `nov`: 10.15% (403/3,970)
  - `jan`: 10.12% (142/1,403)
  - `jul`: 9.09% (627/6,895)
  - `may`: 6.72% (925/13,766)

- Previous campaign outcome is strongly informative descriptively, but it must be treated as an available pre-call feature only when the business process confirms that it is known before the new call.
- Repeated contacts should be examined carefully. The campaign-contact visualization is truncated to the first 10 contacts so that the main pattern remains readable; the raw data still contains all values.

## Generated visualizations

- `reports/figures/target_distribution.png`
- `reports/figures/age_by_outcome.png`
- `reports/figures/balance_signed_log_distribution.png`
- `reports/figures/response_rate_by_job.png`
- `reports/figures/response_rate_by_education.png`
- `reports/figures/response_rate_by_contact.png`
- `reports/figures/response_rate_by_month.png`
- `reports/figures/response_rate_by_poutcome.png`
- `reports/figures/response_rate_by_campaign_contacts.png`
- `reports/figures/numeric_correlation_matrix.png`

## Phase 1 boundary

No model training, backend development, frontend development, deployment, or fitted model saving was performed.
