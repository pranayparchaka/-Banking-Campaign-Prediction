"""Phase 1 validation, leakage-safe preprocessing smoke test, and EDA.

This script does not train a predictive model. It creates descriptive reports
and figures from bank-full.csv, and verifies that the preprocessing transformer
can be fitted on a training partition without using the holdout partition.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


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
REPORT_PATH = PROJECT_ROOT / "reports" / "phase1_data_validation_eda.md"
PROFILE_PATH = PROJECT_ROOT / "reports" / "phase1_profile.json"
FIGURE_DIR = PROJECT_ROOT / "reports" / "figures"


def save_figure(name: str) -> str:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    output_path = FIGURE_DIR / name
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    return str(output_path.relative_to(PROJECT_ROOT)).replace("\\", "/")


def rate_table(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    grouped = (
        frame.groupby(column, dropna=False)[TARGET_COLUMN]
        .agg(responders=lambda values: (values == "yes").sum(), customers="size")
        .reset_index()
    )
    grouped["response_rate"] = grouped["responders"] / grouped["customers"]
    return grouped.sort_values("response_rate", ascending=False)


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    frame = pd.read_csv(DATA_PATH, sep=";", quotechar='"')
    expected_columns = [
        "age", "job", "marital", "education", "default", "balance", "housing",
        "loan", "contact", "day", "month", "duration", "campaign", "pdays",
        "previous", "poutcome", "y",
    ]
    missing_expected_columns = [c for c in expected_columns if c not in frame.columns]
    unexpected_columns = [c for c in frame.columns if c not in expected_columns]
    if missing_expected_columns:
        raise ValueError(f"Missing expected columns: {missing_expected_columns}")

    features, target = prepare_features_target(frame)
    numeric_columns, categorical_columns = infer_feature_types(features)

    # Descriptive checks are allowed on the complete source. The transformer
    # smoke test below is intentionally fitted only on X_train.
    missing_counts = frame.isna().sum()
    blank_counts = (frame.astype("string").apply(lambda s: s.str.strip().eq(""))).sum()
    unknown_counts = (frame.astype("string").apply(lambda s: s.str.lower().eq("unknown"))).sum()
    duplicate_rows = int(frame.duplicated(keep=False).sum())
    duplicate_records = int(frame.duplicated(keep="first").sum())
    target_counts = target.value_counts(dropna=False).to_dict()
    positive_count = int((target == "yes").sum())
    negative_count = int((target == "no").sum())

    X_train, X_holdout, y_train, y_holdout = train_test_split(
        features,
        target,
        test_size=0.20,
        random_state=42,
        stratify=target,
    )
    preprocessor = build_preprocessor(numeric_columns, categorical_columns)
    train_transformed = preprocessor.fit_transform(X_train)
    holdout_transformed = preprocessor.transform(X_holdout)
    transformed_feature_names = preprocessor.get_feature_names_out().tolist()
    if TARGET_COLUMN in transformed_feature_names:
        raise AssertionError("Target y appeared in transformed feature names.")
    if any(column in transformed_feature_names for column in LEAKAGE_COLUMNS):
        raise AssertionError("duration appeared in transformed feature names.")

    figures: list[str] = []

    # Target distribution.
    plt.figure(figsize=(6, 4))
    counts = frame[TARGET_COLUMN].value_counts().reindex(["no", "yes"])
    plt.bar(counts.index, counts.values, color=["#486581", "#2f855a"])
    plt.title("Subscription outcome distribution")
    plt.xlabel("Term deposit subscription")
    plt.ylabel("Customers")
    figures.append(save_figure("target_distribution.png"))

    # Age distribution by outcome.
    plt.figure(figsize=(8, 4))
    for label, color in [("no", "#486581"), ("yes", "#2f855a")]:
        frame.loc[frame[TARGET_COLUMN] == label, "age"].plot(
            kind="hist", bins=30, alpha=0.55, label=label, color=color
        )
    plt.title("Age distribution by subscription outcome")
    plt.xlabel("Age")
    plt.ylabel("Customers")
    plt.legend(title="Outcome")
    figures.append(save_figure("age_by_outcome.png"))

    # Balance distribution with a symmetric log scale to retain negative and
    # high-value observations.
    plt.figure(figsize=(8, 4))
    signed_log_balance = np.sign(frame["balance"]) * np.log1p(np.abs(frame["balance"]))
    plt.hist(signed_log_balance, bins=45, color="#7b61a8", alpha=0.85)
    plt.title("Signed log-transformed account balance")
    plt.xlabel("sign(balance) × log(1 + |balance|)")
    plt.ylabel("Customers")
    figures.append(save_figure("balance_signed_log_distribution.png"))

    # Response rates for important categorical dimensions.
    categorical_plot_columns = ["job", "education", "contact", "month", "poutcome"]
    for column in categorical_plot_columns:
        table = rate_table(frame, column).sort_values("response_rate")
        plt.figure(figsize=(8, max(4, 0.35 * len(table))))
        plt.barh(table[column].astype(str), table["response_rate"] * 100, color="#2f855a")
        plt.title(f"Subscription rate by {column}")
        plt.xlabel("Subscription rate (%)")
        plt.ylabel(column)
        figures.append(save_figure(f"response_rate_by_{column}.png"))

    # Campaign contacts versus outcome.
    campaign_table = (
        frame.assign(contacted_yes=frame[TARGET_COLUMN].eq("yes"))
        .groupby("campaign", as_index=False)
        .agg(customers=(TARGET_COLUMN, "size"), responders=("contacted_yes", "sum"))
    )
    campaign_table["response_rate"] = campaign_table["responders"] / campaign_table["customers"]
    campaign_plot = campaign_table[campaign_table["campaign"] <= 10]
    plt.figure(figsize=(8, 4))
    plt.plot(campaign_plot["campaign"], campaign_plot["response_rate"] * 100, marker="o", color="#d97706")
    plt.title("Subscription rate by number of campaign contacts")
    plt.xlabel("Campaign contacts")
    plt.ylabel("Subscription rate (%)")
    plt.grid(alpha=0.25)
    figures.append(save_figure("response_rate_by_campaign_contacts.png"))

    # Numerical correlation heatmap implemented with matplotlib only.
    numerical_frame = frame.select_dtypes(include=["number"])
    correlation = numerical_frame.corr(numeric_only=True)
    plt.figure(figsize=(8, 6))
    image = plt.imshow(correlation, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar(image, fraction=0.046, pad=0.04)
    plt.xticks(range(len(correlation.columns)), correlation.columns, rotation=45, ha="right")
    plt.yticks(range(len(correlation.columns)), correlation.columns)
    plt.title("Numerical feature correlation matrix")
    figures.append(save_figure("numeric_correlation_matrix.png"))

    profile = {
        "dataset": {
            "path": str(DATA_PATH.relative_to(PROJECT_ROOT)),
            "rows": int(frame.shape[0]),
            "columns": int(frame.shape[1]),
            "column_names": frame.columns.tolist(),
            "expected_columns_missing": missing_expected_columns,
            "unexpected_columns": unexpected_columns,
            "dtypes": {column: str(dtype) for column, dtype in frame.dtypes.items()},
        },
        "target": {
            "column": TARGET_COLUMN,
            "counts": {str(key): int(value) for key, value in target_counts.items()},
            "positive_count": positive_count,
            "negative_count": negative_count,
            "positive_rate": positive_count / len(frame),
        },
        "quality": {
            "missing_by_column": {column: int(value) for column, value in missing_counts.items() if value},
            "blank_by_column": {column: int(value) for column, value in blank_counts.items() if value},
            "unknown_by_column": {column: int(value) for column, value in unknown_counts.items() if value},
            "duplicate_rows_in_duplicate_groups": duplicate_rows,
            "duplicate_records_after_first_occurrence": duplicate_records,
        },
        "preprocessing": {
            "target_column": TARGET_COLUMN,
            "excluded_leakage_columns": list(LEAKAGE_COLUMNS),
            "numeric_columns": numeric_columns,
            "categorical_columns": categorical_columns,
            "train_rows_for_smoke_test": int(len(X_train)),
            "holdout_rows_for_smoke_test": int(len(X_holdout)),
            "transformed_feature_count": len(transformed_feature_names),
            "train_transformed_shape": list(train_transformed.shape),
            "holdout_transformed_shape": list(holdout_transformed.shape),
            "fit_scope": "training partition only; no fitted transformer is saved in Phase 1",
        },
        "eda": {
            "top_job_response_rates": rate_table(frame, "job").head(5).to_dict(orient="records"),
            "top_contact_response_rates": rate_table(frame, "contact").to_dict(orient="records"),
            "top_month_response_rates": rate_table(frame, "month").to_dict(orient="records"),
            "top_poutcome_response_rates": rate_table(frame, "poutcome").to_dict(orient="records"),
            "figures": figures,
        },
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    PROFILE_PATH.parent.mkdir(parents=True, exist_ok=True)
    PROFILE_PATH.write_text(json.dumps(profile, indent=2), encoding="utf-8")

    def fmt_rate(value: float) -> str:
        return f"{value:.2%}"

    job_rates = rate_table(frame, "job")
    contact_rates = rate_table(frame, "contact")
    month_rates = rate_table(frame, "month")
    poutcome_rates = rate_table(frame, "poutcome")
    report = f"""# Phase 1: Data Validation, Preprocessing, and EDA

## Scope

This phase validates `bank-full.csv`, documents the leakage-safe pre-call
feature contract, and produces descriptive EDA. No predictive model was
trained, and no fitted model or preprocessing artifact was saved.

## Dataset profile

- Source: `{DATA_PATH.name}`
- Rows: **{len(frame):,}**
- Columns: **{frame.shape[1]}**
- Target: **`{TARGET_COLUMN}`**
- Positive outcomes (`y=yes`): **{positive_count:,} ({positive_count / len(frame):.2%})**
- Negative outcomes (`y=no`): **{negative_count:,} ({negative_count / len(frame):.2%})**
- Numeric features after exclusions: `{', '.join(numeric_columns)}`
- Categorical features after exclusions: `{', '.join(categorical_columns)}`

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

- Missing values: **{int(missing_counts.sum()):,}** total missing cells.
- Blank-string values: **{int(blank_counts.sum()):,}** total blank cells.
- Literal `unknown` values: **{int(unknown_counts.sum()):,}** cells. These are
  retained as an explicit category because they represent an observed source
  value rather than a blank cell.
- Rows belonging to duplicate groups: **{duplicate_rows:,}**.
- Duplicate records after the first occurrence: **{duplicate_records:,}**.

## Leakage checks

- `y` present in feature columns: **No**.
- `duration` present in feature columns: **No**.
- Transformed feature count: **{len(transformed_feature_names):,}**.
- Training transformed shape: **{train_transformed.shape}**.
- Holdout transformed shape: **{holdout_transformed.shape}**.
- Preprocessing fit scope: **training partition only**.

## EDA findings

- The target is imbalanced, with a subscription rate of **{positive_count / len(frame):.2%}**. Later modeling should use stratification and should report PR-AUC and ranking metrics in addition to accuracy.
- Highest observed job-level subscription rates in this dataset:
"""
    for row in job_rates.head(5).itertuples(index=False):
        report += f"  - `{row.job}`: {fmt_rate(row.response_rate)} ({row.responders:,}/{row.customers:,})\n"
    report += "\n- Subscription rate by contact method:\n"
    for row in contact_rates.itertuples(index=False):
        report += f"  - `{row.contact}`: {fmt_rate(row.response_rate)} ({row.responders:,}/{row.customers:,})\n"
    report += "\n- Subscription rate by campaign month:\n"
    for row in month_rates.itertuples(index=False):
        report += f"  - `{row.month}`: {fmt_rate(row.response_rate)} ({row.responders:,}/{row.customers:,})\n"
    report += "\n- Previous campaign outcome is strongly informative descriptively, but it must be treated as an available pre-call feature only when the business process confirms that it is known before the new call.\n"
    report += "- Repeated contacts should be examined carefully. The campaign-contact visualization is truncated to the first 10 contacts so that the main pattern remains readable; the raw data still contains all values.\n"
    report += "\n## Generated visualizations\n\n"
    for figure in figures:
        report += f"- `{figure}`\n"
    report += "\n## Phase 1 boundary\n\nNo model training, backend development, frontend development, deployment, or fitted model saving was performed.\n"
    REPORT_PATH.write_text(report, encoding="utf-8")

    print(f"Validated {len(frame):,} rows and {frame.shape[1]} columns.")
    print(f"Created report: {REPORT_PATH}")
    print(f"Created profile: {PROFILE_PATH}")
    print(f"Created {len(figures)} figures in {FIGURE_DIR}")


if __name__ == "__main__":
    main()

