"""Phase 3 customer prioritization and contact-capacity analysis.

This phase uses only the held-out test rows and only the Tuned Gradient
Boosting probabilities preserved by Phase 2. Actual outcomes are retained in
the output only as retrospective evaluation labels.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "bank-full.csv"
TEST_PREDICTIONS_PATH = PROJECT_ROOT / "reports" / "phase2_test_predictions.csv"
PHASE2_METADATA_PATH = PROJECT_ROOT / "models" / "best_model_metadata.json"
REPORT_DIR = PROJECT_ROOT / "reports"
FIGURE_DIR = REPORT_DIR / "figures"
RANKED_PATH = REPORT_DIR / "phase3_ranked_customers.csv"
CAPACITY_CSV_PATH = REPORT_DIR / "phase3_capacity_analysis.csv"
CAPACITY_JSON_PATH = REPORT_DIR / "phase3_capacity_analysis.json"
REPORT_PATH = REPORT_DIR / "phase3_prioritization_capacity.md"

RANDOM_SEED = 42
RANDOM_TRIALS = 100
CAPACITY_PCTS = [5, 10, 20, 30, 50, 75, 100]
FINAL_MODEL_NAME = "Tuned Gradient Boosting"


def metrics_for_selection(
    selected: pd.DataFrame,
    total_test_customers: int,
    total_test_responders: int,
    overall_response_rate: float,
) -> dict[str, float | int]:
    contacted = len(selected)
    actual_responders = int(selected["actual_y_retrospective"].eq("yes").sum())
    expected_responders = float(selected["predicted_probability_yes"].sum())
    response_rate = actual_responders / contacted if contacted else 0.0
    precision_at_k = actual_responders / contacted if contacted else 0.0
    recall_at_k = actual_responders / total_test_responders if total_test_responders else 0.0
    lift_at_k = response_rate / overall_response_rate if overall_response_rate else 0.0
    return {
        "customers_contacted": int(contacted),
        "actual_responders_found": actual_responders,
        "response_rate": float(response_rate),
        "precision_at_k": float(precision_at_k),
        "recall_at_k": float(recall_at_k),
        "lift_at_k": float(lift_at_k),
        "expected_responders_estimate": expected_responders,
    }


def save_capacity_plot(capacity_df: pd.DataFrame, filename: str, y_columns: list[str], title: str, ylabel: str) -> str:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURE_DIR / filename
    fig, ax = plt.subplots(figsize=(8, 5))
    for column, label, color, marker in [
        (y_columns[0], "Model-based Top-K", "#2f855a", "o"),
        (y_columns[1], "Random baseline mean", "#718096", "s"),
    ]:
        if column in y_columns:
            ax.plot(capacity_df["capacity_pct"], capacity_df[column], marker=marker, label=label, color=color)
    ax.set_title(title)
    ax.set_xlabel("Calling capacity (%)")
    ax.set_ylabel(ylabel)
    ax.set_xticks(CAPACITY_PCTS)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(path.relative_to(PROJECT_ROOT)).replace("\\", "/")


def main() -> None:
    for required_path in [DATA_PATH, TEST_PREDICTIONS_PATH, PHASE2_METADATA_PATH]:
        if not required_path.exists():
            raise FileNotFoundError(f"Required Phase 2 artifact not found: {required_path}")

    full_frame = pd.read_csv(DATA_PATH, sep=";", quotechar='"')
    prediction_frame = pd.read_csv(TEST_PREDICTIONS_PATH)
    metadata = json.loads(PHASE2_METADATA_PATH.read_text(encoding="utf-8"))
    if metadata.get("best_model") != FINAL_MODEL_NAME:
        raise ValueError(
            f"Phase 2 metadata identifies '{metadata.get('best_model')}', expected '{FINAL_MODEL_NAME}'."
        )

    tuned = prediction_frame.loc[prediction_frame["model"].eq(FINAL_MODEL_NAME)].copy()
    if tuned.empty:
        raise ValueError("No Tuned Gradient Boosting rows found in phase2_test_predictions.csv.")
    if not prediction_frame["model"].eq(FINAL_MODEL_NAME).any():
        raise AssertionError("The Phase 2 prediction artifact does not contain the final selected model.")
    if tuned["row_id"].duplicated().any():
        raise AssertionError("Duplicate held-out row identifiers found for Tuned Gradient Boosting.")

    # train_test_split uses ceil for a fractional test_size, so 20% of 45,211
    # produces 9,043 held-out rows.
    expected_test_count = int(np.ceil(len(full_frame) * 0.20))
    if len(tuned) != expected_test_count:
        raise AssertionError(
            f"Expected {expected_test_count} held-out customers, found {len(tuned)} for Tuned Gradient Boosting."
        )
    if tuned["row_id"].isna().any() or not pd.api.types.is_integer_dtype(tuned["row_id"]):
        raise AssertionError("Held-out row identifiers must be non-null integer indexes.")
    held_out_ids = tuned["row_id"].astype(int).to_numpy()
    if len(np.unique(held_out_ids)) != len(held_out_ids):
        raise AssertionError("Held-out row identifiers are not unique.")
    if held_out_ids.min() < 0 or held_out_ids.max() >= len(full_frame):
        raise AssertionError("A held-out row identifier falls outside bank-full.csv index bounds.")

    # Match the Phase 2 artifact to the original CSV using the original index.
    matched_source = full_frame.iloc[held_out_ids].copy()
    matched_source.index = held_out_ids
    tuned.index = held_out_ids
    tuned_actual = tuned["actual_y"].astype(str).to_numpy()
    source_actual = matched_source["y"].astype(str).to_numpy()
    if not np.array_equal(tuned_actual, source_actual):
        raise AssertionError("Phase 2 actual outcomes do not match bank-full.csv at the original row indexes.")

    pre_call_attributes = matched_source.drop(columns=["duration", "y"]).copy()
    pre_call_attributes.insert(0, "customer_index", held_out_ids)
    ranked = pre_call_attributes.copy()
    ranked["actual_y_retrospective"] = source_actual
    ranked["predicted_probability_yes"] = tuned["predicted_probability_yes"].astype(float).to_numpy()
    ranked = ranked.sort_values(
        ["predicted_probability_yes", "customer_index"], ascending=[False, True], kind="mergesort"
    ).reset_index(drop=True)
    ranked.insert(1, "priority_rank", np.arange(1, len(ranked) + 1))
    ranked["actual_y_retrospective"] = ranked["actual_y_retrospective"].astype(str)
    ranked.to_csv(RANKED_PATH, index=False)

    total_test_customers = len(ranked)
    total_test_responders = int(ranked["actual_y_retrospective"].eq("yes").sum())
    overall_response_rate = total_test_responders / total_test_customers
    total_expected_responders = float(ranked["predicted_probability_yes"].sum())

    # Capacity counts are floors so the number contacted never exceeds capacity;
    # 100% is explicitly set to the full held-out test population.
    capacity_counts = {
        pct: total_test_customers if pct == 100 else int(np.floor(total_test_customers * pct / 100))
        for pct in CAPACITY_PCTS
    }
    if any(capacity_counts[pct] < 1 for pct in CAPACITY_PCTS):
        raise AssertionError("A requested capacity produced zero customers.")
    if capacity_counts[100] != total_test_customers:
        raise AssertionError("100% capacity does not equal the full held-out test set.")

    random_generator = np.random.default_rng(RANDOM_SEED)
    model_rows: list[dict[str, object]] = []
    random_rows: list[dict[str, object]] = []
    random_trial_rows: list[dict[str, object]] = []
    cumulative_model_actual: list[int] = []
    cumulative_random_actual: list[float] = []

    for pct in CAPACITY_PCTS:
        k = capacity_counts[pct]
        model_selected = ranked.iloc[:k]
        model_metrics = metrics_for_selection(
            model_selected, total_test_customers, total_test_responders, overall_response_rate
        )
        model_row = {
            "capacity_pct": pct,
            "selection_method": "model_based_top_k",
            "random_trial": None,
            **model_metrics,
        }
        model_rows.append(model_row)
        cumulative_model_actual.append(int(model_metrics["actual_responders_found"]))

        trial_metrics: list[dict[str, float | int]] = []
        for trial in range(1, RANDOM_TRIALS + 1):
            permutation = random_generator.permutation(total_test_customers)
            random_selected = ranked.iloc[permutation[:k]]
            metrics = metrics_for_selection(
                random_selected, total_test_customers, total_test_responders, overall_response_rate
            )
            trial_metrics.append(metrics)
            random_trial_rows.append({
                "capacity_pct": pct,
                "selection_method": "random_baseline",
                "random_trial": trial,
                **metrics,
            })
        trial_frame = pd.DataFrame(trial_metrics)
        means = trial_frame.mean(numeric_only=True)
        stds = trial_frame.std(numeric_only=True, ddof=1).fillna(0.0)
        random_row = {
            "capacity_pct": pct,
            "selection_method": "random_baseline_mean",
            "random_trial": RANDOM_TRIALS,
            **{column: float(means[column]) for column in trial_frame.columns},
            **{f"{column}_std": float(stds[column]) for column in trial_frame.columns},
        }
        random_rows.append(random_row)
        cumulative_random_actual.append(float(means["actual_responders_found"]))

    capacity_rows = []
    for model_row, random_row in zip(model_rows, random_rows):
        row = {"capacity_pct": model_row["capacity_pct"], "customers_contacted": model_row["customers_contacted"]}
        for metric in [
            "actual_responders_found", "response_rate", "precision_at_k", "recall_at_k",
            "lift_at_k", "expected_responders_estimate",
        ]:
            row[f"model_{metric}"] = model_row[metric]
            row[f"random_mean_{metric}"] = random_row[metric]
            row[f"random_std_{metric}"] = random_row[f"{metric}_std"]
        capacity_rows.append(row)
    capacity_df = pd.DataFrame(capacity_rows)

    # Required full-capacity invariants.
    full_row = capacity_df.loc[capacity_df["capacity_pct"].eq(100)].iloc[0]
    expected_full_rate = overall_response_rate
    checks = {
        "held_out_customer_count": len(tuned) == expected_test_count,
        "held_out_unique_index_count": tuned["row_id"].nunique() == expected_test_count,
        "actual_outcomes_match_source_by_index": bool(np.array_equal(tuned_actual, source_actual)),
        "capacity_counts": {str(pct): int(capacity_counts[pct]) for pct in CAPACITY_PCTS},
        "capacity_counts_within_requested_limits": all(
            capacity_counts[pct] <= total_test_customers * pct / 100 + 1e-9 for pct in CAPACITY_PCTS
        ),
        "model_recall_at_100_is_100_percent": bool(np.isclose(full_row["model_recall_at_k"], 1.0)),
        "model_lift_at_100_is_one": bool(np.isclose(full_row["model_lift_at_k"], 1.0)),
        "random_recall_at_100_is_100_percent": bool(np.isclose(full_row["random_mean_recall_at_k"], 1.0)),
        "random_lift_at_100_is_one": bool(np.isclose(full_row["random_mean_lift_at_k"], 1.0)),
        "model_full_capacity_actual_total_matches_test": int(full_row["model_actual_responders_found"]) == total_test_responders,
        "random_full_capacity_actual_total_matches_test": np.isclose(
            full_row["random_mean_actual_responders_found"], total_test_responders
        ),
        "model_full_capacity_response_rate_matches_test": np.isclose(
            full_row["model_response_rate"], expected_full_rate
        ),
        "random_full_capacity_response_rate_matches_test": np.isclose(
            full_row["random_mean_response_rate"], expected_full_rate
        ),
    }
    def check_pass(value: object) -> bool:
        if isinstance(value, dict):
            return all(bool(item) for item in value.values())
        return bool(value)

    if not all(check_pass(value) for value in checks.values()):
        raise AssertionError(f"Phase 3 validation checks failed: {checks}")

    capacity_df.to_csv(CAPACITY_CSV_PATH, index=False)

    # Plot 1: capacity versus response rate.
    figure_paths = []
    figure_paths.append(save_capacity_plot(
        capacity_df,
        "phase3_capacity_vs_response_rate.png",
        ["model_response_rate", "random_mean_response_rate"],
        "Calling capacity versus retrospective response rate",
        "Response rate",
    ))
    figure_paths.append(save_capacity_plot(
        capacity_df,
        "phase3_capacity_vs_actual_responders.png",
        ["model_actual_responders_found", "random_mean_actual_responders_found"],
        "Calling capacity versus actual responders found",
        "Actual responders found",
    ))
    figure_paths.append(save_capacity_plot(
        capacity_df,
        "phase3_capacity_vs_recall.png",
        ["model_recall_at_k", "random_mean_recall_at_k"],
        "Calling capacity versus retrospective recall",
        "Recall@K",
    ))

    # Plot 4: model versus random selection using response rate and responders.
    path = FIGURE_DIR / "phase3_model_vs_random.png"
    x = np.arange(len(CAPACITY_PCTS))
    width = 0.36
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].bar(x - width / 2, capacity_df["model_response_rate"] * 100, width, label="Model-based", color="#2f855a")
    axes[0].bar(x + width / 2, capacity_df["random_mean_response_rate"] * 100, width, label="Random mean", color="#718096")
    axes[0].set_title("Response rate")
    axes[0].set_ylabel("Response rate (%)")
    axes[1].bar(x - width / 2, capacity_df["model_actual_responders_found"], width, label="Model-based", color="#2f855a")
    axes[1].bar(x + width / 2, capacity_df["random_mean_actual_responders_found"], width, label="Random mean", color="#718096")
    axes[1].set_title("Actual responders found")
    axes[1].set_ylabel("Responders")
    for axis in axes:
        axis.set_xticks(x, [f"{pct}%" for pct in CAPACITY_PCTS])
        axis.grid(axis="y", alpha=0.25)
    axes[0].legend()
    fig.suptitle("Model-based Top-K selection versus random selection")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    figure_paths.append(str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"))

    # Plot 5: cumulative gains and lift curve for the model, with random line.
    sorted_actual = ranked["actual_y_retrospective"].eq("yes").astype(int).to_numpy()
    cumulative_actual = np.cumsum(sorted_actual)
    population_fraction = np.arange(1, total_test_customers + 1) / total_test_customers
    cumulative_gain = cumulative_actual / total_test_responders
    lift_curve = np.divide(cumulative_gain, population_fraction, out=np.zeros_like(cumulative_gain, dtype=float), where=population_fraction > 0)
    path = FIGURE_DIR / "phase3_cumulative_gains_lift.png"
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(population_fraction * 100, cumulative_gain * 100, color="#2f855a", label="Model-based gains")
    axes[0].plot([0, 100], [0, 100], linestyle="--", color="#718096", label="Random baseline")
    axes[0].set_title("Cumulative gains")
    axes[0].set_xlabel("Customers contacted (%)")
    axes[0].set_ylabel("Responders captured (%)")
    axes[1].plot(population_fraction * 100, lift_curve, color="#d97706", label="Model lift")
    axes[1].axhline(1.0, linestyle="--", color="#718096", label="Random baseline")
    axes[1].set_title("Lift curve")
    axes[1].set_xlabel("Customers contacted (%)")
    axes[1].set_ylabel("Lift")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    figure_paths.append(str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"))

    checks_json = {
        "capacity_counts": {str(key): int(value) for key, value in capacity_counts.items()},
        **{
            key: bool(value)
            for key, value in checks.items()
            if key != "capacity_counts"
        },
    }
    result_json = {
        "source_dataset": str(DATA_PATH.relative_to(PROJECT_ROOT)),
        "source_predictions": str(TEST_PREDICTIONS_PATH.relative_to(PROJECT_ROOT)),
        "model_used": FINAL_MODEL_NAME,
        "actual_outcome_field": "actual_y_retrospective",
        "prediction_field": "predicted_probability_yes",
        "random_seed": RANDOM_SEED,
        "random_trials": RANDOM_TRIALS,
        "random_methodology": "For each trial, randomly permute the exact held-out test customers and select the first K customers at each capacity. Report the mean and sample standard deviation over 100 reproducible trials.",
        "capacity_count_method": "floor(N * capacity_pct / 100), with 100% explicitly equal to N so capacity is never exceeded.",
        "total_test_customers": total_test_customers,
        "total_test_responders": total_test_responders,
        "overall_test_response_rate": overall_response_rate,
        "total_expected_responders_estimate": total_expected_responders,
        "validation_checks": checks_json,
        "capacity_analysis": capacity_df.to_dict(orient="records"),
        "figures": figure_paths,
    }
    CAPACITY_JSON_PATH.write_text(json.dumps(result_json, indent=2), encoding="utf-8")

    report = f"""# Phase 3: Customer Prioritization and Contact Capacity Analysis

## Scope and data controls

- Primary model: **{FINAL_MODEL_NAME}** only.
- Source predictions: `{TEST_PREDICTIONS_PATH.name}` filtered to the final selected model.
- Held-out customers: **{total_test_customers:,}** unique original row indexes.
- Held-out responders: **{total_test_responders:,}**.
- Retrospective test response rate: **{overall_response_rate:.2%}**.
- Actual outcomes are stored as `actual_y_retrospective` and are used only for retrospective evaluation.
- Future use should use `predicted_probability_yes`; it does not guarantee a subscription.

The held-out prediction rows were matched back to `bank-full.csv` using their
original row identifiers. All actual outcomes matched exactly. No full-dataset
outcomes were used for capacity evaluation.

## Capacity methodology

Capacity counts use `floor(N × capacity%)` so the selected customer count never
exceeds the requested calling capacity. At 100%, the full held-out test set is
selected. The random baseline uses **{RANDOM_TRIALS}** random permutations of the
same held-out test customers with fixed seed **{RANDOM_SEED}**. Random results
below are means across trials with sample standard deviations in the CSV/JSON
artifacts.

Expected responders are the sum of `predicted_probability_yes` for the
selected customers. They are probability-based estimates, not guaranteed
future subscriptions.

## Full capacity analysis table

| Capacity | Customers | Model actual responders | Model response rate | Model Precision@K | Model Recall@K | Model Lift@K | Model expected responders | Random actual responders mean | Random response rate mean | Random Precision@K mean | Random Recall@K mean | Random Lift@K mean | Random expected responders mean |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
"""
    for row in capacity_rows:
        report += (
            f"| {row['capacity_pct']}% | {row['customers_contacted']:,} | {row['model_actual_responders_found']:.1f} | "
            f"{row['model_response_rate']:.2%} | {row['model_precision_at_k']:.2%} | {row['model_recall_at_k']:.2%} | "
            f"{row['model_lift_at_k']:.2f} | {row['model_expected_responders_estimate']:.1f} | "
            f"{row['random_mean_actual_responders_found']:.1f} | {row['random_mean_response_rate']:.2%} | "
            f"{row['random_mean_precision_at_k']:.2%} | {row['random_mean_recall_at_k']:.2%} | "
            f"{row['random_mean_lift_at_k']:.2f} | {row['random_mean_expected_responders_estimate']:.1f} |\n"
        )
    report += "\nRandom-baseline standard deviations for every metric are available in `phase3_capacity_analysis.csv`.\n"
    report += "\n## Prioritization insights\n\n"
    first_model = capacity_rows[0]
    third_model = capacity_rows[2]
    half_model = capacity_rows[4]
    report += f"- At 5% capacity, the model selects {int(first_model['customers_contacted']):,} customers and finds {int(first_model['model_actual_responders_found']):,} retrospective responders, with {first_model['model_lift_at_k']:.2f}× lift over the overall test response rate.\n"
    report += f"- At 20% capacity, the model captures {third_model['model_recall_at_k']:.2%} of all held-out responders while contacting only {int(third_model['customers_contacted']):,} customers.\n"
    report += f"- At 50% capacity, the model captures {half_model['model_recall_at_k']:.2%} of held-out responders. This illustrates the efficiency-versus-coverage trade-off: broader coverage finds more total responders but lowers the average response rate.\n"
    report += "- Model-based Top-K selection improves efficiency by concentrating calls on customers with the highest estimated response probabilities.\n"
    report += "- Increasing capacity generally increases the number of responders found and recall, while response rate and lift move toward the full-population baseline.\n"
    report += "- Operationally, a bank can set K from available calling capacity, use the ranked list as a call queue, and monitor realized outcomes separately from the model’s probability estimates.\n"
    report += "- The model supports prioritization and resource allocation; it does not guarantee that a customer will subscribe.\n"
    report += "\n## Validation checks\n\n"
    for name, value in checks.items():
        if isinstance(value, dict):
            report += f"- `{name}`: `{json.dumps(value)}`\n"
        else:
            report += f"- `{name}`: **{value}**\n"
    report += "\n## Generated artifacts\n\n"
    report += f"- `{RANKED_PATH.relative_to(PROJECT_ROOT).as_posix()}`\n"
    report += f"- `{CAPACITY_CSV_PATH.relative_to(PROJECT_ROOT).as_posix()}`\n"
    report += f"- `{CAPACITY_JSON_PATH.relative_to(PROJECT_ROOT).as_posix()}`\n"
    for figure in figure_paths:
        report += f"- `{figure}`\n"
    report += "\n## Phase 3 boundary\n\nNo FastAPI backend, frontend, deployment, URL generation, or Phase 4 work was performed.\n"
    REPORT_PATH.write_text(report, encoding="utf-8")

    print(f"Validated exact held-out customer count: {total_test_customers:,}")
    print(f"Generated ranked customers: {RANKED_PATH}")
    print(f"Generated capacity analysis: {CAPACITY_CSV_PATH}")
    print(f"Validation checks passed: {all(check_pass(value) for value in checks.values())}")


if __name__ == "__main__":
    main()

