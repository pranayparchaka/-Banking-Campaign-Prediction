"""Leakage-safe preprocessing utilities for the banking campaign dataset.

This module deliberately keeps ``y`` outside the feature matrix and drops
``duration`` from the primary pre-call feature set. The returned transformer
must be fitted only on a training partition in later modeling phases.
"""

from __future__ import annotations

from typing import Iterable, Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TARGET_COLUMN = "y"
LEAKAGE_COLUMNS = ("duration",)


def prepare_features_target(
    frame: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Return leakage-safe features and the target without fitting anything.

    ``duration`` is excluded because it is known only after a campaign call.
    The target column is removed explicitly and is never passed to the feature
    transformer.
    """

    if target_column not in frame.columns:
        raise ValueError(f"Required target column '{target_column}' is missing.")

    target = frame[target_column].copy()
    feature_columns = [
        column
        for column in frame.columns
        if column != target_column and column not in LEAKAGE_COLUMNS
    ]
    features = frame.loc[:, feature_columns].copy()

    if TARGET_COLUMN in features.columns:
        raise AssertionError("Target leakage: y was retained in the feature matrix.")
    leaked = set(LEAKAGE_COLUMNS).intersection(features.columns)
    if leaked:
        raise AssertionError(f"Leakage columns retained: {sorted(leaked)}")

    return features, target


def infer_feature_types(
    features: pd.DataFrame,
) -> Tuple[list[str], list[str]]:
    """Identify numerical and categorical columns from the feature frame."""

    numeric_columns = features.select_dtypes(include=["number"]).columns.tolist()
    categorical_columns = [
        column for column in features.columns if column not in numeric_columns
    ]
    return numeric_columns, categorical_columns


def build_preprocessor(
    numeric_columns: Iterable[str],
    categorical_columns: Iterable[str],
) -> ColumnTransformer:
    """Build the unfitted preprocessing transformer.

    Numerical values use median imputation followed by standardization.
    Categorical values use most-frequent imputation followed by one-hot
    encoding with unknown-category tolerance for future inference.

    The caller is responsible for fitting this transformer on training data
    only, for example ``preprocessor.fit(X_train)``.
    """

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, list(numeric_columns)),
            ("categorical", categorical_pipeline, list(categorical_columns)),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

