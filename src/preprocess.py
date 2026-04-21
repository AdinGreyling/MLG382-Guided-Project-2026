"""Data preprocessing utilities for the diabetes project."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# Main config values used in multiple scripts
TARGET_COL = "diabetes_stage"
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Columns that can leak target information if included as features
DEFAULT_DROP_COLS = ["diabetes_risk_score", "diagnosed_diabetes"]


def load_data(csv_path: str | Path) -> pd.DataFrame:
    """Load the raw CSV and return a dataframe."""
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")

    df = pd.read_csv(path)
    if df.empty:
        raise ValueError("Dataset is empty.")
    return df


def validate_schema(
    df: pd.DataFrame,
    target_col: str = TARGET_COL,
    required_cols: Iterable[str] | None = None,
) -> None:
    """Check that required columns exist before training."""
    missing_cols = []

    if target_col not in df.columns:
        missing_cols.append(target_col)

    if required_cols:
        for col in required_cols:
            if col not in df.columns:
                missing_cols.append(col)

    if missing_cols:
        raise ValueError(f"Missing columns: {sorted(set(missing_cols))}")


def split_features_target(
    df: pd.DataFrame,
    target_col: str = TARGET_COL,
    drop_cols: Iterable[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Split dataframe into model features X and target y."""
    validate_schema(df, target_col=target_col)

    y = df[target_col].copy()
    X = df.drop(columns=[target_col]).copy()

    if drop_cols:
        existing_drop_cols = [col for col in drop_cols if col in X.columns]
        X = X.drop(columns=existing_drop_cols)

    if X.empty:
        raise ValueError("No feature columns left after dropping columns.")

    return X, y


def get_feature_types(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Return categorical and numerical column lists."""
    categorical_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()

    if not categorical_cols and not numeric_cols:
        raise ValueError("No usable feature columns found.")

    return categorical_cols, numeric_cols


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Create a reusable preprocessing block for mixed data types."""
    categorical_cols, numeric_cols = get_feature_types(X)

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", categorical_pipeline, categorical_cols),
            ("num", numeric_pipeline, numeric_cols),
        ],
        remainder="drop",
    )
    return preprocessor


def make_train_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
):
    """Create train/test split with stratification for class balance."""
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )


def build_classification_pipeline(model, X: pd.DataFrame) -> Pipeline:
    """Combine preprocessing and model into one pipeline."""
    preprocessor = build_preprocessor(X)
    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )


if __name__ == "__main__":
    # Quick local check so you can run this file directly.
    csv_file = Path("data/raw/Diabetes_and_LifeStyle_Dataset_.csv")
    df = load_data(csv_file)
    X, y = split_features_target(df, drop_cols=DEFAULT_DROP_COLS)
    X_train, X_test, y_train, y_test = make_train_test_split(X, y)

    print("Dataset shape:", df.shape)
    print("Feature shape:", X.shape)
    print("Target classes:", y.nunique())
    print("Train:", X_train.shape, "Test:", X_test.shape)
