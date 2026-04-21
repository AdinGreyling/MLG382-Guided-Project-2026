"""Training entry point for diabetes risk classification models."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from preprocess import (
    DEFAULT_DROP_COLS,
    TARGET_COL,
    build_classification_pipeline,
    load_data,
    make_train_test_split,
    split_features_target,
)

RANDOM_STATE = 42


def get_dataset_path() -> Path:
    """Return the most likely CSV path."""
    root = Path(__file__).resolve().parents[1]
    preferred = root / "data" / "raw" / "Diabetes_and_LifeStyle_Dataset_.csv"
    fallback = root / "Diabetes_and_LifeStyle_Dataset_.csv"

    if preferred.exists():
        return preferred
    if fallback.exists():
        return fallback

    raise FileNotFoundError(
        "Dataset not found. Place CSV at data/raw/Diabetes_and_LifeStyle_Dataset_.csv "
        "or in the project root."
    )


def build_models() -> dict[str, object]:
    """Create required classifiers from the project brief."""
    return {
        "DecisionTree": DecisionTreeClassifier(
            random_state=RANDOM_STATE,
            class_weight="balanced",
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=300,
            random_state=RANDOM_STATE,
            class_weight="balanced",
            n_jobs=-1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="multi:softprob",
            eval_metric="mlogloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }


def ensure_output_dirs(root: Path) -> tuple[Path, Path]:
    """Create folders for models and reports if needed."""
    model_dir = root / "models"
    report_dir = root / "reports" / "classification"
    model_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    return model_dir, report_dir


def train_and_evaluate() -> None:
    root = Path(__file__).resolve().parents[1]
    csv_path = get_dataset_path()
    model_dir, report_dir = ensure_output_dirs(root)

    print(f"Loading data from: {csv_path}")
    df = load_data(csv_path)
    X, y = split_features_target(df, target_col=TARGET_COL, drop_cols=DEFAULT_DROP_COLS)
    X_train, X_test, y_train, y_test = make_train_test_split(X, y)

    models = build_models()
    summary_rows: list[dict[str, float | str]] = []
    best_name = ""
    best_pipeline = None
    best_macro_f1 = -1.0

    for name, model in models.items():
        print(f"\nTraining: {name}")
        pipeline = build_classification_pipeline(model=model, X=X_train)
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        macro_f1 = f1_score(y_test, y_pred, average="macro")
        weighted_f1 = f1_score(y_test, y_pred, average="weighted")

        summary_rows.append(
            {
                "model": name,
                "macro_f1": round(float(macro_f1), 4),
                "weighted_f1": round(float(weighted_f1), 4),
            }
        )

        class_report = classification_report(y_test, y_pred, output_dict=True)
        report_path = report_dir / f"{name.lower()}_classification_report.json"
        report_path.write_text(json.dumps(class_report, indent=2), encoding="utf-8")

        labels = sorted(y_test.unique())
        cm = confusion_matrix(y_test, y_pred, labels=labels)
        cm_df = pd.DataFrame(cm, index=labels, columns=labels)
        cm_df.to_csv(report_dir / f"{name.lower()}_confusion_matrix.csv", index=True)

        print(f"{name} macro_f1={macro_f1:.4f} weighted_f1={weighted_f1:.4f}")

        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            best_name = name
            best_pipeline = pipeline

    summary_df = pd.DataFrame(summary_rows).sort_values(by="macro_f1", ascending=False)
    summary_df.to_csv(report_dir / "model_comparison.csv", index=False)

    if best_pipeline is None:
        raise RuntimeError("Training finished without a best model.")

    best_model_path = model_dir / "best_classification_pipeline.joblib"
    joblib.dump(best_pipeline, best_model_path)

    metadata = {
        "best_model": best_name,
        "best_macro_f1": round(float(best_macro_f1), 4),
        "target_column": TARGET_COL,
        "dropped_feature_columns": DEFAULT_DROP_COLS,
        "dataset_path": str(csv_path),
    }
    (model_dir / "best_classification_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print("\nTraining complete.")
    print(f"Best model: {best_name} (macro_f1={best_macro_f1:.4f})")
    print(f"Saved model: {best_model_path}")
    print(f"Saved reports: {report_dir}")


if __name__ == "__main__":
    train_and_evaluate()
