"""Generate explainability outputs for the project using SHAP."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from preprocess import DEFAULT_DROP_COLS, TARGET_COL, load_data, split_features_target

RANDOM_STATE = 42


def get_dataset_path() -> Path:
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


def ensure_output_dir(root: Path) -> Path:
    out_dir = root / "reports" / "explainability"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def load_best_classifier(root: Path) -> Pipeline:
    path = root / "models" / "best_classification_pipeline.joblib"
    if not path.exists():
        raise FileNotFoundError(
            "Best classification model not found. Run src/train_classify.py first."
        )
    return joblib.load(path)


def explain_classifier(sample_size: int = 2000) -> None:
    root = Path(__file__).resolve().parents[1]
    out_dir = ensure_output_dir(root)

    clf = load_best_classifier(root)
    csv_path = get_dataset_path()
    df = load_data(csv_path)
    X, _ = split_features_target(df, target_col=TARGET_COL, drop_cols=DEFAULT_DROP_COLS)

    # Keep SHAP fast by sampling
    if len(X) > sample_size:
        rng = np.random.default_rng(RANDOM_STATE)
        idx = rng.choice(len(X), size=sample_size, replace=False)
        X_sample = X.iloc[idx].copy()
    else:
        X_sample = X.copy()

    # Transform features so SHAP sees what the model sees
    preprocess = clf.named_steps["preprocess"]
    model = clf.named_steps["model"]

    X_trans = preprocess.transform(X_sample)

    # Feature names after one-hot encoding
    feature_names = preprocess.get_feature_names_out()

    explainer = shap.Explainer(model, feature_names=feature_names)
    shap_values = explainer(X_trans)

    # Save a global feature importance table
    # For multiclass models, SHAP values can be 3D: (rows, features, classes)
    values = shap_values.values
    if values.ndim == 3:
        mean_abs = np.abs(values).mean(axis=(0, 2))
    else:
        mean_abs = np.abs(values).mean(axis=0)
    importance = pd.DataFrame(
        {
            "feature": feature_names,
            "mean_abs_shap": mean_abs,
        }
    ).sort_values("mean_abs_shap", ascending=False)

    importance.to_csv(out_dir / "classifier_shap_feature_importance.csv", index=False)

    # Save metadata to make reporting easier
    meta = {
        "model_type": type(model).__name__,
        "sample_size": int(len(X_sample)),
        "dataset_path": str(csv_path),
    }
    (out_dir / "classifier_shap_metadata.json").write_text(
        json.dumps(meta, indent=2),
        encoding="utf-8",
    )

    print("Saved SHAP importance to:", out_dir / "classifier_shap_feature_importance.csv")


def explain_segmentation_surrogate(sample_size: int = 8000, shap_rows: int = 1500) -> None:
    """SHAP for clusters via a surrogate model (K-Means has no direct SHAP target)."""
    root = Path(__file__).resolve().parents[1]
    out_dir = ensure_output_dir(root)
    seg_path = root / "models" / "kmeans_segmentation_pipeline.joblib"
    if not seg_path.exists():
        raise FileNotFoundError("Segmentation model not found. Run src/train_segment.py first.")

    seg: Pipeline = joblib.load(seg_path)
    csv_path = get_dataset_path()
    df = load_data(csv_path)
    drop_cols = [TARGET_COL] + [c for c in DEFAULT_DROP_COLS if c in df.columns]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns]).copy()

    if len(X) > sample_size:
        rng = np.random.default_rng(RANDOM_STATE)
        idx = rng.choice(len(X), size=sample_size, replace=False)
        X_sample = X.iloc[idx]
    else:
        X_sample = X

    clusters = seg.predict(X_sample)
    preprocess = seg.named_steps["preprocess"]
    X_mat = preprocess.transform(X_sample)

    surrogate = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    surrogate.fit(X_mat, clusters)

    if X_mat.shape[0] > shap_rows:
        rng = np.random.default_rng(RANDOM_STATE + 1)
        ridx = rng.choice(X_mat.shape[0], size=shap_rows, replace=False)
        X_shap = X_mat[ridx]
    else:
        X_shap = X_mat

    feature_names = preprocess.get_feature_names_out()
    explainer = shap.TreeExplainer(surrogate)
    shap_values = explainer(X_shap)

    values = shap_values.values
    if values.ndim == 3:
        mean_abs = np.abs(values).mean(axis=(0, 2))
    else:
        mean_abs = np.abs(values).mean(axis=0)

    importance = pd.DataFrame(
        {"feature": feature_names, "mean_abs_shap": mean_abs}
    ).sort_values("mean_abs_shap", ascending=False)
    importance.to_csv(out_dir / "segmentation_surrogate_shap_feature_importance.csv", index=False)

    meta = {
        "method": "SHAP on RandomForest surrogate predicting KMeans cluster id",
        "sample_size": int(len(X_sample)),
        "shap_rows": int(X_shap.shape[0]),
        "dataset_path": str(csv_path),
    }
    (out_dir / "segmentation_surrogate_shap_metadata.json").write_text(
        json.dumps(meta, indent=2),
        encoding="utf-8",
    )
    print(
        "Saved segmentation SHAP (surrogate) to:",
        out_dir / "segmentation_surrogate_shap_feature_importance.csv",
    )


if __name__ == "__main__":
    explain_classifier()
    explain_segmentation_surrogate()

