"""Training entry point for K-Means patient segmentation (k=3)."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from preprocess import DEFAULT_DROP_COLS, TARGET_COL, load_data

RANDOM_STATE = 42
N_CLUSTERS = 3


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


def ensure_output_dirs(root: Path) -> tuple[Path, Path]:
    model_dir = root / "models"
    report_dir = root / "reports" / "segmentation"
    model_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    return model_dir, report_dir


def build_preprocessor_for_kmeans(X: pd.DataFrame) -> ColumnTransformer:
    num_cols = X.select_dtypes(include="number").columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    num_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("cat", cat_pipe, cat_cols),
            ("num", num_pipe, num_cols),
        ],
        remainder="drop",
    )


def train_kmeans() -> None:
    root = Path(__file__).resolve().parents[1]
    csv_path = get_dataset_path()
    model_dir, report_dir = ensure_output_dirs(root)

    print(f"Loading data from: {csv_path}")
    df = load_data(csv_path)

    # For segmentation, we exclude the target and any leakage-style columns
    drop_cols = [TARGET_COL] + [c for c in DEFAULT_DROP_COLS if c in df.columns]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns]).copy()

    preprocessor = build_preprocessor_for_kmeans(X)
    X_mat = preprocessor.fit_transform(X)

    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE, n_init=10)
    clusters = kmeans.fit_predict(X_mat)

    df_out = df.copy()
    df_out["cluster"] = clusters

    # Silhouette can be expensive; sample if needed
    if X_mat.shape[0] > 20000:
        rng = np.random.default_rng(RANDOM_STATE)
        idx = rng.choice(X_mat.shape[0], size=20000, replace=False)
        sil = silhouette_score(X_mat[idx], clusters[idx])
    else:
        sil = silhouette_score(X_mat, clusters)

    cluster_counts = df_out["cluster"].value_counts().sort_index()
    cluster_counts.to_csv(report_dir / "cluster_counts.csv", index=True)

    # Simple cluster profiles using numeric means + categorical modes
    numeric_cols = df_out.select_dtypes(include=["number"]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c not in ["cluster"]]

    numeric_profile = (
        df_out.groupby("cluster")[numeric_cols].mean(numeric_only=True).round(3)
    )
    numeric_profile.to_csv(report_dir / "cluster_profile_numeric_means.csv", index=True)

    num_for_profile = df_out.select_dtypes(include="number").columns.tolist()
    cat_cols = [c for c in df_out.columns if c not in num_for_profile and c not in ("cluster", TARGET_COL)]

    cat_modes = []
    for c in cat_cols:
        modes = (
            df_out.groupby("cluster")[c]
            .agg(lambda s: s.mode().iloc[0] if not s.mode().empty else None)
            .rename(c)
        )
        cat_modes.append(modes)
    if cat_modes:
        cat_profile = pd.concat(cat_modes, axis=1)
        cat_profile.to_csv(report_dir / "cluster_profile_categorical_modes.csv", index=True)

    # Save the full segmentation pipeline for later use in the app
    pipeline = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("kmeans", kmeans),
        ]
    )
    joblib.dump(pipeline, model_dir / "kmeans_segmentation_pipeline.joblib")

    meta = {
        "n_clusters": N_CLUSTERS,
        "random_state": RANDOM_STATE,
        "silhouette_score": float(round(sil, 4)),
        "dataset_path": str(csv_path),
        "dropped_columns": drop_cols,
    }
    (model_dir / "kmeans_segmentation_metadata.json").write_text(
        json.dumps(meta, indent=2),
        encoding="utf-8",
    )

    print("\nSegmentation complete.")
    print("Cluster counts:")
    print(cluster_counts.to_string())
    print(f"Silhouette score: {sil:.4f}")
    print(f"Saved model: {model_dir / 'kmeans_segmentation_pipeline.joblib'}")
    print(f"Saved reports: {report_dir}")


if __name__ == "__main__":
    train_kmeans()

