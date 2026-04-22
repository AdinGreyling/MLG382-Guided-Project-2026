"""Dash web app for diabetes risk decision support."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
from dash import Dash, Input, Output, State, dcc, html

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from preprocess import DEFAULT_DROP_COLS, TARGET_COL, load_data  # noqa: E402


CLF_PATH = ROOT / "models" / "best_classification_pipeline.joblib"
CLF_META_PATH = ROOT / "models" / "best_classification_metadata.json"
SEG_PATH = ROOT / "models" / "kmeans_segmentation_pipeline.joblib"
SHAP_PATH = ROOT / "reports" / "explainability" / "classifier_shap_feature_importance.csv"
DATA_PATH = ROOT / "data" / "raw" / "Diabetes_and_LifeStyle_Dataset_.csv"

clf_pipeline = joblib.load(CLF_PATH)
clf_meta = json.loads(CLF_META_PATH.read_text(encoding="utf-8"))
seg_pipeline = joblib.load(SEG_PATH)
shap_importance = pd.read_csv(SHAP_PATH)

df_full = load_data(DATA_PATH)
feature_cols = [c for c in df_full.columns if c not in [TARGET_COL] + DEFAULT_DROP_COLS]

# Defaults for fields not on the form: use "No Diabetes" rows so hidden features are not
# dominated by dataset-wide Type 2 medians (which made almost every synthetic patient Type 2).
_baseline_df = df_full[df_full[TARGET_COL] == "No Diabetes"]
if len(_baseline_df) < 500:
    _baseline_df = df_full

numeric_defaults = (
    _baseline_df[feature_cols].select_dtypes(include="number").median(numeric_only=True).to_dict()
)
cat_defaults = {}
for c in _baseline_df[feature_cols].select_dtypes(exclude="number").columns:
    mode_series = _baseline_df[c].mode()
    cat_defaults[c] = mode_series.iloc[0] if not mode_series.empty else df_full[c].iloc[0]

numeric_feature_set = set(
    df_full[feature_cols].select_dtypes(include="number").columns.tolist()
)

# (column_id, short label, optional hint)
NUMERIC_INPUTS = [
    ("Age", "Age", "years"),
    ("bmi", "Body mass index (BMI)", ""),
    ("hba1c", "HbA1c", "% — long-term glucose control"),
    ("glucose_fasting", "Fasting glucose", "mg/dL"),
    ("glucose_postprandial", "Glucose after meal", "mg/dL"),
    ("systolic_bp", "Systolic blood pressure", "mmHg"),
    ("diastolic_bp", "Diastolic blood pressure", "mmHg"),
    ("physical_activity_minutes_per_week", "Physical activity", "minutes per week"),
    ("sleep_hours_per_day", "Sleep duration", "hours per night"),
    ("diet_score", "Diet quality score", "0–10 scale"),
]

CATEGORICAL_INPUTS = [
    ("gender", "Sex at birth", ""),
    ("smoking_status", "Smoking", ""),
    ("family_history_diabetes", "Family history of diabetes", "0 = no, 1 = yes"),
]


def field_style():
    return {"marginBottom": "1rem"}


def label_block(title: str, hint: str) -> html.Div:
    children = [html.Span(title, className="field-title")]
    if hint:
        children.append(html.Span(f" · {hint}", className="field-hint"))
    return html.Div(children, className="field-label-row")


def build_inputs() -> list:
    children = []
    for col, title, hint in NUMERIC_INPUTS:
        default = float(numeric_defaults.get(col, 0))
        children.append(
            html.Div(
                [
                    label_block(title, hint),
                    dcc.Input(
                        id=f"input-{col}",
                        type="number",
                        value=default,
                        className="input-text",
                    ),
                ],
                style=field_style(),
            )
        )

    for col, title, hint in CATEGORICAL_INPUTS:
        options = sorted(df_full[col].dropna().unique().tolist())
        default = cat_defaults.get(col, options[0] if options else None)
        children.append(
            html.Div(
                [
                    label_block(title, hint),
                    dcc.Dropdown(
                        id=f"input-{col}",
                        options=[{"label": str(o), "value": o} for o in options],
                        value=default,
                        clearable=False,
                        className="dropdown-compact",
                    ),
                ],
                style=field_style(),
            )
        )

    children.append(
        html.Button(
            "Run assessment",
            id="predict-btn",
            n_clicks=0,
            className="btn-primary",
        )
    )
    return children


def build_patient_row(values: dict) -> pd.DataFrame:
    row = {}
    for col in feature_cols:
        raw = values.get(col, None)
        if raw is not None and raw != "":
            if col in numeric_feature_set:
                try:
                    row[col] = float(raw)
                except (TypeError, ValueError):
                    row[col] = float(numeric_defaults.get(col, 0) or 0)
            else:
                row[col] = raw
        elif col in numeric_defaults:
            v = numeric_defaults[col]
            row[col] = float(v) if pd.notna(v) else 0.0
        elif col in cat_defaults:
            row[col] = cat_defaults[col]
        else:
            row[col] = 0
    return pd.DataFrame([row], columns=feature_cols)


def stage_probabilities(patient_df: pd.DataFrame) -> list[tuple[str, float]]:
    """Top predicted classes with probability (for transparency)."""
    if not hasattr(clf_pipeline, "predict_proba"):
        return []
    proba = clf_pipeline.predict_proba(patient_df)[0]
    labels = clf_meta.get("class_labels", [])
    if clf_meta.get("best_model_uses_label_encoding") and labels:
        names = [str(x) for x in labels]
    else:
        classes = getattr(clf_pipeline, "classes_", None)
        if classes is None:
            return []
        names = [str(c) for c in classes]
    pairs = sorted(zip(names, proba), key=lambda x: -x[1])[:5]
    return [(n, float(p)) for n, p in pairs]


def predict_stage(patient_df: pd.DataFrame) -> str:
    pred = clf_pipeline.predict(patient_df)[0]
    labels = clf_meta.get("class_labels", [])
    if clf_meta.get("best_model_uses_label_encoding") and labels:
        return str(labels[int(pred)])
    return str(pred)


def predict_cluster(patient_df: pd.DataFrame) -> int:
    return int(seg_pipeline.predict(patient_df)[0])


def clean_feature_name(raw: str) -> str:
    s = raw.replace("num__", "").replace("cat__", "_")
    return s.replace("_", " ").title()


top_drivers = shap_importance.head(10).copy()
top_drivers["feature"] = top_drivers["feature"].apply(clean_feature_name)

driver_fig = px.bar(
    top_drivers.sort_values("mean_abs_shap"),
    x="mean_abs_shap",
    y="feature",
    orientation="h",
)
driver_fig.update_traces(marker_color="#2563eb", marker_line_width=0)
driver_fig.update_layout(
    template="plotly_white",
    title=None,
    margin=dict(l=8, r=8, t=8, b=8),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    xaxis_title="Influence on risk prediction",
    yaxis_title="",
    font=dict(family="system-ui, Segoe UI, sans-serif", size=13, color="#334155"),
    xaxis=dict(showgrid=True, gridcolor="#f1f5f9", zeroline=False),
    yaxis=dict(showgrid=False),
    height=320,
)

RECOMMENDATION_BY_STAGE = {
    "No Diabetes": "Keep current habits; schedule routine screening and stay active.",
    "Pre-Diabetes": "Focus on diet quality, 150+ min activity weekly, and follow-up glucose or HbA1c checks.",
    "Type 1": "Ensure specialist endocrine care and insulin plan; this tool does not replace clinical advice.",
    "Type 2": "Combine clinician follow-up with weight, activity, and glucose targets as agreed with your care team.",
    "Gestational": "Coordinate with obstetrics and diabetes care for pregnancy-specific glucose monitoring.",
}

CLUSTER_LABELS = {
    0: "Lifestyle profile 1",
    1: "Lifestyle profile 2",
    2: "Lifestyle profile 3",
}

INDEX_STRING = """
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
        <link href="https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700;1,9..40,400&display=swap" rel="stylesheet">
        <style>
            :root {
                --bg: #f8fafc;
                --surface: #ffffff;
                --border: #e2e8f0;
                --text: #0f172a;
                --muted: #64748b;
                --accent: #2563eb;
                --accent-soft: #eff6ff;
                --radius: 12px;
                --shadow: 0 1px 3px rgba(15, 23, 42, 0.06);
            }
            body {
                margin: 0;
                background: var(--bg);
                color: var(--text);
                font-family: "DM Sans", system-ui, -apple-system, sans-serif;
                font-size: 15px;
                line-height: 1.5;
                -webkit-font-smoothing: antialiased;
            }
            .page {
                max-width: 1040px;
                margin: 0 auto;
                padding: 2rem 1.25rem 3rem;
            }
            .header {
                margin-bottom: 2rem;
            }
            .eyebrow {
                font-size: 0.75rem;
                font-weight: 600;
                letter-spacing: 0.06em;
                text-transform: uppercase;
                color: var(--muted);
                margin-bottom: 0.35rem;
            }
            .title {
                font-size: 1.65rem;
                font-weight: 700;
                letter-spacing: -0.02em;
                margin: 0 0 0.5rem 0;
                color: var(--text);
            }
            .subtitle {
                font-size: 0.95rem;
                color: var(--muted);
                max-width: 36rem;
                margin: 0;
            }
            .grid {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 1.25rem;
                align-items: start;
            }
            @media (max-width: 880px) {
                .grid { grid-template-columns: 1fr; }
            }
            .card {
                background: var(--surface);
                border: 1px solid var(--border);
                border-radius: var(--radius);
                box-shadow: var(--shadow);
                padding: 1.35rem 1.4rem;
            }
            .card-title {
                font-size: 0.8rem;
                font-weight: 600;
                text-transform: uppercase;
                letter-spacing: 0.04em;
                color: var(--muted);
                margin: 0 0 1rem 0;
            }
            .field-label-row {
                margin-bottom: 0.35rem;
            }
            .field-title {
                font-size: 0.875rem;
                font-weight: 600;
                color: var(--text);
            }
            .field-hint {
                font-size: 0.8rem;
                font-weight: 400;
                color: var(--muted);
            }
            .input-text {
                width: 100%;
                box-sizing: border-box;
                padding: 0.55rem 0.75rem;
                border: 1px solid var(--border);
                border-radius: 8px;
                font-family: inherit;
                font-size: 0.9rem;
                background: #fff;
                color: var(--text);
            }
            .input-text:focus {
                outline: none;
                border-color: var(--accent);
                box-shadow: 0 0 0 3px var(--accent-soft);
            }
            .dropdown-compact .Select-control {
                border-radius: 8px !important;
                border-color: var(--border) !important;
                min-height: 38px !important;
            }
            .btn-primary {
                width: 100%;
                margin-top: 0.5rem;
                padding: 0.65rem 1rem;
                font-family: inherit;
                font-size: 0.9rem;
                font-weight: 600;
                color: #fff;
                background: var(--accent);
                border: none;
                border-radius: 8px;
                cursor: pointer;
                transition: background 0.15s ease;
            }
            .btn-primary:hover {
                background: #1d4ed8;
            }
            .results-empty {
                color: var(--muted);
                font-size: 0.95rem;
                padding: 0.5rem 0;
            }
            .result-value {
                font-size: 1.35rem;
                font-weight: 700;
                letter-spacing: -0.02em;
                color: var(--text);
                margin: 0.15rem 0 0 0;
            }
            .result-label {
                font-size: 0.75rem;
                font-weight: 600;
                text-transform: uppercase;
                letter-spacing: 0.04em;
                color: var(--muted);
                margin-top: 1.1rem;
            }
            .result-label:first-of-type { margin-top: 0; }
            .pill {
                display: inline-block;
                margin-top: 0.35rem;
                padding: 0.2rem 0.55rem;
                font-size: 0.75rem;
                font-weight: 600;
                border-radius: 6px;
                background: var(--accent-soft);
                color: var(--accent);
            }
            .recommendation {
                margin-top: 1rem;
                padding-top: 1rem;
                border-top: 1px solid var(--border);
                font-size: 0.9rem;
                color: #334155;
            }
            .recommendation strong {
                display: block;
                font-size: 0.72rem;
                text-transform: uppercase;
                letter-spacing: 0.04em;
                color: var(--muted);
                margin-bottom: 0.35rem;
            }
            details.raw-data {
                margin-top: 1rem;
                font-size: 0.8rem;
                color: var(--muted);
            }
            details.raw-data summary {
                cursor: pointer;
                font-weight: 500;
            }
            details.raw-data pre {
                margin: 0.5rem 0 0;
                padding: 0.75rem;
                background: #f1f5f9;
                border-radius: 8px;
                overflow: auto;
                font-size: 0.72rem;
                color: #475569;
            }
            .prob-block {
                margin-top: 0.75rem;
                padding: 0.65rem 0.75rem;
                background: #f8fafc;
                border-radius: 8px;
                border: 1px solid var(--border);
            }
            .prob-block strong {
                display: block;
                font-size: 0.72rem;
                text-transform: uppercase;
                letter-spacing: 0.04em;
                color: var(--muted);
                margin-bottom: 0.35rem;
            }
            .prob-row {
                display: flex;
                justify-content: space-between;
                gap: 0.75rem;
                font-size: 0.82rem;
                color: #334155;
                padding: 0.15rem 0;
            }
            .chart-card {
                margin-top: 1.25rem;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
"""

app = Dash(__name__)
app.title = "BC Analytics · Diabetes risk"
app.index_string = INDEX_STRING
server = app.server

app.layout = html.Div(
    className="page",
    children=[
        html.Header(
            className="header",
            children=[
                html.Div("BC Analytics", className="eyebrow"),
                html.H1("Diabetes risk decision support", className="title"),
                html.P(
                    "Enter a few clinical and lifestyle measures. Fields you do not see use a "
                    "typical low-risk profile as the starting point, then your entries update the prediction.",
                    className="subtitle",
                ),
            ],
        ),
        html.Div(
            className="grid",
            children=[
                html.Div(
                    className="card",
                    children=[
                        html.H2("Patient profile", className="card-title"),
                        *build_inputs(),
                    ],
                ),
                html.Div(
                    className="card",
                    children=[
                        html.H2("Assessment summary", className="card-title"),
                        html.Div(
                            id="results",
                            children=html.Div(
                                "Adjust values on the left, then run the assessment.",
                                className="results-empty",
                            ),
                        ),
                    ],
                ),
            ],
        ),
        html.Div(
            className="card chart-card",
            children=[
                html.H2("What drives the model", className="card-title"),
                html.P(
                    "Approximate global influence (SHAP) on predicted diabetes stage. "
                    "Higher values mean the model leans on that signal more often.",
                    className="subtitle",
                    style={"marginBottom": "0.75rem", "maxWidth": "none"},
                ),
                dcc.Graph(figure=driver_fig, config={"displayModeBar": False}),
            ],
        ),
    ],
)


@app.callback(
    Output("results", "children"),
    Input("predict-btn", "n_clicks"),
    [State(f"input-{col}", "value") for col, _, _ in NUMERIC_INPUTS]
    + [State(f"input-{col}", "value") for col, _, _ in CATEGORICAL_INPUTS],
)
def run_prediction(n_clicks, *values):
    if not n_clicks:
        return html.Div(
            "Adjust values on the left, then run the assessment.",
            className="results-empty",
        )

    numeric_keys = [c for c, _, _ in NUMERIC_INPUTS]
    cat_keys = [c for c, _, _ in CATEGORICAL_INPUTS]
    input_map = dict(zip(numeric_keys + cat_keys, values))

    patient_df = build_patient_row(input_map)
    stage = predict_stage(patient_df)
    cluster = predict_cluster(patient_df)
    probs = stage_probabilities(patient_df)
    recommendation = RECOMMENDATION_BY_STAGE.get(
        stage, "Discuss results with a clinician for personalised advice."
    )
    cluster_label = CLUSTER_LABELS.get(cluster, f"Lifestyle profile {cluster + 1}")

    prob_block = None
    if probs:
        prob_block = html.Div(
            className="prob-block",
            children=[
                html.Strong("How confident the model is"),
                *[
                    html.Div(
                        className="prob-row",
                        children=[html.Span(name), html.Span(f"{pct * 100:.1f}%")],
                    )
                    for name, pct in probs
                ],
            ],
        )

    return html.Div(
        [
            html.Div(
                [
                    html.Div("Predicted diabetes stage", className="result-label"),
                    html.P(stage, className="result-value"),
                    html.Span("Model output · not a diagnosis", className="pill"),
                    prob_block,
                ]
            ),
            html.Div(
                [
                    html.Div("Lifestyle segment (K-means)", className="result-label"),
                    html.P(cluster_label, className="result-value"),
                ]
            ),
            html.Div(
                [
                    html.Strong("Suggested next step"),
                    recommendation,
                ],
                className="recommendation",
            ),
            html.Details(
                className="raw-data",
                children=[
                    html.Summary("Technical · full feature row sent to models"),
                    html.Pre(patient_df.iloc[0].to_string()),
                ],
            ),
        ]
    )


if __name__ == "__main__":
    app.run(debug=True)
