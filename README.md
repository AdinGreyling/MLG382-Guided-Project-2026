# MLG382 Guided Project 2026

Solo submission — BC Analytics diabetes risk prototype for MLG382: classify `diabetes_stage` (decision tree, random forest, XGBoost), k-means (k=3) for segments, SHAP exports under `reports/explainability/`, Dash UI in `app/app.py`.

**Repo:** https://github.com/AdinGreyling/MLG382-Guided-Project-2026  
**Live app:** https://mlg382-guided-project-2026.onrender.com/  
**BC Connect report:** two pages, CRISP-DM, links to this repo and the live app.

`data/raw/` holds the CSV (and the brief PDF). `src/` has preprocess + train + explain scripts. `models/` gets the pipelines after a train run. `*.joblib` is gitignored by default. `reports/` holds the exported metrics and SHAP CSVs.

`diabetes_stage` is the target (No Diabetes, Pre-Diabetes, Type 1, Type 2, Gestational). Type 2 is most rows; macro F1 in `reports/classification/model_comparison.csv` matters alongside weighted F1.

```powershell
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

```powershell
.\venv\Scripts\python src\train_classify.py
.\venv\Scripts\python src\train_segment.py
.\venv\Scripts\python src\explain.py
```

```powershell
.\venv\Scripts\python app\app.py
```

Local URL: http://127.0.0.1:8050/

Render: `Procfile`, `render.yaml`, `runtime.txt` target gunicorn on `app.app:server` with `$PORT`. Build runs the three Python scripts under `src/` so the app finds `models/` without pushing large binaries. If Render build times out: train locally, add the `.joblib` files to the repo, shorten the build command.
