# MLG382 Guided Project 2026

Diabetes risk decision support project for BC Analytics.

## Project Goals

- Predict diabetes risk class using `diabetes_stage`
- Segment patients into lifestyle groups using K-Means (`k=3`)
- Explain model outputs using SHAP
- Deploy an interactive Dash app

## Recommended Setup

```powershell
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

## Project Structure

- `data/raw/` raw dataset files
- `notebooks/` exploration notebooks
- `src/` training and preprocessing scripts
- `models/` saved trained models
- `app/` Dash web app code
- `reports/` report assets and final report
