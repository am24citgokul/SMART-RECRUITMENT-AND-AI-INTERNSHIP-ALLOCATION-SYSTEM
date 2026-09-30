# Internship Portal — ML Foundation Rebuild V2

> **This dataset is SYNTHETIC, generated to enable a valid train/test pipeline. It is not real institutional data. Any accuracy metrics below are computed on this synthetic data and should be described as such in any report or resume.**

This is a fully isolated ML foundation prototype. It reads the source schema only; it does not alter the existing portal, original data, templates, static files, or model pickles.

## Run

From this directory, install the existing project's Python dependencies, then run:

```powershell
python src/run_all.py
pytest tests
python ui_prototype/app.py
```

Open `http://127.0.0.1:5052`. The dashboard fetches `/api/dashboard`, whose payload is generated from the trained models after each pipeline run. No dashboard metrics or prediction values are manually embedded in the UI.

## Outputs

- `data/synthetic_students.csv`: reproducible, 900-row synthetic data matching the source schema plus documented derived fields.
- `models/`: freshly trained classifier, regressor, and scheme recommender.
- `reports/metrics.json`: held-out evaluation values actually computed by the run.
- `reports/evaluation_report.md`: honest metrics, plots, and limitations.

The synthetic generator combines correlated academic and portfolio variables with independent noise; it does not produce purely random labels. Before any real use, replace it with consented, representative institutional data and perform fresh validation.
