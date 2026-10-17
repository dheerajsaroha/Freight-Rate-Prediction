# Freight Rate Prediction Challenge

Predict spot freight rates for 12,000 validation loads and generate December 2025 rate predictions for a fixed lane (Lexington → Fort Wayne).

## Deployed App

The interactive Streamlit app is deployed and live at:

**https://freight-rate-prediction-hpth23jvmg8cvft23gvn8v.streamlit.app/**

The app provides four tabs:
- **Single load** — enter one load's details and get an instant rate prediction
- **Batch (CSV)** — upload a CSV of many loads, download predictions
- **December 2025 chart** — view the fixed-lane December predictions + chart image
- **Model info** — CV metrics, best hyperparameters, feature columns

No installation is required to use the deployed app. To run it locally, see the Usage section below.

## Project goal

Train a model on historical loads (`data/train-test.csv`, Jan–Oct 2025, target `posted_rate`) and predict:

1. Every load in `data/validation.csv` (Nov–Dec 2025) → `validation_predictions.csv` (`load_id`, `predicted_rate`).
2. Every day in December 2025 for the fixed lane → `data/december-chart-inputs.csv` (`predicted_rate`), then validate with `score.py`, which produces `scorer_results/candidate_december.png`.

## How it works (`train_predict.py`)

The pipeline runs end to end with no global preprocessing leakage:

1. **Load & validate** data. Training file must contain `posted_rate`; validation/December files contain load features.
2. **Feature engineering (no statistics learned yet)** — date cyclical encoding (month, day-of-week, weekend, day-of-year sin/cos), weight cleaning (negatives/missing flagged & imputed later), `log_distance`, `log_weight`, `distance/weight`, `market_index × quote_signal`, one-hot encoding for `pickup`/`delivery`/`equipment`.
3. **Target-transform comparison** — `log1p(posted_rate)` vs raw `posted_rate`, evaluated under identical CV; `log1p` selected (lower RMSE).
4. **Chronological cross-validation** — training rows sorted by `date`, then split into contiguous date-ordered folds (no random leakage across time).
5. **Fold-local preprocessing** — medians and one-hot encoder are fitted inside each fold on that fold's training rows only, so validation-fold information never influences preprocessing.
6. **Hyperparameter tuning** — grid search over `max_iter`, `learning_rate`, `max_depth`, `l2_regularization` using `HistGradientBoostingRegressor`, selecting by RMSE (lowest).
7. **Final model** — refit on all 48,000 training rows with the best transform + best params.
8. **Predict** validation set and December chart inputs; clip rates to be positive.
9. **Write outputs** — `validation_predictions.csv`, `data/december-chart-inputs.csv`, `model.joblib`, plus CV audit files (`cv_fold_metrics.csv`, `cv_pooled_metrics.csv`, `oof_predictions.csv`).

## Data

| File | Rows | Period | Purpose |
|---|---|---|---|
| `data/train-test.csv` | 48,000 | 2025-01-01 → 2025-10-31 | Training, target `posted_rate` |
| `data/validation.csv` | 12,000 | 2025-11-01 → 2025-12-31 | Holdout to predict |
| `data/validation_predictions_template.csv` | 12,000 | — | Expected `load_id` order |
| `data/december-chart-inputs.csv` | 31 | 2025-12-01 → 2025-12-31 | Fixed lane: Lexington → Fort Wayne, 360 mi, Dry Van, 32,000 lb |

## Requirements

`matplotlib`, `numpy`, `pandas`, `scikit-learn`, `joblib`.  
The pinned environment is captured in `requirements.txt`.

## Usage

```bash
# 1. Install dependencies (inside the project venv)
python -m pip install -r requirements.txt

# 2. Train, tune, and predict
python train_predict.py

# 3. Validate outputs with the official scorer
python score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs.csv

# 4. Launch the interactive prediction app
streamlit run app.py
```

The app opens on `http://localhost:8501` and provides four tabs:

| Tab | Purpose |
|---|---|
| **Single load** | Enter one load's details and get an instant rate prediction |
| **Batch (CSV)** | Upload a CSV of many loads, download predictions |
| **December 2025 chart** | View the fixed-lane December predictions + chart image |
| **Model info** | CV metrics, best hyperparameters, feature columns |


`train_predict.py` writes:
- `validation_predictions.csv` — 12,000 rows (`load_id`, `predicted_rate`)
- `data/december_chart_inputs.csv` and `data/december-chart-inputs.csv` — 31 rows with `predicted_rate`
- `model.joblib` — trained model + preprocessor
- `cv_fold_metrics.csv`, `cv_pooled_metrics.csv`, `oof_predictions.csv` — CV audit trail

`score.py` validates both output files and creates `scorer_results/candidate_december.png`.

## Results

Chronological 5-fold CV, pooled out-of-fold, target transform `log1p`:

| Metric | Value |
|---|---|
| MAE | ~120.88 |
| RMSE | ~597.54 |
| R² | ~0.8384 |
| Best params | `max_iter=400, learning_rate=0.03, max_depth=6, l2_regularization=0.5` |

## Output checks passed

Scorer confirms:
- 12,000 validation predictions valid (all `TE-000001`..`TE-012000`, positive rates)
- 31 December predictions valid (one per day, fixed lane inputs, positive rates)
- Chart created: `scorer_results/candidate_december.png`

## Submit

- This repository (code, `requirements.txt`, run instructions)
- `validation_predictions.csv`
- Report (PDF/DOCX) with validation approach, data split, and `candidate_december.png`
