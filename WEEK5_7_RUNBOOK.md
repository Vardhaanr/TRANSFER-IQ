# Week 5-7 Modeling Runbook

## What this pipeline does

The script `week5_7_modeling_pipeline.py` covers the required milestones:

- Week 5:
  - Univariate LSTM for market value forecasting
  - Multivariate LSTM with performance, injury, and sentiment features
  - Encoder-decoder LSTM for multi-step forecasting
- Week 6:
  - XGBoost regressor
  - Integrated ensemble model combining XGBoost + LSTM predictions
- Week 7:
  - Random-search style tuning for LSTM and XGBoost
  - Evaluation with RMSE, MAE, and R2

## Required input dataset

The script expects `final_modeling_dataset.csv` with at least:

- `player`
- `season`
- `market_value`

Optional columns used if present:

- `games_played`, `minutes_played`, `total_passes`, `pass_accuracy`
- `goals`, `assists`, `tackles`, `form_trend`
- `sentiment_positive`, `sentiment_negative`, `sentiment_compound`
- `injury_days`, `injury_count`

## Install dependencies

```powershell
c:/Users/vardh/OneDrive/Desktop/infosys/.venv/Scripts/python.exe -m pip install -r requirements_week5_7.txt
```

## Run

```powershell
c:/Users/vardh/OneDrive/Desktop/infosys/.venv/Scripts/python.exe week5_7_modeling_pipeline.py
```

## Outputs

Artifacts are generated in `outputs/week5_7`:

- `models/`
  - `univariate_lstm.pt`
  - `multivariate_lstm.pt`
  - `encoder_decoder_lstm.pt`
  - `xgboost_model.pkl`
  - `ensemble_meta_model.pkl`
  - `best_tuned_lstm.pt`
  - `best_tuned_xgb.pkl`
- `plots/`
  - loss curve PNG files
- `reports/`
  - `validation_predictions.csv`
  - `model_evaluation_report.json`
