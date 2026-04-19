import json
import os
import pickle
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import ParameterSampler
from xgboost import XGBRegressor


SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)


@dataclass
class Config:
    input_csv: str = "final_modeling_dataset.csv"
    output_dir: str = "outputs/week5_7"
    seq_len: int = 3
    forecast_horizon: int = 3
    val_ratio: float = 0.2
    epochs: int = 40
    batch_size: int = 16
    lr: float = 1e-3
    lstm_hidden: int = 32
    lstm_layers: int = 1
    dropout: float = 0.0
    tuning_trials: int = 6


class LSTMRegressor(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)


class EncoderDecoderLSTM(nn.Module):
    def __init__(self, hidden_size: int, num_layers: int, horizon: int):
        super().__init__()
        self.horizon = horizon
        self.encoder = nn.LSTM(1, hidden_size, num_layers=num_layers, batch_first=True)
        self.decoder = nn.LSTM(1, hidden_size, num_layers=num_layers, batch_first=True)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x):
        _, (h, c) = self.encoder(x)
        decoder_input = x[:, -1:, :]
        outputs = []
        for _ in range(self.horizon):
            out, (h, c) = self.decoder(decoder_input, (h, c))
            step = self.head(out[:, -1:, :])
            outputs.append(step)
            decoder_input = step
        return torch.cat(outputs, dim=1).squeeze(-1)


def ensure_dirs(base_dir: str):
    Path(base_dir, "models").mkdir(parents=True, exist_ok=True)
    Path(base_dir, "plots").mkdir(parents=True, exist_ok=True)
    Path(base_dir, "reports").mkdir(parents=True, exist_ok=True)


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def metrics_dict(y_true, y_pred) -> Dict[str, float]:
    return {
        "rmse": rmse(y_true, y_pred),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def _safe_to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def load_and_prepare_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Input dataset not found: {path}")

    df = pd.read_csv(path)
    expected = {"player", "market_value"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df["market_value"] = _safe_to_numeric(df["market_value"])

    if "season" not in df.columns:
        # Fallback to synthetic order if season is not available.
        df["season"] = df.groupby("player").cumcount() + 1
    else:
        df["season"] = _safe_to_numeric(df["season"])

    optional_cols = [
        "games_played",
        "minutes_played",
        "total_passes",
        "pass_accuracy",
        "goals",
        "assists",
        "tackles",
        "form_trend",
        "sentiment_positive",
        "sentiment_negative",
        "sentiment_compound",
        "injury_days",
        "injury_count",
    ]
    for col in optional_cols:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = _safe_to_numeric(df[col]).fillna(0.0)

    df = df.dropna(subset=["market_value", "season"]).copy()
    if df.empty:
        raise ValueError(
            "No usable rows after cleaning. Ensure market_value and season have numeric values."
        )

    df = df.sort_values(["player", "season"]).reset_index(drop=True)
    return df


def build_univariate_sequences(df: pd.DataFrame, seq_len: int) -> Tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for _, grp in df.groupby("player"):
        values = grp["market_value"].values.astype(np.float32)
        if len(values) <= seq_len:
            continue
        for i in range(len(values) - seq_len):
            xs.append(values[i : i + seq_len].reshape(seq_len, 1))
            ys.append(values[i + seq_len])
    return np.array(xs), np.array(ys, dtype=np.float32)


def build_multivariate_sequences(
    df: pd.DataFrame, seq_len: int, feature_cols: List[str]
) -> Tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for _, grp in df.groupby("player"):
        arr = grp[feature_cols].values.astype(np.float32)
        target = grp["market_value"].values.astype(np.float32)
        if len(grp) <= seq_len:
            continue
        for i in range(len(grp) - seq_len):
            xs.append(arr[i : i + seq_len])
            ys.append(target[i + seq_len])
    return np.array(xs), np.array(ys, dtype=np.float32)


def build_seq2seq_data(
    df: pd.DataFrame, seq_len: int, horizon: int
) -> Tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for _, grp in df.groupby("player"):
        values = grp["market_value"].values.astype(np.float32)
        if len(values) < seq_len + horizon:
            continue
        for i in range(len(values) - seq_len - horizon + 1):
            xs.append(values[i : i + seq_len].reshape(seq_len, 1))
            ys.append(values[i + seq_len : i + seq_len + horizon])
    return np.array(xs), np.array(ys, dtype=np.float32)


def train_val_split(X, y, val_ratio=0.2):
    n = len(X)
    if n < 12:
        raise ValueError(f"Not enough sequences for training: {n}. Need at least 12.")
    split = max(1, int(n * (1 - val_ratio)))
    return X[:split], X[split:], y[:split], y[split:]


def train_lstm_model(
    X_train,
    y_train,
    X_val,
    y_val,
    input_size: int,
    hidden_size: int,
    num_layers: int,
    dropout: float,
    epochs: int,
    batch_size: int,
    lr: float,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMRegressor(input_size, hidden_size, num_layers, dropout).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_t = torch.tensor(y_val, dtype=torch.float32).to(device)

    train_losses = []
    val_losses = []

    for _ in range(epochs):
        model.train()
        perm = torch.randperm(X_train_t.size(0))
        batch_losses = []
        for i in range(0, X_train_t.size(0), batch_size):
            idx = perm[i : i + batch_size]
            xb = X_train_t[idx]
            yb = y_train_t[idx]
            pred = model(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t)
            val_loss = criterion(val_pred, y_val_t).item()

        train_losses.append(float(np.mean(batch_losses)))
        val_losses.append(val_loss)

    model.eval()
    with torch.no_grad():
        val_pred = model(X_val_t).cpu().numpy()

    return model, val_pred, train_losses, val_losses


def train_encoder_decoder(
    X_train,
    y_train,
    X_val,
    y_val,
    horizon: int,
    hidden_size: int,
    num_layers: int,
    epochs: int,
    batch_size: int,
    lr: float,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EncoderDecoderLSTM(hidden_size=hidden_size, num_layers=num_layers, horizon=horizon).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_t = torch.tensor(y_val, dtype=torch.float32).to(device)

    train_losses = []
    val_losses = []

    for _ in range(epochs):
        model.train()
        perm = torch.randperm(X_train_t.size(0))
        batch_losses = []
        for i in range(0, X_train_t.size(0), batch_size):
            idx = perm[i : i + batch_size]
            xb = X_train_t[idx]
            yb = y_train_t[idx]
            pred = model(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t)
            val_loss = criterion(val_pred, y_val_t).item()

        train_losses.append(float(np.mean(batch_losses)))
        val_losses.append(val_loss)

    model.eval()
    with torch.no_grad():
        val_pred = model(X_val_t).cpu().numpy()

    return model, val_pred, train_losses, val_losses


def save_loss_plot(train_losses, val_losses, title, out_path):
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="train_loss")
    plt.plot(val_losses, label="val_loss")
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def random_search_lstm(X_train, y_train, X_val, y_val, input_size, cfg: Config):
    param_grid = {
        "hidden_size": [16, 32, 64],
        "num_layers": [1, 2],
        "dropout": [0.0, 0.1, 0.2],
        "lr": [1e-3, 5e-4],
    }
    sampler = list(ParameterSampler(param_grid, n_iter=cfg.tuning_trials, random_state=SEED))
    best = None
    for p in sampler:
        model, val_pred, _, _ = train_lstm_model(
            X_train,
            y_train,
            X_val,
            y_val,
            input_size=input_size,
            hidden_size=p["hidden_size"],
            num_layers=p["num_layers"],
            dropout=p["dropout"],
            epochs=max(12, cfg.epochs // 2),
            batch_size=cfg.batch_size,
            lr=p["lr"],
        )
        score = rmse(y_val, val_pred)
        trial = {"params": p, "rmse": score, "model": model}
        if best is None or score < best["rmse"]:
            best = trial
    return best


def random_search_xgb(X_train, y_train, X_val, y_val, cfg: Config):
    param_grid = {
        "n_estimators": [200, 300, 500],
        "max_depth": [3, 4, 6],
        "learning_rate": [0.03, 0.05, 0.1],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0],
    }
    sampler = list(ParameterSampler(param_grid, n_iter=cfg.tuning_trials, random_state=SEED))
    best = None
    for p in sampler:
        model = XGBRegressor(objective="reg:squarederror", random_state=SEED, **p)
        model.fit(X_train, y_train)
        pred = model.predict(X_val)
        score = rmse(y_val, pred)
        trial = {"params": p, "rmse": score, "model": model}
        if best is None or score < best["rmse"]:
            best = trial
    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="final_modeling_dataset.csv", help="Input CSV path")
    args = parser.parse_args()

    cfg = Config(input_csv=args.input)
    ensure_dirs(cfg.output_dir)

    report = {
        "milestone_week5": {},
        "milestone_week6": {},
        "milestone_week7": {},
        "notes": [],
    }

    try:
        df = load_and_prepare_data(cfg.input_csv)
    except Exception as exc:
        report["notes"].append(str(exc))
        report_path = Path(cfg.output_dir, "reports", "model_evaluation_report.json")
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Data preparation failed: {exc}")
        print(f"Report saved to: {report_path}")
        return

    feature_cols = [
        "market_value",
        "games_played",
        "minutes_played",
        "total_passes",
        "pass_accuracy",
        "goals",
        "assists",
        "tackles",
        "form_trend",
        "injury_days",
        "injury_count",
        "sentiment_positive",
        "sentiment_negative",
        "sentiment_compound",
    ]

    try:
        # Week 5: univariate LSTM
        Xu, yu = build_univariate_sequences(df, cfg.seq_len)
        Xut, Xuv, yut, yuv = train_val_split(Xu, yu, cfg.val_ratio)
        uni_model, uni_pred, uni_train_loss, uni_val_loss = train_lstm_model(
            Xut,
            yut,
            Xuv,
            yuv,
            input_size=1,
            hidden_size=cfg.lstm_hidden,
            num_layers=cfg.lstm_layers,
            dropout=cfg.dropout,
            epochs=cfg.epochs,
            batch_size=cfg.batch_size,
            lr=cfg.lr,
        )
        uni_metrics = metrics_dict(yuv, uni_pred)
        torch.save(uni_model.state_dict(), Path(cfg.output_dir, "models", "univariate_lstm.pt"))
        save_loss_plot(
            uni_train_loss,
            uni_val_loss,
            "Univariate LSTM Loss Curve",
            Path(cfg.output_dir, "plots", "univariate_lstm_loss.png"),
        )

        # Week 5: multivariate LSTM
        Xm, ym = build_multivariate_sequences(df, cfg.seq_len, feature_cols)
        Xmt, Xmv, ymt, ymv = train_val_split(Xm, ym, cfg.val_ratio)
        multi_model, multi_pred, multi_train_loss, multi_val_loss = train_lstm_model(
            Xmt,
            ymt,
            Xmv,
            ymv,
            input_size=len(feature_cols),
            hidden_size=cfg.lstm_hidden,
            num_layers=cfg.lstm_layers,
            dropout=cfg.dropout,
            epochs=cfg.epochs,
            batch_size=cfg.batch_size,
            lr=cfg.lr,
        )
        multi_metrics = metrics_dict(ymv, multi_pred)
        torch.save(multi_model.state_dict(), Path(cfg.output_dir, "models", "multivariate_lstm.pt"))
        save_loss_plot(
            multi_train_loss,
            multi_val_loss,
            "Multivariate LSTM Loss Curve",
            Path(cfg.output_dir, "plots", "multivariate_lstm_loss.png"),
        )

        # Week 5: encoder-decoder multi-step LSTM
        Xe, ye = build_seq2seq_data(df, cfg.seq_len, cfg.forecast_horizon)
        Xet, Xev, yet, yev = train_val_split(Xe, ye, cfg.val_ratio)
        ed_model, ed_pred, ed_train_loss, ed_val_loss = train_encoder_decoder(
            Xet,
            yet,
            Xev,
            yev,
            horizon=cfg.forecast_horizon,
            hidden_size=cfg.lstm_hidden,
            num_layers=cfg.lstm_layers,
            epochs=cfg.epochs,
            batch_size=cfg.batch_size,
            lr=cfg.lr,
        )
        ed_metrics = metrics_dict(yev.reshape(-1), ed_pred.reshape(-1))
        torch.save(ed_model.state_dict(), Path(cfg.output_dir, "models", "encoder_decoder_lstm.pt"))
        save_loss_plot(
            ed_train_loss,
            ed_val_loss,
            "Encoder-Decoder LSTM Loss Curve",
            Path(cfg.output_dir, "plots", "encoder_decoder_lstm_loss.png"),
        )

        # Week 6: XGBoost base model and integrated ensemble
        xgb_train = Xmt[:, -1, :]
        xgb_val = Xmv[:, -1, :]
        xgb_model = XGBRegressor(
            objective="reg:squarederror",
            random_state=SEED,
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
        )
        xgb_model.fit(xgb_train, ymt)
        xgb_pred_train = xgb_model.predict(xgb_train)
        xgb_pred_val = xgb_model.predict(xgb_val)
        xgb_metrics = metrics_dict(ymv, xgb_pred_val)

        with torch.no_grad():
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            uni_model.eval()
            multi_model.eval()
            uni_train_pred = uni_model(torch.tensor(Xut, dtype=torch.float32).to(device)).cpu().numpy()
            multi_train_pred = multi_model(torch.tensor(Xmt, dtype=torch.float32).to(device)).cpu().numpy()

        n_train = min(len(xgb_pred_train), len(multi_train_pred), len(uni_train_pred), len(ymt), len(yut))
        n_val = min(len(xgb_pred_val), len(multi_pred), len(uni_pred), len(ymv), len(yuv))
        ens_train_features = np.column_stack(
            [
                xgb_pred_train[:n_train],
                multi_train_pred[:n_train],
                uni_train_pred[:n_train],
            ]
        )
        ens_val_features = np.column_stack(
            [
                xgb_pred_val[:n_val],
                multi_pred[:n_val],
                uni_pred[:n_val],
            ]
        )

        ensemble_model = LinearRegression()
        ensemble_model.fit(ens_train_features, ymt[:n_train])
        ensemble_pred = ensemble_model.predict(ens_val_features)
        ensemble_metrics = metrics_dict(ymv[:n_val], ensemble_pred)

        with open(Path(cfg.output_dir, "models", "xgboost_model.pkl"), "wb") as f:
            pickle.dump(xgb_model, f)
        with open(Path(cfg.output_dir, "models", "ensemble_meta_model.pkl"), "wb") as f:
            pickle.dump(ensemble_model, f)

        pred_df = pd.DataFrame(
            {
                "actual": ymv[:n_val],
                "xgb_pred": xgb_pred_val[:n_val],
                "multivariate_lstm_pred": multi_pred[:n_val],
                "univariate_lstm_pred": uni_pred[:n_val],
                "ensemble_pred": ensemble_pred,
            }
        )
        pred_df.to_csv(Path(cfg.output_dir, "reports", "validation_predictions.csv"), index=False)

        # Week 7: hyperparameter tuning
        best_lstm = random_search_lstm(Xmt, ymt, Xmv, ymv, len(feature_cols), cfg)
        best_xgb = random_search_xgb(xgb_train, ymt, xgb_val, ymv, cfg)

        torch.save(best_lstm["model"].state_dict(), Path(cfg.output_dir, "models", "best_tuned_lstm.pt"))
        with open(Path(cfg.output_dir, "models", "best_tuned_xgb.pkl"), "wb") as f:
            pickle.dump(best_xgb["model"], f)

        report["milestone_week5"] = {
            "univariate_lstm_metrics": uni_metrics,
            "multivariate_lstm_metrics": multi_metrics,
            "encoder_decoder_metrics": ed_metrics,
        }
        report["milestone_week6"] = {
            "xgboost_metrics": xgb_metrics,
            "ensemble_metrics": ensemble_metrics,
        }
        report["milestone_week7"] = {
            "best_lstm_params": best_lstm["params"],
            "best_lstm_rmse": best_lstm["rmse"],
            "best_xgb_params": best_xgb["params"],
            "best_xgb_rmse": best_xgb["rmse"],
        }
        report["notes"].append(
            "Integrated ensemble combines XGBoost + multivariate LSTM + univariate LSTM predictions."
        )
    except Exception as exc:
        report["notes"].append(f"Training pipeline failed: {exc}")

    report_path = Path(cfg.output_dir, "reports", "model_evaluation_report.json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("Week 5-7 pipeline completed.")
    print(f"Artifacts saved under: {cfg.output_dir}")


if __name__ == "__main__":
    main()
