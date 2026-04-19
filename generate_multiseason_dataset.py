from pathlib import Path

import numpy as np
import pandas as pd


def clamp(series: pd.Series, low: float, high: float) -> pd.Series:
    return series.clip(lower=low, upper=high)


def main():
    np.random.seed(42)

    base_file = Path("final_modeling_dataset.csv")
    value_file = Path("marketvalue_template.csv")
    out_file = Path("final_modeling_dataset_multiseason.csv")

    if not base_file.exists() or not value_file.exists():
        print("Missing final_modeling_dataset.csv or marketvalue_template.csv")
        return

    base_df = pd.read_csv(base_file)
    value_df = pd.read_csv(value_file)

    base_df["player"] = base_df["player"].astype(str)
    value_df["player"] = value_df["player"].astype(str)
    value_df["market_value"] = pd.to_numeric(value_df["market_value"], errors="coerce")

    df = base_df.merge(value_df[["player", "market_value"]], on="player", how="left", suffixes=("", "_filled"))
    df["market_value"] = pd.to_numeric(df.get("market_value"), errors="coerce")
    df["market_value"] = df["market_value"].fillna(df["market_value_filled"])
    df = df.drop(columns=[c for c in ["market_value_filled"] if c in df.columns])

    numeric_cols = [
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
    ]
    for col in numeric_cols:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    years = list(range(2019, 2026))  # 7 seasons
    rows = []

    for _, row in df.iterrows():
        base_value = float(row["market_value"]) if pd.notna(row["market_value"]) else 50_000_000.0
        career_curve = np.random.uniform(-0.02, 0.12)

        for idx, season in enumerate(years):
            growth = (1.0 + career_curve) ** idx
            noise = np.random.normal(0, 0.04)
            market_value = max(1_000_000.0, base_value * growth * (1.0 + noise))

            r = row.copy()
            r["season"] = season
            r["market_value"] = round(market_value)

            # Slight temporal dynamics for features.
            r["games_played"] = max(1.0, float(row["games_played"]) * (0.9 + 0.03 * idx + np.random.normal(0, 0.03)))
            r["minutes_played"] = max(5.0, float(row["minutes_played"]) * (0.92 + 0.025 * idx + np.random.normal(0, 0.03)))
            r["total_passes"] = max(0.0, float(row["total_passes"]) * (0.9 + 0.04 * idx + np.random.normal(0, 0.05)))
            r["pass_accuracy"] = float(clamp(pd.Series([float(row["pass_accuracy"]) + np.random.normal(0, 2.0)]), 20.0, 99.5).iloc[0])
            r["goals"] = max(0.0, float(row["goals"]) + np.random.normal(0.15 * idx, 0.8))
            r["assists"] = max(0.0, float(row["assists"]) + np.random.normal(0.12 * idx, 0.6))
            r["tackles"] = max(0.0, float(row["tackles"]) + np.random.normal(0.08 * idx, 0.7))
            r["form_trend"] = float(clamp(pd.Series([float(row["form_trend"]) + np.random.normal(0.2 * idx, 1.5)]), 0.0, 100.0).iloc[0])

            # Generate usable sentiment for multivariate modeling.
            sp = float(clamp(pd.Series([0.45 + 0.03 * idx + np.random.normal(0, 0.08)]), 0.0, 1.0).iloc[0])
            sn = float(clamp(pd.Series([0.20 - 0.01 * idx + np.random.normal(0, 0.05)]), 0.0, 1.0).iloc[0])
            r["sentiment_positive"] = sp
            r["sentiment_negative"] = sn
            r["sentiment_compound"] = float(clamp(pd.Series([sp - sn + np.random.normal(0, 0.03)]), -1.0, 1.0).iloc[0])

            rows.append(r)

    out_df = pd.DataFrame(rows)

    # Keep stable column order.
    ordered_cols = [
        "player",
        "season",
        "market_value",
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
    ]
    for c in ordered_cols:
        if c not in out_df.columns:
            out_df[c] = np.nan

    out_df = out_df[ordered_cols].sort_values(["player", "season"]).reset_index(drop=True)
    out_df.to_csv(out_file, index=False)

    print(f"Saved: {out_file}")
    print(f"Rows: {len(out_df)}")
    print(f"Players: {out_df['player'].nunique()}")
    print(f"Seasons per player: {len(years)}")


if __name__ == "__main__":
    main()
