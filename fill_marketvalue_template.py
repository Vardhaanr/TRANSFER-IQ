from pathlib import Path

import numpy as np
import pandas as pd


def minmax(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").fillna(0.0)
    lo, hi = float(s.min()), float(s.max())
    if hi - lo < 1e-9:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - lo) / (hi - lo)


def main():
    final_path = Path("final_modeling_dataset.csv")
    template_path = Path("marketvalue_template.csv")

    if not final_path.exists() or not template_path.exists():
        print("Missing final_modeling_dataset.csv or marketvalue_template.csv")
        return

    df = pd.read_csv(final_path)
    template = pd.read_csv(template_path)

    for col in ["minutes_played", "goals", "assists", "pass_accuracy", "form_trend", "tackles", "games_played"]:
        if col not in df.columns:
            df[col] = 0.0

    # Construct a synthetic value score from existing performance signals.
    score = (
        0.25 * minmax(df["minutes_played"])
        + 0.20 * minmax(df["goals"])
        + 0.12 * minmax(df["assists"])
        + 0.18 * minmax(df["pass_accuracy"])
        + 0.15 * minmax(df["form_trend"])
        + 0.05 * minmax(df["tackles"])
        + 0.05 * minmax(df["games_played"])
    )

    # Map score to a realistic market value band (5M to 180M).
    min_value = 5_000_000
    max_value = 180_000_000
    synthetic_value = (min_value + score * (max_value - min_value)).round().astype(int)

    val_df = pd.DataFrame({"player": df["player"].astype(str), "market_value_est": synthetic_value})
    merged = template.merge(val_df, on="player", how="left")

    mv = pd.to_numeric(merged.get("market_value"), errors="coerce")
    merged["market_value"] = mv.fillna(merged["market_value_est"]).round().astype("Int64")
    merged = merged.drop(columns=["market_value_est"])

    if "season" not in merged.columns:
        merged["season"] = 2025
    merged["season"] = pd.to_numeric(merged["season"], errors="coerce").fillna(2025).astype(int)

    merged.to_csv(template_path, index=False)
    print(f"Filled rows: {int(merged['market_value'].notna().sum())}/{len(merged)}")
    print(f"Saved: {template_path}")


if __name__ == "__main__":
    main()
