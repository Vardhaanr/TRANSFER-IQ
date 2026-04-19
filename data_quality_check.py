import json
import argparse
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="final_modeling_dataset.csv", help="Input CSV path")
    args = parser.parse_args()

    input_file = Path(args.input)
    out_dir = Path("outputs/week5_7/reports")
    out_dir.mkdir(parents=True, exist_ok=True)

    if not input_file.exists():
        print(f"Missing input file: {input_file}")
        return

    df = pd.read_csv(input_file)

    required = ["player", "market_value"]
    missing_cols = [c for c in required if c not in df.columns]

    report = {
        "file": str(input_file),
        "total_rows": int(len(df)),
        "missing_required_columns": missing_cols,
        "column_stats": {},
        "usable_rows": 0,
        "invalid_rows": 0,
        "notes": [],
    }

    if missing_cols:
        report["notes"].append("Dataset is missing required columns for model training.")
        out_json = out_dir / "data_quality_report.json"
        out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Saved: {out_json}")
        return

    # Normalize key columns for validation.
    player_clean = df["player"].astype(str).str.strip()
    if "season" in df.columns:
        season_num = pd.to_numeric(df["season"], errors="coerce")
        season_ok = season_num.notna()
        season_mode = "provided"
    else:
        season_ok = pd.Series([True] * len(df), index=df.index)
        season_mode = "synthetic_fallback"
    market_num = pd.to_numeric(df["market_value"], errors="coerce")

    player_ok = player_clean.ne("") & player_clean.ne("nan")
    market_ok = market_num.notna()

    valid_mask = player_ok & season_ok & market_ok
    invalid_mask = ~valid_mask

    report["column_stats"] = {
        "player_non_empty": int(player_ok.sum()),
        "season_valid_or_fallback": int(season_ok.sum()),
        "season_mode": season_mode,
        "market_value_numeric": int(market_ok.sum()),
    }
    report["usable_rows"] = int(valid_mask.sum())
    report["invalid_rows"] = int(invalid_mask.sum())

    invalid_df = df.loc[invalid_mask].copy()
    invalid_df.insert(0, "invalid_player", (~player_ok[invalid_mask]).astype(int))
    invalid_df.insert(1, "invalid_season", (~season_ok[invalid_mask]).astype(int))
    invalid_df.insert(2, "invalid_market_value", (~market_ok[invalid_mask]).astype(int))

    out_json = out_dir / "data_quality_report.json"
    out_csv = out_dir / "invalid_rows.csv"
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    invalid_df.to_csv(out_csv, index=False)

    print(f"Saved: {out_json}")
    print(f"Saved: {out_csv}")
    print(f"Usable rows: {report['usable_rows']} / {report['total_rows']}")


if __name__ == "__main__":
    main()
