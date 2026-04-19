import json
import re
import unicodedata
from pathlib import Path

import pandas as pd


def normalize_name(name: str) -> str:
    if pd.isna(name):
        return ""
    text = str(name).strip().lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def main():
    final_path = Path("final_modeling_dataset.csv")
    market_path = Path("marketvalue.csv")
    out_path = Path("final_modeling_dataset_enriched.csv")
    report_path = Path("outputs/week5_7/reports/market_merge_report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)

    if not final_path.exists() or not market_path.exists():
        print("Missing input files")
        return

    final_df = pd.read_csv(final_path)
    market_df = pd.read_csv(market_path)

    if "player" not in final_df.columns:
        print("final_modeling_dataset.csv is missing player column")
        return

    if not {"player", "market_value"}.issubset(set(market_df.columns)):
        print("marketvalue.csv must include player and market_value columns")
        return

    final_df["_player_norm"] = final_df["player"].map(normalize_name)
    market_df["_player_norm"] = market_df["player"].map(normalize_name)

    # Keep one market row per normalized name.
    market_map = (
        market_df.sort_values("market_value", ascending=False)
        .drop_duplicates(subset=["_player_norm"], keep="first")
        [["_player_norm", "market_value"] + (["season"] if "season" in market_df.columns else [])]
    )

    merged = final_df.merge(market_map, on="_player_norm", how="left", suffixes=("", "_from_market"))

    merged["market_value"] = pd.to_numeric(merged.get("market_value"), errors="coerce")
    merged["market_value_from_market"] = pd.to_numeric(
        merged.get("market_value_from_market"), errors="coerce"
    )

    before_non_null = int(merged["market_value"].notna().sum())
    merged["market_value"] = merged["market_value"].fillna(merged["market_value_from_market"])
    after_non_null = int(merged["market_value"].notna().sum())

    if "season" not in merged.columns and "season_from_market" in merged.columns:
        if merged["season_from_market"].notna().any():
            merged["season"] = pd.to_numeric(merged["season_from_market"], errors="coerce")

    if "season" in merged.columns:
        merged["season"] = pd.to_numeric(merged["season"], errors="coerce")

    match_count = int(merged["market_value_from_market"].notna().sum())

    drop_cols = [c for c in ["_player_norm", "market_value_from_market", "season_from_market"] if c in merged.columns]
    merged = merged.drop(columns=drop_cols)

    merged.to_csv(out_path, index=False)

    report = {
        "input_final_rows": int(len(final_df)),
        "input_market_rows": int(len(market_df)),
        "matched_rows_by_normalized_name": match_count,
        "market_value_non_null_before": before_non_null,
        "market_value_non_null_after": after_non_null,
        "output_file": str(out_path),
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Saved: {out_path}")
    print(f"Saved: {report_path}")
    print(f"market_value filled rows: {before_non_null} -> {after_non_null}")


if __name__ == "__main__":
    main()
