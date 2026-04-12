from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def LoadJson(path: Path) -> dict:
    """
    Load a JSON file from disk.

    Args:
        path: Path.
        Filesystem path to a JSON file.

    Returns:
        payload: dict.
        Parsed JSON content.
    """
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def LoadText(path: Path) -> str:
    """
    Load a UTF-8 text file from disk.

    Args:
        path: Path.
        Filesystem path to a text file.

    Returns:
        text: str.
        File contents as a string.
    """
    return path.read_text(encoding="utf-8")


def DataFrameToRecords(df: pd.DataFrame, rows: int = 8) -> list[dict]:
    """
    Convert the head of a dataframe into JSON-friendly records.

    Args:
        df: pd.DataFrame.
        Input dataframe.
        rows: int.
        Number of top rows to include.

    Returns:
        records: list[dict].
        Head rows converted to JSON-serializable dictionaries.
    """
    subset = df.head(rows).copy()
    for column in subset.columns:
        if pd.api.types.is_datetime64_any_dtype(subset[column]):
            subset[column] = subset[column].dt.strftime("%Y-%m-%d")
    return json.loads(subset.to_json(orient="records"))


def BuildMetricsSummary(metrics_paths: list[Path]) -> tuple[list[dict], list[str]]:
    """
    Load per-asset metrics and return summary rows plus asset order.

    Args:
        metrics_paths: list[Path].
        Paths to per-asset metrics JSON files.

    Returns:
        metrics_summary: tuple[list[dict], list[str]].
        Summary rows for display and the corresponding asset key order.
    """
    metrics_payloads = [LoadJson(path) for path in metrics_paths]
    metrics_df = pd.DataFrame(metrics_payloads).sort_values("asset_name").reset_index(drop=True)

    summary_columns = [
        "asset_key",
        "asset_name",
        "rmse",
        "mae",
        "directional_accuracy",
        "correlation",
        "rows_train",
        "rows_test",
        "test_period_start",
        "test_period_end",
    ]
    summary_df = metrics_df[summary_columns].copy()
    summary_df["directional_accuracy"] = summary_df["directional_accuracy"].round(4)
    summary_df["correlation"] = summary_df["correlation"].round(4)
    summary_df["rmse"] = summary_df["rmse"].round(4)
    summary_df["mae"] = summary_df["mae"].round(4)

    return json.loads(summary_df.to_json(orient="records")), metrics_df["asset_key"].tolist()


def BuildNotesByAsset(asset_keys: list[str], notes_dir: Path) -> dict[str, str]:
    """
    Load per-asset markdown notes keyed by asset id.

    Args:
        asset_keys: list[str].
        Asset identifiers to load notes for.
        notes_dir: Path.
        Directory containing per-asset markdown notes.

    Returns:
        notes_by_asset: dict[str, str].
        Mapping from asset key to markdown note text.
    """
    return {
        asset_key: LoadText(notes_dir / f"{asset_key}.md")
        for asset_key in asset_keys
        if (notes_dir / f"{asset_key}.md").exists()
    }


def main(
    raw_csv: str,
    features_csv: str,
    metrics_json: list[str],
    cross_asset_csv: str,
    cross_asset_note: str,
    notes_dir: str,
    out_json: str,
    plot_base_path: str,
) -> None:
    """
    Build the JSON payload used by the static web page.

    Args:
        raw_csv: str.
        Path to a representative raw asset CSV.
        features_csv: str.
        Path to a representative engineered feature CSV.
        metrics_json: list[str].
        Paths to per-asset metrics JSON files.
        cross_asset_csv: str.
        Path to the cross-asset comparison CSV.
        cross_asset_note: str.
        Path to the cross-asset markdown note.
        notes_dir: str.
        Directory containing per-asset markdown notes.
        out_json: str.
        Destination path for the generated web payload.
        plot_base_path: str.
        Base relative path used for plot image URLs in the web payload.

    Returns:
        None.
        Writes the site data JSON used by the static web page.
    """
    raw_df = pd.read_csv(raw_csv, parse_dates=["date"])
    features_df = pd.read_csv(features_csv, parse_dates=["date"])
    comparison_df = pd.read_csv(cross_asset_csv)

    metrics_paths = [Path(path) for path in metrics_json]
    metrics_summary, asset_keys = BuildMetricsSummary(metrics_paths)
    notes_by_asset = BuildNotesByAsset(asset_keys, Path(notes_dir))

    payload = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "steps": [
            {
                "title": "Ingest",
                "description": "Pull monthly commodity series from FRED and standardize them into raw CSV files.",
            },
            {
                "title": "Store",
                "description": "Load each commodity into SQLite so downstream steps consume a stable local source.",
            },
            {
                "title": "Features",
                "description": "Engineer lagged and rolling-return features for one-step-ahead forecasting.",
            },
            {
                "title": "Train & Evaluate",
                "description": "Fit a Ridge baseline per commodity and compare forecast quality across assets.",
            },
            {
                "title": "Report",
                "description": "Generate grounded local-LLM notes plus cross-asset figures for the final demo layer.",
            },
        ],
        "raw_preview": {
            "title": "Coffee raw CSV head",
            "columns": list(raw_df.columns),
            "rows": DataFrameToRecords(raw_df, rows=8),
        },
        "feature_preview": {
            "title": "Engineered feature preview",
            "columns": list(features_df.columns),
            "rows": DataFrameToRecords(features_df, rows=8),
        },
        "metrics_summary": metrics_summary,
        "cross_asset_table": json.loads(comparison_df.round(4).to_json(orient="records")),
        "cross_asset_note": LoadText(Path(cross_asset_note)) if Path(cross_asset_note).exists() else "",
        "notes_by_asset": notes_by_asset,
        "plots": [
            {
                "title": "Recent Returns",
                "path": f"{plot_base_path}/recent_returns.png",
                "caption": "Recent observed monthly returns across the tracked commodities.",
            },
            {
                "title": "Model Metrics",
                "path": f"{plot_base_path}/model_metrics.png",
                "caption": "RMSE and directional accuracy by commodity.",
            },
            {
                "title": "Prediction Compression",
                "path": f"{plot_base_path}/prediction_compression.png",
                "caption": "How much the model compresses volatility relative to the real series.",
            },
            {
                "title": "Actual vs Predicted",
                "path": f"{plot_base_path}/actual_vs_predicted.png",
                "caption": "Recent observed versus predicted returns for each commodity.",
            },
        ],
    }

    out_path = Path(out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote web data payload -> {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-csv", required=True)
    parser.add_argument("--features-csv", required=True)
    parser.add_argument("--metrics-json", nargs="+", required=True)
    parser.add_argument("--cross-asset-csv", required=True)
    parser.add_argument("--cross-asset-note", required=True)
    parser.add_argument("--notes-dir", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--plot-base-path", default="./reports/plots")
    args = parser.parse_args()

    main(
        raw_csv=args.raw_csv,
        features_csv=args.features_csv,
        metrics_json=args.metrics_json,
        cross_asset_csv=args.cross_asset_csv,
        cross_asset_note=args.cross_asset_note,
        notes_dir=args.notes_dir,
        out_json=args.out_json,
        plot_base_path=args.plot_base_path,
    )
