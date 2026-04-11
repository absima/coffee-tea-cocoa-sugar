from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def LoadMetrics(metrics_paths: list[str]) -> pd.DataFrame:
    """
    Load per-asset metrics JSON files into one dataframe.

    Args:
        metrics_paths: list[str].
        Paths to per-asset metrics JSON files.

    Returns:
        metrics_df: pd.DataFrame.
        Combined metrics table with one row per asset.
    """
    rows = []
    for path in metrics_paths:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        payload["metrics_path"] = path
        rows.append(payload)
    return pd.DataFrame(rows)


def LoadPredictions(preds_paths: list[str]) -> pd.DataFrame:
    """
    Load per-asset prediction CSV files into one dataframe.

    Args:
        preds_paths: list[str].
        Paths to per-asset prediction CSV files.

    Returns:
        preds_df: pd.DataFrame.
        Combined predictions table with asset metadata attached.
    """
    frames = []
    for path in preds_paths:
        df = pd.read_csv(path, parse_dates=["date"])
        if "asset_key" not in df.columns:
            df["asset_key"] = Path(path).stem
        if "asset_name" not in df.columns:
            df["asset_name"] = df["asset_key"]
        df["preds_path"] = path
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def BuildRecentBehaviour(preds: pd.DataFrame, recent_months: int) -> pd.DataFrame:
    """
    Summarize recent observed behavior and forecast error for each asset.

    Args:
        preds: pd.DataFrame.
        Combined prediction dataframe across assets.
        recent_months: int.
        Number of recent observations to summarize per asset.

    Returns:
        recent_df: pd.DataFrame.
        Per-asset recent return, volatility, and error summary table.
    """
    recent_rows = []
    for asset_key, asset_df in preds.groupby("asset_key"):
        asset_df = asset_df.sort_values("date").copy()
        recent = asset_df.tail(recent_months).copy()
        recent["abs_err"] = (recent["y_true"] - recent["y_pred"]).abs()

        recent_rows.append(
            {
                "asset_key": asset_key,
                "asset_name": recent["asset_name"].iloc[0],
                "recent_rows": int(len(recent)),
                "recent_start": str(recent["date"].min().date()) if len(recent) else None,
                "recent_end": str(recent["date"].max().date()) if len(recent) else None,
                "recent_mean_return": float(recent["y_true"].mean()) if len(recent) else None,
                "recent_volatility": float(recent["y_true"].std(ddof=1)) if len(recent) > 1 else 0.0,
                "recent_last_return": float(recent["y_true"].iloc[-1]) if len(recent) else None,
                "recent_mean_abs_error": float(recent["abs_err"].mean()) if len(recent) else None,
                "recent_max_abs_error": float(recent["abs_err"].max()) if len(recent) else None,
                "recent_positive_share": float((recent["y_true"] > 0).mean()) if len(recent) else None,
            }
        )
    return pd.DataFrame(recent_rows)


def BuildCorrelationMatrix(preds: pd.DataFrame) -> dict[str, dict[str, float | None]]:
    """
    Compute the cross-asset correlation matrix for observed returns.

    Args:
        preds: pd.DataFrame.
        Combined prediction dataframe containing observed returns.

    Returns:
        correlation_matrix: dict[str, dict[str, float | None]].
        Nested mapping of asset-name pairs to correlation values.
    """
    pivot = preds.pivot_table(index="date", columns="asset_name", values="y_true", aggfunc="first")
    corr = pivot.corr()
    matrix: dict[str, dict[str, float | None]] = {}
    for row_name, row in corr.iterrows():
        matrix[row_name] = {}
        for col_name, value in row.items():
            matrix[row_name][col_name] = None if pd.isna(value) else float(value)
    return matrix


def BuildSummaryBundle(summary: pd.DataFrame, preds: pd.DataFrame, recent_months: int) -> dict:
    """
    Build the grounded summary bundle used by the cross-asset report step.

    Args:
        summary: pd.DataFrame.
        Side-by-side per-asset metrics and recent behavior table.
        preds: pd.DataFrame.
        Combined prediction dataframe across assets.
        recent_months: int.
        Number of recent months represented in the summary.

    Returns:
        bundle: dict.
        JSON-serializable cross-asset summary for reporting.
    """
    best_rmse = summary.sort_values("rmse").iloc[0]
    best_directional = summary.sort_values("directional_accuracy", ascending=False).iloc[0]
    highest_recent_vol = summary.sort_values("recent_volatility", ascending=False).iloc[0]

    return {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "target_variable": "monthly log returns (not prices)",
        "recent_window_months": int(recent_months),
        "asset_count": int(len(summary)),
        "leaders": {
            "lowest_rmse": {
                "asset": best_rmse["asset_name"],
                "rmse": float(best_rmse["rmse"]),
            },
            "best_directional_accuracy": {
                "asset": best_directional["asset_name"],
                "directional_accuracy": float(best_directional["directional_accuracy"]),
            },
            "highest_recent_volatility": {
                "asset": highest_recent_vol["asset_name"],
                "recent_volatility": float(highest_recent_vol["recent_volatility"]),
            },
        },
        "cross_asset_return_correlation": BuildCorrelationMatrix(preds),
        "assets": summary.sort_values("asset_name").to_dict(orient="records"),
    }


def main(
    metrics_json: list[str],
    preds_csv: list[str],
    out_csv: str,
    out_json: str,
    recent_months: int,
) -> None:
    """
    Build cross-asset comparison artifacts from per-asset outputs.

    Args:
        metrics_json: list[str].
        Paths to per-asset metrics JSON files.
        preds_csv: list[str].
        Paths to per-asset prediction CSV files.
        out_csv: str.
        Output path for the comparison table.
        out_json: str.
        Output path for the grounded comparison bundle.
        recent_months: int.
        Number of recent months to summarize per asset.

    Returns:
        None.
        Writes the comparison table and JSON bundle to disk.
    """
    metrics_df = LoadMetrics(metrics_json)
    preds_df = LoadPredictions(preds_csv)

    if metrics_df.empty or preds_df.empty:
        raise ValueError("Expected non-empty metrics and prediction inputs for cross-asset comparison.")

    recent_df = BuildRecentBehaviour(preds_df, recent_months=recent_months)

    summary = metrics_df.merge(recent_df, on=["asset_key", "asset_name"], how="left")
    summary["prediction_volatility_ratio"] = summary["y_pred_std"] / summary["y_true_std"].replace(0, pd.NA)
    summary = summary.sort_values("rmse").reset_index(drop=True)

    bundle = BuildSummaryBundle(summary, preds_df, recent_months=recent_months)

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    summary.to_csv(out_csv, index=False)

    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(bundle, f, indent=2)

    print(f"Wrote cross-asset summary table -> {out_csv}")
    print(f"Wrote cross-asset summary bundle -> {out_json}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics-json", nargs="+", required=True)
    parser.add_argument("--preds-csv", nargs="+", required=True)
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--recent-months", type=int, default=12)
    args = parser.parse_args()

    main(
        metrics_json=args.metrics_json,
        preds_csv=args.preds_csv,
        out_csv=args.out_csv,
        out_json=args.out_json,
        recent_months=args.recent_months,
    )
