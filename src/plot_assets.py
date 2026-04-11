from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


plt.style.use("seaborn-v0_8-whitegrid")


def LoadMetrics(metrics_paths: list[str]) -> pd.DataFrame:
    """
    Load per-asset metrics JSON files into a plotting dataframe.

    Args:
        metrics_paths: list[str].
        Paths to metrics JSON files.

    Returns:
        metrics_df: pd.DataFrame.
        Combined metrics table sorted by asset name.
    """
    rows = []
    for path in metrics_paths:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        payload["metrics_path"] = path
        rows.append(payload)
    return pd.DataFrame(rows).sort_values("asset_name").reset_index(drop=True)


def LoadPredictions(pred_paths: list[str]) -> pd.DataFrame:
    """
    Load per-asset prediction CSV files into a plotting dataframe.

    Args:
        pred_paths: list[str].
        Paths to prediction CSV files.

    Returns:
        preds_df: pd.DataFrame.
        Combined prediction table sorted by asset and date.
    """
    frames = []
    for path in pred_paths:
        df = pd.read_csv(path, parse_dates=["date"])
        if "asset_key" not in df.columns:
            df["asset_key"] = Path(path).stem
        if "asset_name" not in df.columns:
            df["asset_name"] = df["asset_key"]
        frames.append(df)
    return pd.concat(frames, ignore_index=True).sort_values(["asset_name", "date"]).reset_index(drop=True)


def PlotRecentReturns(preds: pd.DataFrame, out_path: str, recent_months: int) -> None:
    """
    Plot recent observed returns for each commodity on one chart.

    Args:
        preds: pd.DataFrame.
        Combined prediction dataframe across assets.
        out_path: str.
        Output image path.
        recent_months: int.
        Number of recent observations to plot per asset.

    Returns:
        None.
        Writes the recent-returns chart to disk.
    """
    fig, ax = plt.subplots(figsize=(11, 6))
    for asset_name, asset_df in preds.groupby("asset_name"):
        recent = asset_df.sort_values("date").tail(recent_months)
        ax.plot(recent["date"], recent["y_true"], marker="o", linewidth=2, label=asset_name)

    ax.axhline(0.0, color="black", linewidth=1, alpha=0.5)
    ax.set_title(f"Recent Monthly Returns by Commodity (Last {recent_months} Months)")
    ax.set_ylabel("Observed log return")
    ax.set_xlabel("Date")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def PlotModelMetrics(metrics: pd.DataFrame, out_path: str) -> None:
    """
    Plot side-by-side model quality bars for each commodity.

    Args:
        metrics: pd.DataFrame.
        Per-asset metrics dataframe.
        out_path: str.
        Output image path.

    Returns:
        None.
        Writes the model-metrics chart to disk.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    metrics = metrics.sort_values("rmse")

    axes[0].bar(metrics["asset_name"], metrics["rmse"], color="#5B8FF9")
    axes[0].set_title("RMSE by Commodity")
    axes[0].set_ylabel("RMSE")
    axes[0].tick_params(axis="x", rotation=20)

    axes[1].bar(metrics["asset_name"], metrics["directional_accuracy"], color="#5AD8A6")
    axes[1].set_title("Directional Accuracy by Commodity")
    axes[1].set_ylabel("Share of correct directions")
    axes[1].set_ylim(0, 1)
    axes[1].tick_params(axis="x", rotation=20)

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def PlotPredictionCompression(metrics: pd.DataFrame, out_path: str) -> None:
    """
    Plot prediction volatility relative to observed volatility by asset.

    Args:
        metrics: pd.DataFrame.
        Per-asset metrics dataframe.
        out_path: str.
        Output image path.

    Returns:
        None.
        Writes the compression chart to disk.
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    ratios = metrics["y_pred_std"] / metrics["y_true_std"].replace(0, pd.NA)
    ax.bar(metrics["asset_name"], ratios, color="#F6BD16")
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1, alpha=0.6)
    ax.set_title("Prediction Volatility Ratio")
    ax.set_ylabel("Predicted std / observed std")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def PlotActualVsPredicted(preds: pd.DataFrame, out_path: str, recent_months: int) -> None:
    """
    Plot recent observed and predicted returns in one panel per asset.

    Args:
        preds: pd.DataFrame.
        Combined prediction dataframe across assets.
        out_path: str.
        Output image path.
        recent_months: int.
        Number of recent observations to show per asset.

    Returns:
        None.
        Writes the multi-panel observed-vs-predicted chart to disk.
    """
    assets = list(preds["asset_name"].drop_duplicates())
    n_assets = len(assets)
    n_cols = 2
    n_rows = math.ceil(n_assets / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(13, 4.5 * n_rows), squeeze=False)
    axes_flat = axes.flatten()

    for idx, asset_name in enumerate(assets):
        ax = axes_flat[idx]
        asset_df = preds[preds["asset_name"] == asset_name].sort_values("date").tail(recent_months)
        ax.plot(asset_df["date"], asset_df["y_true"], marker="o", linewidth=2, label="Observed")
        ax.plot(asset_df["date"], asset_df["y_pred"], marker="o", linewidth=2, linestyle="--", label="Predicted")
        ax.axhline(0.0, color="black", linewidth=1, alpha=0.4)
        ax.set_title(asset_name)
        ax.set_ylabel("Log return")
        ax.tick_params(axis="x", rotation=25)

    for idx in range(n_assets, len(axes_flat)):
        axes_flat[idx].axis("off")

    axes_flat[0].legend(loc="upper left", frameon=True)
    fig.suptitle(f"Observed vs Predicted Returns (Last {recent_months} Months)", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main(metrics_json: list[str], preds_csv: list[str], out_dir: str, recent_months: int) -> None:
    """
    Generate all cross-asset comparison plots for the reporting stage.

    Args:
        metrics_json: list[str].
        Paths to per-asset metrics JSON files.
        preds_csv: list[str].
        Paths to per-asset prediction CSV files.
        out_dir: str.
        Output directory for generated plots.
        recent_months: int.
        Number of recent observations to visualize.

    Returns:
        None.
        Writes the plot images to the output directory.
    """
    metrics = LoadMetrics(metrics_json)
    preds = LoadPredictions(preds_csv)

    if metrics.empty or preds.empty:
        raise ValueError("Expected non-empty metrics and predictions for plot generation.")

    os.makedirs(out_dir, exist_ok=True)
    PlotRecentReturns(preds, os.path.join(out_dir, "recent_returns.png"), recent_months)
    PlotModelMetrics(metrics, os.path.join(out_dir, "model_metrics.png"))
    PlotPredictionCompression(metrics, os.path.join(out_dir, "prediction_compression.png"))
    PlotActualVsPredicted(preds, os.path.join(out_dir, "actual_vs_predicted.png"), recent_months)

    print(f"Wrote plots -> {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics-json", nargs="+", required=True)
    parser.add_argument("--preds-csv", nargs="+", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--recent-months", type=int, default=12)
    args = parser.parse_args()

    main(
        metrics_json=args.metrics_json,
        preds_csv=args.preds_csv,
        out_dir=args.out_dir,
        recent_months=args.recent_months,
    )
