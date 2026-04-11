from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


plt.style.use("seaborn-v0_8-whitegrid")


def load_metrics(metrics_paths: list[str]) -> pd.DataFrame:
    rows = []
    for path in metrics_paths:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        payload["metrics_path"] = path
        rows.append(payload)
    return pd.DataFrame(rows).sort_values("asset_name").reset_index(drop=True)


def load_predictions(pred_paths: list[str]) -> pd.DataFrame:
    frames = []
    for path in pred_paths:
        df = pd.read_csv(path, parse_dates=["date"])
        if "asset_key" not in df.columns:
            df["asset_key"] = Path(path).stem
        if "asset_name" not in df.columns:
            df["asset_name"] = df["asset_key"]
        frames.append(df)
    return pd.concat(frames, ignore_index=True).sort_values(["asset_name", "date"]).reset_index(drop=True)


def plot_recent_returns(preds: pd.DataFrame, out_path: str, recent_months: int) -> None:
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


def plot_model_metrics(metrics: pd.DataFrame, out_path: str) -> None:
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


def plot_prediction_compression(metrics: pd.DataFrame, out_path: str) -> None:
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


def plot_actual_vs_predicted(preds: pd.DataFrame, out_path: str, recent_months: int) -> None:
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
    metrics = load_metrics(metrics_json)
    preds = load_predictions(preds_csv)

    if metrics.empty or preds.empty:
        raise ValueError("Expected non-empty metrics and predictions for plot generation.")

    os.makedirs(out_dir, exist_ok=True)
    plot_recent_returns(preds, os.path.join(out_dir, "recent_returns.png"), recent_months)
    plot_model_metrics(metrics, os.path.join(out_dir, "model_metrics.png"))
    plot_prediction_compression(metrics, os.path.join(out_dir, "prediction_compression.png"))
    plot_actual_vs_predicted(preds, os.path.join(out_dir, "actual_vs_predicted.png"), recent_months)

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
