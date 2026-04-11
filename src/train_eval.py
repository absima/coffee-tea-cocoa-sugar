from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error


def TimeSplit(df: pd.DataFrame, test_size: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split a dataframe into chronological train and test partitions.

    Args:
        df: pd.DataFrame.
        Time-ordered feature dataframe.
        test_size: float.
        Fraction of rows reserved for testing.

    Returns:
        train_test_split: tuple[pd.DataFrame, pd.DataFrame].
        Train and test dataframes split without shuffling.
    """
    n = len(df)
    n_test = max(1, int(round(n * test_size)))
    train = df.iloc[: n - n_test].copy()
    test = df.iloc[n - n_test :].copy()
    return train, test


def main(features_csv: str, out_metrics: str, out_preds: str, test_size: float) -> None:
    """
    Train the baseline model and write prediction artifacts.

    Args:
        features_csv: str.
        Input feature dataset path.
        out_metrics: str.
        Output JSON path for evaluation metrics.
        out_preds: str.
        Output CSV path for test predictions.
        test_size: float.
        Fraction of rows reserved for the test period.

    Returns:
        None.
        Writes model metrics and test-period predictions to disk.
    """
    df = pd.read_csv(features_csv, parse_dates=["date"]).sort_values("date")

    asset_key = df["asset_key"].iloc[0] if "asset_key" in df.columns and not df.empty else "unknown"
    asset_name = df["asset_name"].iloc[0] if "asset_name" in df.columns and not df.empty else asset_key

    target = "y_next_return"
    drop_cols = {"date", "value", "log_price", "log_return", target, "asset_key", "asset_name"}
    feature_cols = [c for c in df.columns if c not in drop_cols]

    train, test = TimeSplit(df, test_size=test_size)

    X_train = train[feature_cols].to_numpy()
    y_train = train[target].to_numpy()
    X_test = test[feature_cols].to_numpy()
    y_test = test[target].to_numpy()

    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae = float(mean_absolute_error(y_test, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    corr = float(np.corrcoef(y_test, y_pred)[0, 1]) if len(y_test) > 1 else 0.0
    if np.isnan(corr):
        corr = 0.0
    directional_accuracy = float(((y_test > 0) == (y_pred > 0)).mean())
    y_true_std = float(np.std(y_test, ddof=1)) if len(y_test) > 1 else 0.0
    y_pred_std = float(np.std(y_pred, ddof=1)) if len(y_pred) > 1 else 0.0

    metrics = {
        "asset_key": asset_key,
        "asset_name": asset_name,
        "rows_total": int(len(df)),
        "rows_train": int(len(train)),
        "rows_test": int(len(test)),
        "model": "Ridge(alpha=1.0)",
        "mae": mae,
        "rmse": rmse,
        "correlation": corr,
        "directional_accuracy": directional_accuracy,
        "y_true_std": y_true_std,
        "y_pred_std": y_pred_std,
        "test_period_start": str(test["date"].min().date()),
        "test_period_end": str(test["date"].max().date()),
        "n_features": int(len(feature_cols)),
        "features_used": feature_cols,
    }

    os.makedirs(os.path.dirname(out_metrics), exist_ok=True)
    with open(out_metrics, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    os.makedirs(os.path.dirname(out_preds), exist_ok=True)
    pred_df = test[["date"]].copy()
    pred_df["asset_key"] = asset_key
    pred_df["asset_name"] = asset_name
    pred_df["y_true"] = y_test
    pred_df["y_pred"] = y_pred
    pred_df.to_csv(out_preds, index=False)

    print(f"Wrote metrics -> {out_metrics}")
    print(f"Wrote predictions -> {out_preds}")
    print(f"MAE={mae:.6f} RMSE={rmse:.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-csv", required=True)
    parser.add_argument("--out-metrics", required=True)
    parser.add_argument("--out-preds", required=True)
    parser.add_argument("--test-size", type=float, default=0.2)
    args = parser.parse_args()

    main(args.features_csv, args.out_metrics, args.out_preds, args.test_size)
