from __future__ import annotations

import argparse
import os
from io import StringIO

import pandas as pd
import requests

FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"


def fetchFredSeries(series_id: str) -> pd.DataFrame:
    """
    Fetch a time series from FRED and return it as a cleaned DataFrame.

    Parameters
    ----------
    series_id:
        FRED series identifier (e.g., "PCOFFOTMUSDM").

    Returns
    -------
    pd.DataFrame
        A DataFrame with standardized columns:
        - date: datetime64[ns]
        - value: float
        Rows with missing values are dropped and the series is sorted by date.

    Raises
    ------
    requests.HTTPError
        If the FRED endpoint returns a non-200 status code.
    ValueError
        If the returned CSV does not have the expected two-column format.
    """
    url = FRED_CSV_URL.format(series_id=series_id)
    response = requests.get(url, timeout=30)
    response.raise_for_status()

    df = pd.read_csv(StringIO(response.text))
    if df.shape[1] != 2:
        raise ValueError(f"Unexpected CSV format for {series_id}. Columns: {df.columns.tolist()}")

    df.columns = ["date", "value"]
    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna().sort_values("date").reset_index(drop=True)
    return df


def writeSeriesToCsv(
    series_id: str,
    out_csv: str,
    asset_key: str | None = None,
    asset_name: str | None = None,
) -> None:
    """
    Fetch a FRED series and write it to a local CSV file.

    Parameters
    ----------
    series_id:
        FRED series identifier.
    out_csv:
        Output path (e.g., "data/raw/coffee.csv").
    """
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df = fetchFredSeries(series_id)
    if asset_key:
        df["asset_key"] = asset_key
    if asset_name:
        df["asset_name"] = asset_name
    df.to_csv(out_csv, index=False)
    print(f"Wrote {len(df):,} rows -> {out_csv}")


def main() -> None:
    """
    CLI entry point.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--series-id", required=True)
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--asset-key")
    parser.add_argument("--asset-name")
    args = parser.parse_args()

    series_id = args.series_id
    out_csv = args.out_csv
    writeSeriesToCsv(
        series_id,
        out_csv,
        asset_key=args.asset_key,
        asset_name=args.asset_name,
    )


if __name__ == "__main__":
    main()
