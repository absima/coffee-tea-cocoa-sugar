from __future__ import annotations

import argparse
import os
from io import StringIO

import pandas as pd
import requests

FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"


def FetchFredSeries(series_id: str) -> pd.DataFrame:
    """
    Fetch a FRED series and return a cleaned two-column dataframe.

    Args:
        series_id: str.
        FRED series identifier, such as "PCOFFOTMUSDM".

    Returns:
        df: pd.DataFrame.
        Dataframe with normalized `date` and `value` columns sorted by date.
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


def WriteSeriesToCsv(
    series_id: str,
    out_csv: str,
    asset_key: str | None = None,
    asset_name: str | None = None,
) -> None:
    """
    Fetch a FRED series, attach asset metadata, and write it to CSV.

    Args:
        series_id: str.
        FRED series identifier to download.
        out_csv: str.
        Destination CSV path.
        asset_key: str | None.
        Short machine-friendly asset identifier.
        asset_name: str | None.
        Human-readable asset label.

    Returns:
        None.
        Writes the fetched series to disk.
    """
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df = FetchFredSeries(series_id)
    if asset_key:
        df["asset_key"] = asset_key
    if asset_name:
        df["asset_name"] = asset_name
    df.to_csv(out_csv, index=False)
    print(f"Wrote {len(df):,} rows -> {out_csv}")


def main() -> None:
    """
    Parse CLI arguments and run the fetch step.

    Args:
        None.

    Returns:
        None.
        Executes the ingestion CLI workflow.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--series-id", required=True)
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--asset-key")
    parser.add_argument("--asset-name")
    args = parser.parse_args()

    series_id = args.series_id
    out_csv = args.out_csv
    WriteSeriesToCsv(
        series_id,
        out_csv,
        asset_key=args.asset_key,
        asset_name=args.asset_name,
    )


if __name__ == "__main__":
    main()
