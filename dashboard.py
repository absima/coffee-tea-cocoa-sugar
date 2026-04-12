from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st


ROOT_DIR = Path(__file__).resolve().parent
REPORTS_DIR = ROOT_DIR / "reports"
METRICS_DIR = REPORTS_DIR / "metrics"
PREDS_DIR = REPORTS_DIR / "preds"
NOTES_DIR = REPORTS_DIR / "notes"
PLOTS_DIR = REPORTS_DIR / "plots"


def LoadJson(path: Path) -> dict:
    """
    Load a JSON file from disk.

    Args:
        path: Path.
        Filesystem path to the JSON file.

    Returns:
        payload: dict.
        Parsed JSON content.
    """
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def LoadText(path: Path) -> str:
    """
    Load a text file from disk.

    Args:
        path: Path.
        Filesystem path to the text file.

    Returns:
        text: str.
        File contents as a UTF-8 string.
    """
    return path.read_text(encoding="utf-8")


def LoadMetricsTable() -> pd.DataFrame:
    """
    Load all per-asset metrics into one dataframe.

    Args:
        None.

    Returns:
        metrics_df: pd.DataFrame.
        Combined metrics table sorted by asset name, or an empty dataframe if missing.
    """
    metric_paths = sorted(METRICS_DIR.glob("*.json"))
    if not metric_paths:
        return pd.DataFrame()

    rows = [LoadJson(path) for path in metric_paths]
    return pd.DataFrame(rows).sort_values("asset_name").reset_index(drop=True)


def LoadCrossAssetTable() -> pd.DataFrame:
    """
    Load the cross-asset comparison CSV if it exists.

    Args:
        None.

    Returns:
        comparison_df: pd.DataFrame.
        Cross-asset summary table, or an empty dataframe if missing.
    """
    path = REPORTS_DIR / "cross_asset_metrics.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def ListAssets(metrics_df: pd.DataFrame) -> list[str]:
    """
    Extract the available asset names from the metrics table.

    Args:
        metrics_df: pd.DataFrame.
        Combined per-asset metrics dataframe.

    Returns:
        asset_names: list[str].
        Sorted asset display names available in the reports.
    """
    if metrics_df.empty or "asset_name" not in metrics_df.columns:
        return []
    return sorted(metrics_df["asset_name"].dropna().unique().tolist())


def ResolveAssetKey(metrics_df: pd.DataFrame, asset_name: str) -> str | None:
    """
    Map a display asset name back to its asset key.

    Args:
        metrics_df: pd.DataFrame.
        Combined per-asset metrics dataframe.
        asset_name: str.
        Human-readable asset name selected in the UI.

    Returns:
        asset_key: str | None.
        Machine-friendly asset key, or None if unavailable.
    """
    if metrics_df.empty:
        return None
    matches = metrics_df.loc[metrics_df["asset_name"] == asset_name, "asset_key"]
    if matches.empty:
        return None
    return str(matches.iloc[0])


def RenderOverview(metrics_df: pd.DataFrame, comparison_df: pd.DataFrame) -> None:
    """
    Render the high-level dashboard summary and comparison table.

    Args:
        metrics_df: pd.DataFrame.
        Combined per-asset metrics dataframe.
        comparison_df: pd.DataFrame.
        Cross-asset comparison dataframe.

    Returns:
        None.
        Writes the overview section into the Streamlit app.
    """
    st.subheader("Overview")

    if metrics_df.empty:
        st.warning("No per-asset metrics found yet. Run the pipeline first.")
        return

    col_1, col_2, col_3 = st.columns(3)
    col_1.metric("Tracked commodities", len(metrics_df))
    col_2.metric("Best RMSE", f"{metrics_df['rmse'].min():.4f}")
    col_3.metric("Best directional accuracy", f"{metrics_df['directional_accuracy'].max():.1%}")

    display_columns = [
        "asset_name",
        "rmse",
        "mae",
        "directional_accuracy",
        "correlation",
        "y_true_std",
        "y_pred_std",
    ]
    existing_columns = [column for column in display_columns if column in metrics_df.columns]
    st.dataframe(metrics_df[existing_columns], use_container_width=True)

    st.subheader("Cross-Asset Comparison Table")
    if comparison_df.empty:
        st.info("`reports/cross_asset_metrics.csv` is not available yet.")
    else:
        st.dataframe(comparison_df, use_container_width=True)


def RenderPlots() -> None:
    """
    Render generated plot artifacts if they exist.

    Args:
        None.

    Returns:
        None.
        Writes the plots section into the Streamlit app.
    """
    st.subheader("Plots")
    plot_specs = [
        ("Recent Returns", PLOTS_DIR / "recent_returns.png"),
        ("Model Metrics", PLOTS_DIR / "model_metrics.png"),
        ("Prediction Compression", PLOTS_DIR / "prediction_compression.png"),
        ("Actual vs Predicted", PLOTS_DIR / "actual_vs_predicted.png"),
    ]

    for title, path in plot_specs:
        st.markdown(f"**{title}**")
        if path.exists():
            st.image(str(path), use_container_width=True)
        else:
            st.info(f"Missing plot: `{path.relative_to(ROOT_DIR)}`")


def RenderAssetNotes(metrics_df: pd.DataFrame) -> None:
    """
    Render a selector for per-asset notes and prediction samples.

    Args:
        metrics_df: pd.DataFrame.
        Combined per-asset metrics dataframe.

    Returns:
        None.
        Writes the per-asset drilldown section into the Streamlit app.
    """
    st.subheader("Per-Asset Drilldown")
    assets = ListAssets(metrics_df)
    if not assets:
        st.info("No per-asset metrics are available yet.")
        return

    selected_asset = st.selectbox("Select a commodity", assets)
    asset_key = ResolveAssetKey(metrics_df, selected_asset)
    if asset_key is None:
        st.warning("Unable to resolve the selected asset key.")
        return

    asset_metrics = metrics_df.loc[metrics_df["asset_key"] == asset_key]
    if not asset_metrics.empty:
        st.json(asset_metrics.iloc[0].to_dict(), expanded=False)

    note_path = NOTES_DIR / f"{asset_key}.md"
    preds_path = PREDS_DIR / f"{asset_key}.csv"

    note_col, preds_col = st.columns([1, 1])

    with note_col:
        st.markdown("**LLM Note**")
        if note_path.exists():
            st.markdown(LoadText(note_path))
        else:
            st.info(f"Missing note: `{note_path.relative_to(ROOT_DIR)}`")

    with preds_col:
        st.markdown("**Recent Predictions**")
        if preds_path.exists():
            preds_df = pd.read_csv(preds_path).tail(12)
            st.dataframe(preds_df, use_container_width=True)
        else:
            st.info(f"Missing predictions: `{preds_path.relative_to(ROOT_DIR)}`")


def RenderCrossAssetNote() -> None:
    """
    Render the cross-asset markdown note if it exists.

    Args:
        None.

    Returns:
        None.
        Writes the cross-asset note section into the Streamlit app.
    """
    st.subheader("Cross-Asset LLM Note")
    note_path = REPORTS_DIR / "cross_asset_note.md"
    if note_path.exists():
        st.markdown(LoadText(note_path))
    else:
        st.info("`reports/cross_asset_note.md` is not available yet.")


def main() -> None:
    """
    Launch the Streamlit dashboard for pipeline outputs.

    Args:
        None.

    Returns:
        None.
        Renders the dashboard UI from generated report artifacts.
    """
    st.set_page_config(page_title="Commodity Analytics Dashboard", layout="wide")
    st.title("Commodity Market Analytics Dashboard")
    st.caption("A local dashboard for pipeline outputs, plots, and grounded LLM notes.")

    metrics_df = LoadMetricsTable()
    comparison_df = LoadCrossAssetTable()

    RenderOverview(metrics_df, comparison_df)
    st.divider()
    RenderPlots()
    st.divider()
    RenderAssetNotes(metrics_df)
    st.divider()
    RenderCrossAssetNote()


if __name__ == "__main__":
    main()
