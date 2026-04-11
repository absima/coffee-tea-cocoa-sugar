from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone

import pandas as pd
import requests


SYSTEM_PROMPT = """You are an analytical assistant writing concise, professional market analysis notes.

Rules:
- Use only the provided numbers and facts.
- Do not speculate on causes.
- Do not give investment advice or trading recommendations.
- Use formal, professional language.
- Write dates in full, unambiguous form (e.g., "November 2024" or "November 2024–May 2025").
- Avoid abbreviations such as "Nov 24".

Output format (markdown):
1) Market behaviour (3–5 bullets)
2) Model performance (2–4 bullets)
3) Confidence & risks (2–4 bullets)
"""


def buildAnalysisBundle(metrics: dict, preds_csv: str, recent_months: int = 12) -> dict:
    """
    Build a compact JSON bundle for LLM consumption from evaluation artifacts.

    The goal is to provide the LLM only with summary facts (metrics + recent errors),
    rather than raw datasets, so the report is grounded and reproducible.

    Parameters
    ----------
    metrics:
        Dict loaded from reports/metrics.json.
    preds_csv:
        Path to reports/preds.csv containing date, y_true, y_pred.
    recent_months:
        Number of most recent test rows to summarize.

    Returns
    -------
    dict
        A JSON-serializable summary bundle.
    """
    preds = pd.read_csv(preds_csv, parse_dates=["date"]).sort_values("date")

    recent = preds.tail(recent_months).copy()
    recent["abs_err"] = (recent["y_true"] - recent["y_pred"]).abs()

    bundle = {
        "asset": metrics.get("asset_name", "Unknown asset"),
        "asset_key": metrics.get("asset_key", "unknown"),
        "target_variable": "monthly log returns (not prices)",
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "test_period": {
            "start": metrics.get("test_period_start"),
            "end": metrics.get("test_period_end"),
            "rows_test": metrics.get("rows_test"),
        },
        "model": metrics.get("model"),
        "metrics": {
            "mae": metrics.get("mae"),
            "rmse": metrics.get("rmse"),
            "correlation": metrics.get("correlation"),
            "directional_accuracy": metrics.get("directional_accuracy"),
            "y_true_std": metrics.get("y_true_std"),
            "y_pred_std": metrics.get("y_pred_std"),
            "n_features": metrics.get("n_features"),
        },
        "recent_window_months": int(recent_months),
        "recent_error_summary": {
            "mean_abs_error": float(recent["abs_err"].mean()) if len(recent) else None,
            "max_abs_error": float(recent["abs_err"].max()) if len(recent) else None,
            "last_date": str(recent["date"].max().date()) if len(recent) else None,
        },
        "recent_points": [
            {
                "date": str(row["date"].date()),
                "y_true": float(row["y_true"]),
                "y_pred": float(row["y_pred"]),
                "abs_err": float(row["abs_err"]),
            }
            for _, row in recent.iterrows()
        ],
    }
    return bundle


def callOllama(model: str, prompt: str, host: str = "http://localhost:11434") -> str:
    """
    Call a local Ollama server to generate a completion.

    Parameters
    ----------
    model:
        Ollama model name (e.g., "llama3.1:8b").
    prompt:
        Full prompt to send (system rules + user task + JSON bundle).
    host:
        Ollama host URL (default: http://localhost:11434).

    Returns
    -------
    str
        The generated text response.

    Raises
    ------
    RuntimeError
        If Ollama is not reachable or returns an unexpected response.
    """
    url = f"{host}/api/generate"
    payload = {"model": model, "prompt": prompt, "stream": False}

    try:
        response = requests.post(url, json=payload, timeout=120)
        response.raise_for_status()
    except requests.RequestException as e:
        raise RuntimeError(
            f"Failed to call Ollama at {host}. "
            f"Is Ollama running and is the host correct? Underlying error: {e}"
        ) from e

    data = response.json()
    text = (data.get("response") or "").strip()
    if not text:
        raise RuntimeError(f"Ollama returned an empty response. Raw payload keys: {list(data.keys())}")
    return text


def main(metrics_json: str, preds_csv: str, out_md: str, recent_months: int, model: str, host: str) -> None:
    """
    Generate an analyst-style markdown note from evaluation artifacts using a local LLM.
    """
    with open(metrics_json, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    bundle = buildAnalysisBundle(metrics, preds_csv, recent_months=recent_months)

    user_prompt = (
        f"{SYSTEM_PROMPT}\n\n"
        "Task: Write a short analyst note about this market time-series and model results.\n"
        "Use the JSON bundle below as your only source of truth.\n\n"
        f"JSON bundle:\n{json.dumps(bundle, indent=2)}\n"
    )

    note = callOllama(model=model, prompt=user_prompt, host=host)

    os.makedirs(os.path.dirname(out_md), exist_ok=True)
    with open(out_md, "w", encoding="utf-8") as f:
        f.write(note + "\n")

    print(f"Wrote local LLM note -> {out_md}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics-json", required=True)
    parser.add_argument("--preds-csv", required=True)
    parser.add_argument("--out-md", required=True)
    parser.add_argument("--recent-months", type=int, default=12)
    parser.add_argument("--model", default="llama3.1:8b")
    parser.add_argument("--host", default="http://localhost:11434")
    args = parser.parse_args()

    main(
        metrics_json=args.metrics_json,
        preds_csv=args.preds_csv,
        out_md=args.out_md,
        recent_months=args.recent_months,
        model=args.model,
        host=args.host,
    )
