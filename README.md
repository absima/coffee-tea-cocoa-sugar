![Reproducible coffee market analytics pipeline](images/coffee.png)

# Commodity Market Analytics (LLM-assisted)

A small, reproducible market time-series analytics pipeline using multiple commodity series. It fetches public data, stores each asset in SQLite, generates time-series features, trains a simple baseline model per asset, evaluates performance, and produces both per-asset notes and a cross-commodity comparison note using a **local LLM** (Ollama).

The LLM is used for **interpretation and reporting**, not for prediction.


## Outputs

Running the pipeline produces:
- `data/raw/<asset>.csv` (ingested series per commodity)
- `data/sqlite/<asset>.db` (SQLite database per commodity)
- `data/processed/<asset>_features.csv` (engineered features + target per commodity)
- `reports/metrics/<asset>.json` (evaluation metrics per commodity)
- `reports/preds/<asset>.csv` (predictions vs truth on the test set per commodity)
- `reports/notes/<asset>.md` (LLM-generated note per commodity)
- `reports/cross_asset_metrics.csv` (side-by-side comparison table)
- `reports/cross_asset_summary.json` (grounded comparison bundle for reporting)
- `reports/plots/*.png` (cross-asset visual diagnostics and comparison charts)
- `reports/cross_asset_note.md` (LLM-generated cross-commodity note)


## Requirements

- Python 3.9+
- Ollama installed and running locally


## Setup

```bash
rm -f .env
rm -rf .venv
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```


## Run the full pipeline

From the project root (with the virtual environment activated):

```bash
snakemake -s workflow/Snakefile --cores 1 --latency-wait 30
```
