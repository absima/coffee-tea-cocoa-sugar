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
python -m snakemake -s workflow/Snakefile --cores 1 --latency-wait 30
```


## Run the dashboard

The project includes a local Streamlit dashboard as one frontend option for exploring the
generated artifacts on your own machine.

After the pipeline has generated artifacts in `reports/`, launch the local dashboard:

```bash
streamlit run dashboard.py
```

This option is:
- local only
- useful for inspecting outputs during development
- not intended for GitHub Pages deployment


## Build the GitHub Pages showcase

The project also includes a separate static frontend for public sharing.
The GitHub Pages showcase is built into `docs/` from the files in `web/` plus the latest pipeline artifacts.
After running the pipeline, build the deployable static site with:

```bash
python src/build_showcase.py \
  --raw-csv data/raw/coffee.csv \
  --features-csv data/processed/coffee_features.csv \
  --metrics-json reports/metrics/coffee.json reports/metrics/cocoa.json reports/metrics/tea.json reports/metrics/sugar.json \
  --cross-asset-csv reports/cross_asset_metrics.csv \
  --cross-asset-note reports/cross_asset_note.md \
  --notes-dir reports/notes \
  --out-dir docs
```

To preview the generated static site locally:

```bash
python -m http.server 8000
```

Then open [http://localhost:8000/docs/](http://localhost:8000/docs/).


## Publish with GitHub Pages

Use the generated `docs/` directory as the Pages source:

1. Run the pipeline locally.
2. Build the showcase into `docs/`.
3. Commit and push the updated `docs/` folder.
4. In GitHub: `Settings -> Pages -> Build and deployment`.
5. Choose `Deploy from a branch`.
6. Select your main branch and the `/docs` folder.

The result is a fully static public showcase with no backend and no always-on Ollama server.
If your repository is named `commodities`, the demo URL will look like:

```text
https://mygithub.github.io/commodities/
```

In short:
- `dashboard.py` = local Streamlit frontend for development and inspection
- `web/` -> `docs/` = static frontend for GitHub Pages and public sharing
