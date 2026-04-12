from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from build_web_data import main as BuildWebData


def CopyTreeContents(source_dir: Path, target_dir: Path) -> None:
    """
    Copy all files from one directory into another directory.

    Args:
        source_dir: Path.
        Directory containing source files.
        target_dir: Path.
        Destination directory to populate.

    Returns:
        None.
        Copies directory contents into the target location.
    """
    if not source_dir.exists():
        raise FileNotFoundError(f"Expected source directory does not exist: {source_dir}")
    target_dir.mkdir(parents=True, exist_ok=True)
    for source_path in source_dir.iterdir():
        target_path = target_dir / source_path.name
        if source_path.is_dir():
            shutil.copytree(source_path, target_path, dirs_exist_ok=True)
        else:
            shutil.copy2(source_path, target_path)


def main(
    raw_csv: str,
    features_csv: str,
    metrics_json: list[str],
    cross_asset_csv: str,
    cross_asset_note: str,
    notes_dir: str,
    source_web_dir: str,
    plots_dir: str,
    out_dir: str,
) -> None:
    """
    Build a GitHub Pages-ready static showcase from pipeline artifacts.

    Args:
        raw_csv: str.
        Path to a representative raw asset CSV.
        features_csv: str.
        Path to a representative engineered feature CSV.
        metrics_json: list[str].
        Paths to per-asset metrics JSON files.
        cross_asset_csv: str.
        Path to the cross-asset comparison CSV.
        cross_asset_note: str.
        Path to the cross-asset markdown note.
        notes_dir: str.
        Directory containing per-asset markdown notes.
        source_web_dir: str.
        Source directory for the static web template files.
        plots_dir: str.
        Directory containing generated plot PNG files.
        out_dir: str.
        Output directory for the deployable static site.

    Returns:
        None.
        Writes a self-contained static showcase into the output directory.
    """
    out_path = Path(out_dir)
    if out_path.exists():
        shutil.rmtree(out_path)
    out_path.mkdir(parents=True, exist_ok=True)

    CopyTreeContents(Path(source_web_dir), out_path)
    CopyTreeContents(Path(plots_dir), out_path / "reports" / "plots")

    BuildWebData(
        raw_csv=raw_csv,
        features_csv=features_csv,
        metrics_json=metrics_json,
        cross_asset_csv=cross_asset_csv,
        cross_asset_note=cross_asset_note,
        notes_dir=notes_dir,
        out_json=str(out_path / "data" / "site_data.json"),
        plot_base_path="./reports/plots",
    )

    (out_path / ".nojekyll").write_text("", encoding="utf-8")
    print(f"Wrote deployable showcase -> {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-csv", required=True)
    parser.add_argument("--features-csv", required=True)
    parser.add_argument("--metrics-json", nargs="+", required=True)
    parser.add_argument("--cross-asset-csv", required=True)
    parser.add_argument("--cross-asset-note", required=True)
    parser.add_argument("--notes-dir", required=True)
    parser.add_argument("--source-web-dir", default="web")
    parser.add_argument("--plots-dir", default="reports/plots")
    parser.add_argument("--out-dir", default="docs")
    args = parser.parse_args()

    main(
        raw_csv=args.raw_csv,
        features_csv=args.features_csv,
        metrics_json=args.metrics_json,
        cross_asset_csv=args.cross_asset_csv,
        cross_asset_note=args.cross_asset_note,
        notes_dir=args.notes_dir,
        source_web_dir=args.source_web_dir,
        plots_dir=args.plots_dir,
        out_dir=args.out_dir,
    )
