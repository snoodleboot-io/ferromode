#!/usr/bin/env python3
"""
Download the Kaggle S&P 500 dataset (camnugent/sandp500).

Dataset contains:
  - individual_stocks_5yr/   ~500 CSVs, one per ticker
    Columns: date, open, high, low, close, volume, Name
  - all_stocks_5yr.csv       single merged file
  - merge.sh                 merge script from the original dataset

Requirements:
    pip install kaggle

Credentials:
    Set KAGGLE_USERNAME and KAGGLE_KEY environment variables,
    OR place ~/.kaggle/kaggle.json with {"username": "...", "key": "..."}.

Usage:
    python scripts/download_sp500.py [--dest data/sp500] [--unzip]
"""

import argparse
import os
import subprocess
import sys
import zipfile
from pathlib import Path


DATASET = "camnugent/sandp500"
DEFAULT_DEST = Path("data/sp500")


def check_kaggle_credentials() -> bool:
    username = os.environ.get("KAGGLE_USERNAME")
    key = os.environ.get("KAGGLE_KEY")
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if (username and key) or kaggle_json.exists():
        return True
    print(
        "ERROR: Kaggle credentials not found.\n"
        "  Option 1 — environment variables:\n"
        "    export KAGGLE_USERNAME=your_username\n"
        "    export KAGGLE_KEY=your_api_key\n"
        "  Option 2 — credential file:\n"
        "    mkdir -p ~/.kaggle && chmod 700 ~/.kaggle\n"
        "    echo '{\"username\":\"...\",\"key\":\"...\"}' > ~/.kaggle/kaggle.json\n"
        "    chmod 600 ~/.kaggle/kaggle.json\n"
        "\n"
        "Get your API key at: https://www.kaggle.com/settings  →  API  →  Create New Token"
    )
    return False


def download(dest: Path, unzip: bool) -> None:
    dest.mkdir(parents=True, exist_ok=True)

    print(f"Downloading dataset '{DATASET}' to {dest} …")
    result = subprocess.run(
        [
            sys.executable, "-m", "kaggle",
            "datasets", "download",
            "--dataset", DATASET,
            "--path", str(dest),
        ],
        check=False,
    )
    if result.returncode != 0:
        print("ERROR: kaggle download failed. See output above.")
        sys.exit(1)

    zip_candidates = list(dest.glob("*.zip"))
    if not zip_candidates:
        print("No ZIP file found after download — dataset may already be unzipped.")
        return

    zip_path = zip_candidates[0]
    print(f"Downloaded: {zip_path} ({zip_path.stat().st_size / 1_048_576:.1f} MB)")

    if unzip:
        print(f"Extracting to {dest} …")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(dest)
        zip_path.unlink()
        print("Extraction complete. ZIP removed.")

        individual = dest / "individual_stocks_5yr"
        if individual.is_dir():
            tickers = list(individual.glob("*_data.csv"))
            print(f"Found {len(tickers)} individual stock CSVs.")
        else:
            print("NOTE: 'individual_stocks_5yr/' folder not found after extraction.")
            print("      Contents:", [p.name for p in dest.iterdir()])
    else:
        print(f"ZIP retained at {zip_path}.")
        print("Re-run with --unzip to extract automatically.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--dest", type=Path, default=DEFAULT_DEST,
        help=f"Directory to download data into (default: {DEFAULT_DEST})",
    )
    parser.add_argument(
        "--unzip", action="store_true", default=True,
        help="Extract the downloaded ZIP immediately (default: True)",
    )
    parser.add_argument(
        "--no-unzip", dest="unzip", action="store_false",
        help="Keep the ZIP without extracting",
    )
    args = parser.parse_args()

    if not check_kaggle_credentials():
        sys.exit(1)

    try:
        import kaggle  # noqa: F401
    except ImportError:
        print("ERROR: kaggle package not installed. Run:  pip install kaggle")
        sys.exit(1)

    download(args.dest, args.unzip)
    print(
        "\nDone. Next step:\n"
        f"  python scripts/generate_imfs.py --data-dir {args.dest} --ticker AAPL\n"
    )


if __name__ == "__main__":
    main()
