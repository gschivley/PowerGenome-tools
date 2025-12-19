"""Build the in-browser fuel prices CSV.

The web app loads fuel price scenarios from `web/data/fuel_prices.csv`.
This script downloads the latest data from PowerGenome-data and saves it locally.

Usage:
    # Download from default PowerGenome-data location
    python web/build_fuel_prices.py

    # Or provide a custom URL
    python web/build_fuel_prices.py \
        --url https://example.com/fuel_prices.csv \
        --out web/data/fuel_prices.csv

    # Or use a local CSV file
    python web/build_fuel_prices.py \
        --csv /path/to/fuel_prices.csv \
        --out web/data/fuel_prices.csv

Notes:
- Requires pandas.
- Validates required columns: data_year, fuel, scenario.
"""

from __future__ import annotations

import argparse
import sys
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

DEFAULT_URL = "https://raw.githubusercontent.com/gschivley/PowerGenome-data/main/data/fuel_prices.csv"


def download_csv(url: str) -> str:
    """Download CSV from URL and return the text content."""
    url = str(url).strip()
    if not url:
        raise ValueError("Empty URL")

    try:
        with urllib.request.urlopen(url) as resp:  # nosec - URL is user-provided
            content = resp.read().decode("utf-8")
        return content
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"Download failed ({exc.code}) for {url}") from exc
    except Exception as exc:
        raise RuntimeError(f"Download failed for {url}: {exc}") from exc


def validate_and_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and clean the fuel prices DataFrame."""
    # Check for required columns (case-insensitive)
    required = {"data_year", "fuel", "scenario"}
    lower_cols = {c.lower() for c in df.columns}

    if not required.issubset(lower_cols):
        missing = required - lower_cols
        raise ValueError(
            f"Missing required columns: {missing}. Found: {list(df.columns)}"
        )

    # Normalize column names
    col_map = {c: c.lower() for c in df.columns}
    df = df.rename(columns=col_map)

    # Ensure proper types
    df["data_year"] = pd.to_numeric(df["data_year"], errors="coerce").astype("Int64")
    df["fuel"] = df["fuel"].astype(str).str.strip()
    df["scenario"] = df["scenario"].astype(str).str.strip()

    # Drop rows with missing critical data
    df = df.dropna(subset=["data_year"])
    df = df[(df["fuel"] != "") & (df["scenario"] != "")]

    # Drop duplicate of the required columns
    df = df.drop_duplicates(subset=["data_year", "fuel", "scenario"])

    if df.empty:
        raise ValueError("No valid data rows after cleaning")

    return df


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download and save fuel_prices.csv for the web app"
    )
    source_group = parser.add_mutually_exclusive_group(required=False)
    source_group.add_argument(
        "--csv",
        type=Path,
        help="Path to local fuel_prices.csv file",
    )
    source_group.add_argument(
        "--url",
        type=str,
        help=f"URL to download fuel_prices.csv from (default: {DEFAULT_URL})",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("web/data/fuel_prices.csv"),
        help="Output CSV path (default: web/data/fuel_prices.csv)",
    )
    args = parser.parse_args()

    try:
        if args.csv:
            print(f"Reading from local file: {args.csv}")
            df = pd.read_csv(args.csv)
        else:
            url = args.url or DEFAULT_URL
            print(f"Downloading from: {url}")
            csv_text = download_csv(url)
            from io import StringIO

            df = pd.read_csv(StringIO(csv_text))

        # Validate and clean
        df = validate_and_clean(df)

        # Save to output
        args.out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.out, index=False)

        # Report stats
        n_years = df["data_year"].nunique()
        n_fuels = df["fuel"].nunique()
        n_scenarios = df["scenario"].nunique()
        print(f"Saved {len(df):,} rows to {args.out}")
        print(f"  Years: {n_years}, Fuels: {n_fuels}, Scenarios: {n_scenarios}")

        return 0

    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
