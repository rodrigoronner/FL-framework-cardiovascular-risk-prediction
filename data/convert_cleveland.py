"""
data/convert_cleveland.py
=========================
Converts UCI Heart Disease raw data file(s) to a clean CSV (H3 client).
By default concatenates all four UCI sites (Cleveland + Hungarian +
Switzerland + Long Beach VA, 920 rows total).

Usage
-----
    1. Download the raw files from the UCI ML Repository:
       https://archive.ics.uci.edu/dataset/45/heart+disease
       (processed.cleveland.data, processed.hungarian.data,
        processed.switzerland.data, processed.va.data)

    2. Place them in one directory and run:
          python data/convert_cleveland.py --input_dir /path/to/heart+disease

    3. The combined output file cleveland.csv (920 rows) is written to data/.

       For a single-site conversion (legacy behavior), pass --input instead:
          python data/convert_cleveland.py --input processed.cleveland.data
"""

import argparse
import os
import pandas as pd

COLUMN_NAMES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak",
    "slope", "ca", "thal", "num",
]

DEFAULT_SITE_FILES = [
    "processed.cleveland.data",
    "processed.hungarian.data",
    "processed.switzerland.data",
    "processed.va.data",
]


def _read_site(path: str) -> pd.DataFrame:
    return pd.read_csv(path, header=None, names=COLUMN_NAMES, na_values="?")


def convert(input_path: str, output_path: str) -> None:
    if not os.path.exists(input_path):
        print(f"ERROR: File not found: {input_path}")
        return
    df = _read_site(input_path)
    print(f"Loaded {len(df)} rows from '{input_path}'")
    print(df.head())
    df.info()
    df.to_csv(output_path, index=False)
    print(f"\nSaved as '{output_path}'")


def convert_combined(input_dir: str, output_path: str, site_files=None) -> None:
    site_files = site_files or DEFAULT_SITE_FILES
    frames = []
    for name in site_files:
        path = os.path.join(input_dir, name)
        if not os.path.exists(path):
            print(f"ERROR: '{path}' not found. Aborting combined conversion.")
            return
        df = _read_site(path)
        print(f"  {name}: {len(df)} rows")
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    print(f"Combined total: {len(combined)} rows from {len(site_files)} sites")
    combined.to_csv(output_path, index=False)
    print(f"\nSaved as '{output_path}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert UCI raw data to CSV")
    parser.add_argument(
        "--input", type=str, default=None,
        help="Path to a single raw UCI data file (legacy, single-site mode).",
    )
    parser.add_argument(
        "--input_dir", type=str, default=None,
        help="Directory containing all four processed.*.data files "
             "(combined mode - matches the paper's reported 920-sample H3).",
    )
    parser.add_argument(
        "--output", type=str, default="cleveland.csv",
        help="Output CSV path (default: cleveland.csv)",
    )
    args = parser.parse_args()

    if args.input_dir:
        convert_combined(args.input_dir, args.output)
    elif args.input:
        convert(args.input, args.output)
    else:
        parser.error("Provide either --input_dir (combined, recommended) or --input.")
