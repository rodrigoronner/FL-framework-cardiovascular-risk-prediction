"""
data/convert_cleveland.py
=========================
Converts the UCI Cleveland Heart Disease raw data file (processed.va.data)
to a clean CSV that can be consumed by FedCVR's data pipeline.

Usage
-----
    1. Download the raw file from the UCI ML Repository:
       https://archive.ics.uci.edu/dataset/45/heart+disease
       (file: processed.cleveland.data or processed.va.data)

    2. Place the file in this directory (data/) and run:
          python data/convert_cleveland.py --input processed.cleveland.data

    3. The output file cleveland.csv will be written to data/.
"""

import argparse
import os
import pandas as pd

COLUMN_NAMES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak",
    "slope", "ca", "thal", "num",
]


def convert(input_path: str, output_path: str) -> None:
    if not os.path.exists(input_path):
        print(f"ERROR: File not found: {input_path}")
        return

    df = pd.read_csv(
        input_path,
        header=None,
        names=COLUMN_NAMES,
        na_values="?",
    )

    print(f"Loaded {len(df)} rows from '{input_path}'")
    print(df.head())
    df.info()

    df.to_csv(output_path, index=False)
    print(f"\nSaved as '{output_path}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert UCI raw data to CSV")
    parser.add_argument(
        "--input", type=str, default="processed.cleveland.data",
        help="Path to the raw UCI data file (default: processed.cleveland.data)",
    )
    parser.add_argument(
        "--output", type=str, default="cleveland.csv",
        help="Output CSV path (default: cleveland.csv)",
    )
    args = parser.parse_args()
    convert(args.input, args.output)
