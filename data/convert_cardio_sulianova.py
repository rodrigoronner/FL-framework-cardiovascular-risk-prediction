"""
data/convert_cardio_sulianova.py
=================================
Converts the "Cardiovascular Disease dataset" (sulianova, Kaggle,
semicolon-delimited cardio_train.csv, 70,000 rows) into
data/kaggle_heart.csv (H4). See data/README.md for the full column
mapping and the one disclosed limitation (cholesterol/glucose stay on
their native 3-level ordinal scale, not continuous mg/dL).

Source
------
    https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset

Transformations
---------------
1. Age converted from days to years.
2. Gender remapped (1=women, 2=men) to this project's (1=Male, 0=Female).
3. Rows with implausible or diastolic>=systolic blood pressure dropped.
4. Stratified subsampling to ``--n_samples`` (default 1,025) so this
   client doesn't dominate federated aggregation.

Usage
-----
    python data/convert_cardio_sulianova.py \
        --input cardio_train.csv --output data/kaggle_heart.csv --n_samples 1025
"""

import argparse
import os

import pandas as pd


def convert(input_path: str, output_path: str, n_samples: int, random_state: int = 42) -> None:
    if not os.path.exists(input_path):
        print(f"ERROR: File not found: {input_path}")
        return

    df = pd.read_csv(input_path, sep=";")
    print(f"Loaded {len(df)} rows from '{input_path}'")

    # Step 1: age days -> years
    df["Age"] = (df["age"] / 365.25).round(1)

    # Step 2: gender remap (source: 1=women, 2=men -> project: 1=Male, 0=Female)
    df["Gender"] = (df["gender"] == 2).astype(int)

    # Step 3: sanity-filter implausible blood pressure entries
    before = len(df)
    df = df[
        df["ap_hi"].between(70, 260)
        & df["ap_lo"].between(40, 200)
        & (df["ap_hi"] > df["ap_lo"])
    ].copy()
    print(f"Dropped {before - len(df)} rows with implausible ap_hi/ap_lo values "
          f"({len(df)} remaining)")

    df["SystolicBP"] = df["ap_hi"]
    df["DiastolicBP"] = df["ap_lo"]
    df["Cholesterol"] = df["cholesterol"]  # ordinal 1/2/3, kept as-is
    df["Glucose"] = df["gluc"]             # ordinal 1/2/3, kept as-is
    df["HeartDisease"] = df["cardio"]

    out = df[
        ["Age", "Gender", "SystolicBP", "DiastolicBP", "Cholesterol", "Glucose", "HeartDisease"]
    ]

    # Step 5: stratified subsample to n_samples
    if n_samples < len(out):
        frac = n_samples / len(out)
        parts = [
            group.sample(frac=frac, random_state=random_state)
            for _, group in out.groupby("HeartDisease")
        ]
        out = pd.concat(parts, ignore_index=True)
        # Trim to exactly n_samples after rounding within each stratum
        if len(out) > n_samples:
            out = out.sample(n=n_samples, random_state=random_state)

    out = out.reset_index(drop=True)
    print(f"Final sample: {len(out)} rows, "
          f"positive rate={out['HeartDisease'].mean():.1%}")
    print(out.head())

    out.to_csv(output_path, index=False)
    print(f"\nSaved as '{output_path}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert sulianova/cardiovascular-disease-dataset to the H5 client CSV"
    )
    parser.add_argument("--input", type=str, default="cardio_train.csv")
    parser.add_argument("--output", type=str, default="kaggle_heart.csv")
    parser.add_argument("--n_samples", type=int, default=1025)
    parser.add_argument("--random_state", type=int, default=42)
    args = parser.parse_args()
    convert(args.input, args.output, args.n_samples, args.random_state)
