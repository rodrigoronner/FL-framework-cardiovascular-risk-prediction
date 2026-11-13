"""
data/convert_mendeley_cvd.py
=============================
Converts the "Cardiovascular Disease Dataset" (Doppala & Bhattacharyya,
Mendeley Data, V1, 2021, doi:10.17632/dzz48mvjht.1, CC BY 4.0; 1,000
patients from a multispecialty hospital in India) into
data/mendeley_cvd.csv (H2).

This cohort replaced the IEEE DataPort "Heart Disease Dataset
(Comprehensive)" as H2 in the first revision, because the IEEE file is a
merge of the UCI Heart Disease sites and therefore contains the H3
records (see data/README.md, "History note (revision 1)").

Transformations
---------------
1. ``patientid`` is dropped (an identifier, never a feature).
2. ``serumcholestrol == 0`` is a missing-value code and becomes NaN.

Usage
-----
    python data/convert_mendeley_cvd.py \
        --input Cardiovascular_Disease_Dataset.csv --output data/mendeley_cvd.csv
"""
import argparse

import numpy as np
import pandas as pd


def convert(input_path: str, output_path: str) -> None:
    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} rows from '{input_path}'")
    df = df.drop(columns=["patientid"])
    n_zero = int((df["serumcholestrol"] == 0).sum())
    df["serumcholestrol"] = df["serumcholestrol"].replace(0, np.nan)
    print(f"Recoded {n_zero} zero cholesterol values as missing")
    print(f"Exact duplicate rows: {int(df.duplicated().sum())}")
    print(f"Final sample: {len(df)} rows, positive rate={df['target'].mean():.1%}")
    df.to_csv(output_path, index=False)
    print(f"Saved as '{output_path}'")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Convert the Mendeley CVD dataset to the H2 client CSV")
    p.add_argument("--input", type=str, default="Cardiovascular_Disease_Dataset.csv")
    p.add_argument("--output", type=str, default="mendeley_cvd.csv")
    a = p.parse_args()
    convert(a.input, a.output)
