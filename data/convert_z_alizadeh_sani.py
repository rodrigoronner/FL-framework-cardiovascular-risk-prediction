"""
data/convert_z_alizadeh_sani.py
================================
Converts the Z-Alizadeh Sani dataset (Shaheed Rajaei Cardiovascular
Center, Tehran, 303 patients) into data/zalizadeh_sani.csv (H1). Total
cholesterol is derived via the Friedewald formula (LDL + HDL + TG/5)
since the raw file has no direct cholesterol column; rows with
non-physiological (<=0) LDL/HDL/TG are dropped. See data/README.md for
the full column mapping and H1's replacement history.

Usage
-----
    python data/convert_z_alizadeh_sani.py \
        --input "Z-Alizadeh sani dataset (2).csv" --output data/zalizadeh_sani.csv
"""
import argparse
import os

import pandas as pd


def convert(input_path: str, output_path: str) -> None:
    if not os.path.exists(input_path):
        print(f"ERROR: File not found: {input_path}")
        return

    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} rows from '{input_path}'")

    # Friedewald formula for total cholesterol; drop physiologically
    # invalid lipid-panel rows rather than impute them.
    before = len(df)
    df = df[(df["LDL"] > 0) & (df["HDL"] > 0) & (df["TG"] > 0)].copy()
    print(f"Dropped {before - len(df)} rows with non-physiological LDL/HDL/TG")
    df["totChol"] = df["LDL"] + df["HDL"] + df["TG"] / 5.0

    df["male"] = (df["Sex"] == "Male").astype(int)
    df["sysBP"] = df["BP"]
    df["glucose"] = df["FBS"]
    df["TenYearCHD"] = (df["Cath"] == "Cad").astype(int)

    # Top-10 features by CatBoost importance - H1-only additions to the
    # global schema (dpfedadam.data_utils.H1_EXTRA_FEATURES), zero elsewhere.
    extra_cols = [
        "Typical Chest Pain", "EF-TTE", "Region RWMA", "HTN", "TG", "K",
        "Tinversion", "ESR", "Neut", "HDL",
    ]

    out = df[["Age", "male", "sysBP", "totChol", "glucose", "TenYearCHD"] + extra_cols].rename(
        columns={"Age": "age"}
    )
    out["diaBP"] = pd.NA  # not present in this dataset; median-imputed downstream

    print(f"\nFinal sample: {len(out)} rows, "
          f"positive rate={out['TenYearCHD'].mean():.1%}")
    print(out.head())

    out.to_csv(output_path, index=False)
    print(f"\nSaved as '{output_path}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert Z-Alizadeh Sani dataset to the H1 (zalizadeh_sani.csv) client CSV"
    )
    parser.add_argument("--input", type=str, default="Z-Alizadeh sani dataset (2).csv")
    parser.add_argument("--output", type=str, default="zalizadeh_sani.csv")
    args = parser.parse_args()
    convert(args.input, args.output)
