"""
data/convert_z_alizadeh_sani.py
================================
Converts the Z-Alizadeh Sani dataset (Shaheed Rajaei Cardiovascular
Center, Tehran, 303 patients; UCI ML Repository, doi:10.24432/C5Q31T)
into data/zalizadeh_sani.csv (H1).

The output keeps the five shared risk factors plus EVERY remaining native
column, numerically encoded and prefixed ``h1c_`` ("H1 candidate"). No
feature selection happens here: the ten H1-only features that enter the
model are ranked inside ``dpfedadam.data_utils`` on the TRAINING FOLD of
each partition only, so no validation or test row can influence the
choice. (Before revision 1 this script hard-coded ten columns that had
been ranked on the full dataset.)

Total cholesterol is derived with the Friedewald identity
(LDL + HDL + TG/5) because the raw file has no direct column.

Usage
-----
    python data/convert_z_alizadeh_sani.py \
        --input "Z-Alizadeh sani dataset.xlsx" --output data/zalizadeh_sani.csv
"""
import argparse

import pandas as pd

YES_NO = {"Y": 1, "N": 0}
VHD_LEVELS = {"N": 0, "mild": 1, "Moderate": 2, "Severe": 3}
BASE_SOURCE = {"Age", "Sex", "BP", "FBS", "Cath"}


def convert(input_path: str, output_path: str) -> None:
    df = pd.read_excel(input_path) if input_path.endswith(".xlsx") else pd.read_csv(input_path)
    df.columns = df.columns.str.strip()
    print(f"Loaded {len(df)} rows, {df.shape[1]} columns from '{input_path}'")

    before = len(df)
    df = df[(df["LDL"] > 0) & (df["HDL"] > 0) & (df["TG"] > 0)].copy()
    print(f"Dropped {before - len(df)} rows with non-physiological LDL/HDL/TG")

    out = pd.DataFrame(index=df.index)
    out["age"] = df["Age"]
    out["male"] = (df["Sex"].str.strip() == "Male").astype(int)
    out["sysBP"] = df["BP"]
    out["totChol"] = df["LDL"] + df["HDL"] + df["TG"] / 5.0
    out["glucose"] = df["FBS"]
    out["target"] = (df["Cath"].str.strip() == "Cad").astype(int)

    for col in sorted(c for c in df.columns if c not in BASE_SOURCE):
        s = df[col]
        name = "h1c_" + col.lower().replace(" ", "_").replace("-", "_")
        if col == "BBB":
            out["h1c_lbbb"] = (s.str.strip() == "LBBB").astype(int)
            out["h1c_rbbb"] = (s.str.strip() == "RBBB").astype(int)
        elif col == "VHD":
            out[name] = s.str.strip().map(VHD_LEVELS).astype(int)
        elif s.dtype == object or str(s.dtype).startswith("str"):
            mapped = s.str.strip().map(YES_NO)
            if mapped.isna().any():
                raise ValueError(f"Unexpected labels in '{col}': {sorted(s.unique())}")
            out[name] = mapped.astype(int)
        else:
            out[name] = s
    constant = [c for c in out.columns if out[c].nunique() <= 1]
    out = out.drop(columns=constant)
    print(f"Dropped constant columns: {constant}")

    n_cand = sum(c.startswith("h1c_") for c in out.columns)
    print(f"Final sample: {len(out)} rows, {n_cand} H1 candidate features, "
          f"positive rate={out['target'].mean():.1%}")
    out.to_csv(output_path, index=False)
    print(f"Saved as '{output_path}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Z-Alizadeh Sani to the H1 client CSV")
    parser.add_argument("--input", type=str, default="Z-Alizadeh sani dataset.xlsx")
    parser.add_argument("--output", type=str, default="zalizadeh_sani.csv")
    args = parser.parse_args()
    convert(args.input, args.output)
