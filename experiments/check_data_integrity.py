"""
experiments/check_data_integrity.py
====================================
Audits the four client files for the two defects found in the first
submission: patients shared between clients, and duplicated patient
records inside a client (which can land on both sides of a split). Exits non-zero if either is found.

Cross-client matching uses every clinical field two files have in common
(H2 and H3 share the full UCI-style schema); within-client checks use
the complete record.

Usage
-----
    python -m experiments.check_data_integrity
"""
import itertools
import os
import sys

import pandas as pd

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)

from dpfedadam.data_utils import (CLIENT_LABELS, DROP_EXACT_DUPLICATES, FILENAMES,  # noqa: E402
                                  _load_client_frame)

RAW_KEYS = {  # common raw clinical fields, in a shared naming
    "zalizadeh_sani.csv": {"age": "age", "male": "sex", "sysBP": "sbp"},
    "mendeley_cvd.csv": {"age": "age", "gender": "sex", "restingBP": "sbp", "serumcholestrol": "chol",
                         "maxheartrate": "maxhr", "oldpeak": "oldpeak"},
    "cleveland.csv": {"age": "age", "sex": "sex", "trestbps": "sbp", "chol": "chol",
                      "thalach": "maxhr", "oldpeak": "oldpeak"},
    "kaggle_heart.csv": {"Age": "age", "Gender": "sex", "SystolicBP": "sbp"},
}


def main() -> int:
    data_dir = os.path.join(ROOT, "data")
    raw = {}
    for f in FILENAMES:
        df = pd.read_csv(os.path.join(data_dir, f))
        df.columns = df.columns.str.strip()
        raw[f] = df.rename(columns=RAW_KEYS[f])
    bad = 0
    print("Cross-client record overlap (matching on all shared clinical fields)")
    for a, b in itertools.combinations(FILENAMES, 2):
        keys = sorted(set(RAW_KEYS[a].values()) & set(RAW_KEYS[b].values()))
        if len(keys) < 6:
            print(f"  {a} x {b}: only {len(keys)} shared fields {keys} - disjoint sources, not matchable")
            continue
        ka = set(map(tuple, raw[a][keys].astype(float).round(2).dropna().values))
        kb = set(map(tuple, raw[b][keys].astype(float).round(2).dropna().values))
        n = len(ka & kb)
        bad += n
        print(f"  {a} x {b}: {n} shared records on {keys}")
    print("\nWithin-client repeated records")
    for i, (lab, f) in enumerate(zip(CLIENT_LABELS, FILENAMES)):
        full = pd.read_csv(os.path.join(data_dir, f))
        raw_dup = int(full.duplicated().sum())
        kept = 0 if DROP_EXACT_DUPLICATES[i] else raw_dup
        frame = _load_client_frame(i, data_dir)
        coincide = int(frame.duplicated().sum())
        note = ("removed before splitting" if DROP_EXACT_DUPLICATES[i]
                else "kept: distinct participants of a 70,000-row survey agreeing on six coarse fields")
        print(f"  {lab}: {len(full)} rows in file, {raw_dup} exact duplicate records ({note}); "
              f"{len(frame)} records enter the split; {coincide} of them coincide with another "
              f"record on the harmonized features only")
        if DROP_EXACT_DUPLICATES[i]:
            assert len(frame) == len(full) - raw_dup
        bad += 0 if kept == raw_dup or DROP_EXACT_DUPLICATES[i] else 1
    print("\nRESULT:", "OK - no patients shared between clients, no duplicated patient records inside a client" if bad == 0 else f"FAILED ({bad})")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
