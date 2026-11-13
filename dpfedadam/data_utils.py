"""
data_utils.py - Data loading, harmonization, and preprocessing for the
four-client federation. See data/README.md for dataset details.

Revision-1 protocol (what changed and why)
------------------------------------------
* H2 is the Mendeley "Cardiovascular Disease Dataset" (a single hospital,
  1,000 patients). The former H2 (IEEE DataPort) is a merge of the UCI
  Heart Disease sites and contained the H3 records, so the two clients
  shared patients.
* Exact duplicate records are removed from H3 before splitting, so one
  patient cannot sit in both the training and the test fold.
* No SMOTE. A synthetic minority point is an interpolation of several
  real records, so per-example clipping no longer bounds one patient's
  influence. Class imbalance is handled by the validation-calibrated
  decision threshold instead (dpfedadam.evaluation).
* Every transformation applied to the shared features is DATA
  INDEPENDENT: clipping and scaling use fixed public physiological
  ranges, and missing values are replaced by fixed clinical reference
  values. Nothing is fitted to the patients, so preprocessing neither
  leaks test information nor consumes privacy budget.
* The single data-dependent step is the ranking of H1's ten site-only
  features, computed on H1's TRAINING fold of the current partition.
  It is a local, non-private preprocessing decision of that site and is
  reported as such in the paper.
* A feature a site does not record is structurally absent and is encoded
  as a constant zero for that site (see FEATURE_AVAILABILITY).
"""

from __future__ import annotations

import os
import warnings
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FILENAMES = [
    "zalizadeh_sani.csv",   # H1 - Z-Alizadeh Sani (Tehran)
    "mendeley_cvd.csv",     # H2 - Mendeley Cardiovascular Disease Dataset (India)
    "cleveland.csv",        # H3 - UCI Heart Disease (four sites, US/EU)
    "kaggle_heart.csv",     # H4 - Cardiovascular Disease dataset (sulianova), subsampled
]
CLIENT_LABELS = ["H1", "H2", "H3", "H4"]

COLUMN_MAPPINGS: List[Dict[str, str]] = [
    {"male": "sex", "age": "age", "totChol": "chol", "sysBP": "trestbps",
     "glucose": "fbs", "target": "target"},
    {"age": "age", "gender": "sex", "restingBP": "trestbps",
     "serumcholestrol": "chol", "fastingbloodsugar": "fbs", "target": "target"},
    {"age": "age", "sex": "sex", "trestbps": "trestbps", "chol": "chol",
     "fbs": "fbs", "num": "target"},
    {"Age": "age", "Gender": "sex", "SystolicBP": "trestbps",
     "DiastolicBP": "diaBP", "Cholesterol": "chol_ordinal",
     "Glucose": "fbs_ordinal", "HeartDisease": "target"},
]

# Remove exact duplicate records before splitting? H3 carries two repeated
# rows. H4's 63 coincident rows are distinct participants of a 70,000-row
# survey who agree on six coarse fields, so they are kept.
DROP_EXACT_DUPLICATES = [True, True, True, False]

BASE_FEATURES = ["age", "sex", "trestbps", "diaBP", "chol", "fbs"]
ONE_HOT_ORDINAL_FEATURES = ["chol_ord_1", "chol_ord_2", "chol_ord_3",
                             "fbs_ord_1", "fbs_ord_2", "fbs_ord_3"]
N_H1_EXTRA = 10
H1_EXTRA_FEATURES = [f"h1_extra_{j+1}" for j in range(N_H1_EXTRA)]
FINAL_FEATURES = BASE_FEATURES + ONE_HOT_ORDINAL_FEATURES + H1_EXTRA_FEATURES
ORDINAL_CLIENT_INDEX = 3
EXTRA_FEATURE_CLIENT_INDEX = 0
TARGET_COLUMN = "target"
FBS_THRESHOLD_MG_DL = 120.0  # shared definition: fasting glucose > 120 mg/dL

# Which shared features each site actually records with the same meaning.
# "fbs" is the binary >120 mg/dL flag everywhere it is available; H4 only
# has ordinal cholesterol/glucose categories, carried by the one-hot block.
FEATURE_AVAILABILITY: Dict[str, List[bool]] = {
    #            H1     H2     H3     H4
    "age":      [True,  True,  True,  True],
    "sex":      [True,  True,  True,  True],
    "trestbps": [True,  True,  True,  True],
    "diaBP":    [False, False, False, True],
    "chol":     [True,  True,  True,  False],
    "fbs":      [True,  True,  True,  False],
}

# Fixed public physiological ranges (lower, upper) used for clipping and
# [0, 1] scaling, and fixed clinical reference values used for imputation.
PUBLIC_RANGES: Dict[str, Tuple[float, float]] = {
    "age": (18.0, 100.0), "sex": (0.0, 1.0), "trestbps": (70.0, 260.0),
    "diaBP": (40.0, 160.0), "chol": (80.0, 600.0), "fbs": (0.0, 1.0),
}
REFERENCE_VALUES: Dict[str, float] = {
    "age": 50.0, "sex": 0.0, "trestbps": 120.0, "diaBP": 80.0,
    "chol": 200.0, "fbs": 0.0,
}
# Public ranges for H1's non-binary native candidates (binary flags use [0, 1]).
H1_CANDIDATE_RANGES: Dict[str, Tuple[float, float]] = {
    "h1c_bmi": (10, 60), "h1c_bun": (2, 100), "h1c_cr": (0.2, 5),
    "h1c_ef_tte": (10, 80), "h1c_esr": (0, 150), "h1c_function_class": (0, 3),
    "h1c_hb": (5, 20), "h1c_hdl": (10, 120), "h1c_k": (2.5, 7),
    "h1c_ldl": (10, 300), "h1c_length": (120, 210), "h1c_lymph": (0, 100),
    "h1c_na": (120, 160), "h1c_neut": (0, 100), "h1c_plt": (10, 1000),
    "h1c_pr": (30, 200), "h1c_region_rwma": (0, 4), "h1c_tg": (20, 1200),
    "h1c_vhd": (0, 3), "h1c_wbc": (1000, 30000), "h1c_weight": (30, 200),
}


# ---------------------------------------------------------------------------
# Loading and harmonization (no statistics are fitted here)
# ---------------------------------------------------------------------------

def _scale(series: pd.Series, lo: float, hi: float) -> pd.Series:
    return (series.clip(lower=lo, upper=hi) - lo) / (hi - lo)


def _load_client_frame(i: int, data_dir: str) -> pd.DataFrame:
    """Read client ``i`` and return its harmonized, already-scaled frame:
    the shared features, H4's one-hot block, the target, and (H1 only)
    every scaled ``h1c_*`` candidate column."""
    raw = pd.read_csv(os.path.join(data_dir, FILENAMES[i]))
    raw.columns = raw.columns.str.strip()
    if DROP_EXACT_DUPLICATES[i]:
        raw = raw.drop_duplicates().reset_index(drop=True)
    raw = raw.rename(columns=COLUMN_MAPPINGS[i])
    if TARGET_COLUMN not in raw.columns:
        raise ValueError(f"Target column not found in {FILENAMES[i]} after mapping.")
    raw = raw.apply(pd.to_numeric, errors="coerce")

    out = pd.DataFrame(index=raw.index)
    for col in BASE_FEATURES:
        if not FEATURE_AVAILABILITY[col][i]:
            out[col] = 0.0  # structurally absent at this site
            continue
        values = raw[col]
        if col == "fbs" and values.max(skipna=True) > 1:
            # continuous mg/dL (H1) -> the shared binary >120 mg/dL flag
            values = (values > FBS_THRESHOLD_MG_DL).astype(float).where(values.notna())
        values = values.fillna(REFERENCE_VALUES[col])
        out[col] = _scale(values, *PUBLIC_RANGES[col])

    for level in (1, 2, 3):
        if i == ORDINAL_CLIENT_INDEX:
            out[f"chol_ord_{level}"] = (raw["chol_ordinal"] == level).astype(float)
            out[f"fbs_ord_{level}"] = (raw["fbs_ordinal"] == level).astype(float)
        else:
            out[f"chol_ord_{level}"] = 0.0
            out[f"fbs_ord_{level}"] = 0.0
    out = out[BASE_FEATURES + ONE_HOT_ORDINAL_FEATURES].copy()

    if i == EXTRA_FEATURE_CLIENT_INDEX:
        for col in [c for c in raw.columns if c.startswith("h1c_")]:
            lo, hi = H1_CANDIDATE_RANGES.get(col, (0.0, 1.0))
            if col not in H1_CANDIDATE_RANGES and raw[col].dropna().isin([0, 1]).mean() < 1:
                raise ValueError(f"No public range declared for non-binary column {col}")
            out[col] = _scale(raw[col].fillna(lo), float(lo), float(hi))

    out[TARGET_COLUMN] = (raw[TARGET_COLUMN] > 0).astype(int)
    return out


def _split(df: pd.DataFrame, val_size: float, test_size: float, random_state: int):
    """Stratified train/val/test split (70/10/20 by default)."""
    trainval, test = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=df[TARGET_COLUMN]
    )
    train, val = train_test_split(
        trainval, test_size=val_size / (1.0 - test_size),
        random_state=random_state, stratify=trainval[TARGET_COLUMN],
    )
    return train, val, test


def rank_h1_extra_features(df_train: pd.DataFrame, n_keep: int = N_H1_EXTRA) -> List[str]:
    """Rank H1's native candidates by CatBoost importance on the TRAINING
    fold only and return the top ``n_keep`` column names."""
    from catboost import CatBoostClassifier

    candidates = sorted(c for c in df_train.columns if c.startswith("h1c_"))
    model = CatBoostClassifier(
        iterations=300, depth=4, learning_rate=0.05, random_seed=0,
        verbose=0, thread_count=1, allow_writing_files=False,
    )
    model.fit(df_train[candidates], df_train[TARGET_COLUMN])
    order = np.argsort(-model.get_feature_importance(), kind="stable")
    return [candidates[j] for j in order[:n_keep]]


@lru_cache(maxsize=64)
def _prepare(data_dir: str, val_size: float, test_size: float, random_state: int):
    train_sets, val_sets, test_sets, meta = [], [], [], []
    for i in range(len(FILENAMES)):
        df = _load_client_frame(i, data_dir)
        train, val, test = _split(df, val_size, test_size, random_state)
        selected: List[str] = []
        if i == EXTRA_FEATURE_CLIENT_INDEX:
            selected = rank_h1_extra_features(train)
        folds = []
        for fold in (train, val, test):
            fold = fold.copy()
            for slot, name in enumerate(H1_EXTRA_FEATURES):
                fold[name] = fold[selected[slot]] if slot < len(selected) else 0.0
            folds.append((
                fold[FINAL_FEATURES].values.astype(np.float32),
                fold[TARGET_COLUMN].values.astype(np.int64),
            ))
        train_sets.append(folds[0]); val_sets.append(folds[1]); test_sets.append(folds[2])
        meta.append({
            "client": CLIENT_LABELS[i], "file": FILENAMES[i], "n_total": len(df),
            "n_train": len(train), "n_val": len(val), "n_test": len(test),
            "positive_rate": float(df[TARGET_COLUMN].mean()),
            "h1_selected_features": selected,
        })
    return train_sets, val_sets, test_sets, meta


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_and_preprocess_data(
    data_dir: str = ".",
    val_size: float = 0.10,
    test_size: float = 0.20,
    random_state: int = 42,
    verbose: bool = True,
) -> Tuple[
    Optional[List[Tuple[np.ndarray, np.ndarray]]],
    Optional[List[Tuple[np.ndarray, np.ndarray]]],
    Optional[List[Tuple[np.ndarray, np.ndarray]]],
    Optional[List[str]],
]:
    """Load and harmonize the four client datasets. ``random_state`` is the
    PARTITION seed. Returns (train, val, test, filenames); the training
    folds contain real records only (no oversampling)."""
    try:
        train, val, test, meta = _prepare(os.path.abspath(data_dir), val_size, test_size, random_state)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}. See data/README.md for download instructions.")
        return None, None, None, None
    if verbose:
        for m in meta:
            print(f"  Client {m['client']} ({m['file']}): {m['n_total']} records | "
                  f"train={m['n_train']}, val={m['n_val']}, test={m['n_test']}, "
                  f"pos_rate={m['positive_rate']:.1%}")
    return train, val, test, FILENAMES


def get_partition_metadata(
    data_dir: str = ".", val_size: float = 0.10, test_size: float = 0.20,
    random_state: int = 42,
) -> List[Dict]:
    """Fold sizes, prevalence and H1's selected features for one partition."""
    return _prepare(os.path.abspath(data_dir), val_size, test_size, random_state)[3]


def get_train_sizes(
    data_dir: str = ".", val_size: float = 0.10, test_size: float = 0.20,
    random_state: int = 42,
) -> List[int]:
    """Per-client training-fold size, i.e. the number of real patients that
    DP-SGD samples from. This is the n used by the privacy accountant."""
    return [m["n_train"] for m in get_partition_metadata(data_dir, val_size, test_size, random_state)]
