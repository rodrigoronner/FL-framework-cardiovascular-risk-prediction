"""
data_utils.py - Data loading, harmonization, and preprocessing for the
four-client federation. See data/README.md for dataset details and the
top-level README's "Datasets" and "Preprocessing pipeline" sections for
the full rationale behind the schema and pipeline choices below.
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from flwr.common import Metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FILENAMES = [
    "zalizadeh_sani.csv",   # H1 - Z-Alizadeh Sani
    "ieee_chd.csv",         # H2 - IEEE Comprehensive Heart Disease
    "cleveland.csv",        # H3 - UCI Cleveland Heart Disease
    "kaggle_heart.csv",     # H4 - Cardiovascular Disease (sulianova)
]

# Column rename maps per dataset -> common 6-feature schema
# Harmonized features: age, sex, trestbps (systolic BP), diaBP (diastolic BP),
#                      chol (cholesterol), fbs (fasting blood glucose)
COLUMN_MAPPINGS: List[Dict[str, str]] = [
    # H1 - Z-Alizadeh Sani
    {
        "male": "sex",
        "age": "age",
        "totChol": "chol",
        "sysBP": "trestbps",
        "diaBP": "diaBP",
        "glucose": "fbs",
        "TenYearCHD": "target",
        "Typical Chest Pain": "h1_typical_chest_pain",
        "EF-TTE": "h1_ef_tte",
        "Region RWMA": "h1_region_rwma",
        "HTN": "h1_htn",
        "TG": "h1_tg",
        "K": "h1_k",
        "Tinversion": "h1_tinversion",
        "ESR": "h1_esr",
        "Neut": "h1_neut",
        "HDL": "h1_hdl",
    },
    # H2 - IEEE Comprehensive Heart Disease
    {
        "age": "age",
        "sex": "sex",
        "resting bp s": "trestbps",
        "cholesterol": "chol",
        "fasting blood sugar": "fbs",
        "target": "target",
    },
    # H3 - UCI Cleveland
    {
        "age": "age",
        "sex": "sex",
        "trestbps": "trestbps",
        "chol": "chol",
        "fbs": "fbs",
        "num": "target",
    },
    # H4 - Cardiovascular Disease dataset (sulianova). Cholesterol/glucose
    # are on a 3-level ordinal scale, not continuous mg/dL.
    {
        "Age": "age",
        "Gender": "sex",
        "SystolicBP": "trestbps",
        "DiastolicBP": "diaBP",
        "Cholesterol": "chol",
        "Glucose": "fbs",
        "HeartDisease": "target",
    },
]

# Harmonized feature set, same order for every client: base 6 shared
# features + 6 one-hot columns for H4's ordinal chol/fbs categories
# (zero for H1-H3) + H1's 10 extra native features (zero for H2-H4).
# See README "Harmonized Schema" for the full rationale.
H1_EXTRA_FEATURES = [
    "h1_typical_chest_pain", "h1_ef_tte", "h1_region_rwma", "h1_htn",
    "h1_tg", "h1_k", "h1_tinversion", "h1_esr", "h1_neut", "h1_hdl",
]

FINAL_FEATURES = [
    "age", "sex", "trestbps", "diaBP", "chol", "fbs",
    "chol_ord_1", "chol_ord_2", "chol_ord_3",
    "fbs_ord_1", "fbs_ord_2", "fbs_ord_3",
] + H1_EXTRA_FEATURES
ONE_HOT_ORDINAL_FEATURES = ["chol_ord_1", "chol_ord_2", "chol_ord_3",
                             "fbs_ord_1", "fbs_ord_2", "fbs_ord_3"]
# Index of the client (H4, 0-based) whose chol/fbs are native ordinal
# categories rather than continuous/binary values.
ORDINAL_CLIENT_INDEX = 3
# Index of the client (H1, 0-based) whose extra native features are H1_EXTRA_FEATURES.
EXTRA_FEATURE_CLIENT_INDEX = 0
TARGET_COLUMN = "target"


# ---------------------------------------------------------------------------
# Helper: IQR-based outlier capping
# ---------------------------------------------------------------------------

def _iqr_bounds(df_train: pd.DataFrame, features: List[str]) -> Dict[str, Tuple[float, float]]:
    """Compute [Q1 - 1.5*IQR, Q3 + 1.5*IQR] bounds per feature from the
    training fold only, to be applied to train/val/test without leakage."""
    bounds: Dict[str, Tuple[float, float]] = {}
    for col in features:
        if col not in df_train.columns:
            continue
        # Binary/near-constant columns have no meaningful IQR outliers;
        # capping them can collapse a skewed flag to a single value.
        if df_train[col].nunique(dropna=True) <= 2:
            continue
        q1 = df_train[col].quantile(0.25)
        q3 = df_train[col].quantile(0.75)
        iqr = q3 - q1
        bounds[col] = (q1 - 1.5 * iqr, q3 + 1.5 * iqr)
    return bounds


def _apply_iqr_cap(
    df: pd.DataFrame, bounds: Dict[str, Tuple[float, float]]
) -> pd.DataFrame:
    """Clip each feature in ``df`` to the bounds computed from the training fold."""
    df = df.copy()
    for col, (lower, upper) in bounds.items():
        df[col] = df[col].clip(lower=lower, upper=upper)
    return df


# ---------------------------------------------------------------------------
# Helper: SMOTE oversampling (training fold only)
# ---------------------------------------------------------------------------

def _apply_smote(
    X: np.ndarray,
    y: np.ndarray,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply SMOTE to balance classes. Falls back gracefully if imbalancedlearn
    is not installed or if minority class has fewer than 2 samples."""
    try:
        from imblearn.over_sampling import SMOTE

        minority_count = int(y.sum())
        if minority_count < 2:
            return X, y

        k = min(5, minority_count - 1)
        sm = SMOTE(k_neighbors=k, random_state=random_state)
        X_res, y_res = sm.fit_resample(X, y)
        return X_res, y_res
    except ImportError:
        warnings.warn(
            "imbalanced-learn not installed; skipping SMOTE. "
            "Install with: pip install imbalanced-learn",
            stacklevel=2,
        )
        return X, y


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_pre_smote_train_sizes(
    data_dir: str = ".",
    val_size: float = 0.10,
    test_size: float = 0.20,
    random_state: int = 42,
) -> List[int]:
    """Per-client training-fold row count before SMOTE, in FILENAMES order.
    Used by RDP accounting and the fuzzy fairness controller, which need
    the real (pre-oversampling) population size."""
    import os

    sizes: List[int] = []
    for i, filename in enumerate(FILENAMES):
        df = pd.read_csv(os.path.join(data_dir, filename))
        df.columns = df.columns.str.strip()
        df.rename(columns=COLUMN_MAPPINGS[i], inplace=True)
        if TARGET_COLUMN not in df.columns:
            raise ValueError(
                f"Target column '{TARGET_COLUMN}' not found in {filename} after mapping."
            )
        df[TARGET_COLUMN] = (
            pd.to_numeric(df[TARGET_COLUMN], errors="coerce") > 0
        ).astype(int)

        df_trainval, _ = train_test_split(
            df, test_size=test_size, random_state=random_state,
            stratify=df[TARGET_COLUMN],
        )
        val_fraction_of_trainval = val_size / (1.0 - test_size)
        df_train, _ = train_test_split(
            df_trainval, test_size=val_fraction_of_trainval,
            random_state=random_state, stratify=df_trainval[TARGET_COLUMN],
        )
        sizes.append(len(df_train))
    return sizes


def load_and_preprocess_data(
    data_dir: str = ".",
    val_size: float = 0.10,
    test_size: float = 0.20,
    random_state: int = 42,
) -> Tuple[
    Optional[List[Tuple[np.ndarray, np.ndarray]]],
    Optional[List[Tuple[np.ndarray, np.ndarray]]],
    Optional[List[Tuple[np.ndarray, np.ndarray]]],
    Optional[List[str]],
]:
    """Load and harmonize all four client datasets. Pipeline per client:
    target binarisation -> stratified 70/10/20 split -> IQR capping ->
    median imputation -> SMOTE (train only) -> MinMaxScaler. See README
    "Preprocessing pipeline" for the rationale.

    Returns (client_train_datasets, client_val_datasets,
    client_test_datasets, filenames); all None on load failure.
    """
    import os

    print(f"--- Loading and Preprocessing Datasets for {len(FILENAMES)} Clients ---")
    print(f"    Features: {FINAL_FEATURES}")
    print(f"    Split: 70% train / {int(val_size*100)}% val / {int(test_size*100)}% test")

    client_train_datasets: List[Tuple[np.ndarray, np.ndarray]] = []
    client_val_datasets: List[Tuple[np.ndarray, np.ndarray]] = []
    client_test_datasets: List[Tuple[np.ndarray, np.ndarray]] = []

    for i, filename in enumerate(FILENAMES):
        filepath = os.path.join(data_dir, filename)
        try:
            df = pd.read_csv(filepath)
            df.columns = df.columns.str.strip()
            df.rename(columns=COLUMN_MAPPINGS[i], inplace=True)

            if TARGET_COLUMN not in df.columns:
                raise ValueError(
                    f"Target column '{TARGET_COLUMN}' not found in {filename} "
                    f"after mapping. Found columns: {list(df.columns)}"
                )

            # Ensure every feature column exists (fill missing with NaN for imputation)
            for col in FINAL_FEATURES:
                if col not in df.columns:
                    df[col] = np.nan

            df = df[FINAL_FEATURES + [TARGET_COLUMN]].copy()

            # One-hot encode H4's ordinal chol/fbs categories; zero elsewhere.
            if i == ORDINAL_CLIENT_INDEX:
                chol_codes = pd.to_numeric(df["chol"], errors="coerce")
                fbs_codes = pd.to_numeric(df["fbs"], errors="coerce")
                for level in (1, 2, 3):
                    df[f"chol_ord_{level}"] = (chol_codes == level).astype(float)
                    df[f"fbs_ord_{level}"] = (fbs_codes == level).astype(float)
            else:
                for col in ONE_HOT_ORDINAL_FEATURES:
                    df[col] = 0.0

            # Harmonise 'sex' string labels -> binary (1=Male, 0=Female).
            if not pd.api.types.is_numeric_dtype(df["sex"]):
                df["sex"] = (
                    df["sex"].astype(str).str.strip().str.capitalize()
                    .map({"Male": 1, "Female": 0})
                    .fillna(pd.to_numeric(df["sex"], errors="coerce"))
                )

            # Coerce all columns to numeric
            for col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

            # Binarise target (any positive class -> 1)
            df[TARGET_COLUMN] = (df[TARGET_COLUMN] > 0).astype(int)
            y_full = df[TARGET_COLUMN].values

            # Split first; every later statistic (IQR bounds, medians,
            # scaler) is fit on the training fold only, no leakage.
            df_trainval, df_test = train_test_split(
                df,
                test_size=test_size,
                random_state=random_state,
                stratify=y_full,
            )
            val_fraction_of_trainval = val_size / (1.0 - test_size)
            df_train, df_val = train_test_split(
                df_trainval,
                test_size=val_fraction_of_trainval,
                random_state=random_state,
                stratify=df_trainval[TARGET_COLUMN],
            )

            iqr_bounds = _iqr_bounds(df_train, FINAL_FEATURES)
            df_train = _apply_iqr_cap(df_train, iqr_bounds)
            df_val = _apply_iqr_cap(df_val, iqr_bounds)
            df_test = _apply_iqr_cap(df_test, iqr_bounds)

            train_medians = df_train[FINAL_FEATURES].median()
            for split_df in (df_train, df_val, df_test):
                split_df[FINAL_FEATURES] = split_df[FINAL_FEATURES].fillna(train_medians)
                split_df[FINAL_FEATURES] = split_df[FINAL_FEATURES].fillna(0)

            X_train = df_train[FINAL_FEATURES].values.astype(np.float32)
            y_train = df_train[TARGET_COLUMN].values.astype(np.int64)
            X_val = df_val[FINAL_FEATURES].values.astype(np.float32)
            y_val = df_val[TARGET_COLUMN].values.astype(np.int64)
            X_test = df_test[FINAL_FEATURES].values.astype(np.float32)
            y_test = df_test[TARGET_COLUMN].values.astype(np.int64)
            y = y_full

            X_train_res, y_train_res = _apply_smote(X_train, y_train, random_state)

            scaler = MinMaxScaler()
            scaler.fit(X_train)
            X_train_scaled = scaler.transform(X_train_res)
            X_val_scaled = scaler.transform(X_val)
            X_test_scaled = scaler.transform(X_test)

            client_train_datasets.append((X_train_scaled, y_train_res))
            client_val_datasets.append((X_val_scaled, y_val))
            client_test_datasets.append((X_test_scaled, y_test))

            print(
                f"  Client H{i+1} ({filename}): {len(df)} samples  |  "
                f"train={len(y_train_res)} (post-SMOTE, raw={len(y_train)}), "
                f"val={len(y_val)}, test={len(y_test)}, "
                f"pos_rate_raw={y.mean():.1%}"
            )

        except FileNotFoundError:
            print(
                f"ERROR: '{filepath}' not found. "
                "See data/README.md for download instructions."
            )
            return None, None, None, None
        except Exception as exc:
            print(f"ERROR processing {filename}: {exc}")
            return None, None, None, None

    return client_train_datasets, client_val_datasets, client_test_datasets, FILENAMES


def aggregate_metrics_fn(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Weighted average of per-client evaluation metrics.

    Used as the ``evaluate_metrics_aggregation_fn`` for Flower strategies.
    Weights are proportional to each client's number of test examples.
    """
    if not metrics:
        return {}

    valid = [(n, m) for n, m in metrics if n > 0 and m]
    if not valid:
        return {}

    total = sum(n for n, _ in valid)
    if total == 0:
        return {}

    keys = ["accuracy", "precision", "recall", "f1_score"]
    return {
        k: sum(n * m[k] for n, m in valid if k in m) / total
        for k in keys
        if any(k in m for _, m in valid)
    }


def cluster_clients_by_distribution(
    client_train_data: List[Tuple[np.ndarray, np.ndarray]],
    n_clusters: int = 2,
    random_state: int = 42,
) -> Dict[str, int]:
    """Partition clients into ``n_clusters`` groups by similarity of their
    local feature distributions, for the FedCluster baseline.

    Each client is represented by the mean feature vector of its local
    (already-scaled) training data; KMeans groups clients whose local
    distributions are most alike. The FedCluster baseline's cited source
    does not fully specify its clustering criterion, so this is this
    project's concrete instantiation of "cluster by data-distribution
    difference".

    Returns
    -------
    dict mapping Flower client id string ("0", "1", ...) to cluster index.
    """
    from sklearn.cluster import KMeans

    fingerprints = np.stack([X.mean(axis=0) for X, _ in client_train_data])
    n_clusters = min(n_clusters, len(client_train_data))
    labels = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10).fit_predict(
        fingerprints
    )
    return {str(i): int(label) for i, label in enumerate(labels)}
