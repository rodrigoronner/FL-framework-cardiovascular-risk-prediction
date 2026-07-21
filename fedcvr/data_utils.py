"""
data_utils.py - Data loading, harmonization, and preprocessing.

Five publicly available cardiovascular datasets are harmonized to a common
6-feature schema and split into per-client training/validation/test partitions
(70/10/20) that simulate the non-IID, institutionally siloed nature of real
healthcare data.

Preprocessing pipeline (applied per client, in order)
------------------------------------------------------
0. Binarisation of the target column (any positive class -> 1) and
   stratified 70/10/20 train/validation/test split.
1. IQR-based outlier capping: bounds (Q1 - 1.5*IQR, Q3 + 1.5*IQR) are
   computed from the training fold only and applied to all folds.
2. Median imputation for missing values: medians are computed from the
   training fold only and applied to all folds, preventing leakage of
   held-out statistics into training-time preprocessing.
3. SMOTE oversampling applied to the training fold only to address
   class imbalance.
4. StandardScaler normalization (fit on training fold, applied to all folds).

Datasets
--------
Client H1  -  Framingham Heart Study           (framingham.csv)
Client H2  -  IEEE Comprehensive Heart Disease  (ieee_chd.csv)
Client H3  -  UCI Cleveland Heart Disease       (cleveland.csv)
Client H4  -  FIC Pakistan                      (fic_pakistan.csv)
Client H5  -  Kaggle Heart Prediction           (kaggle_heart.csv)

Download instructions: see data/README.md
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from flwr.common import Metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FILENAMES = [
    "framingham.csv",       # H1 - Framingham Heart Study
    "ieee_chd.csv",         # H2 - IEEE Comprehensive Heart Disease
    "cleveland.csv",        # H3 - UCI Cleveland Heart Disease
    "fic_pakistan.csv",     # H4 - FIC Pakistan
    "kaggle_heart.csv",     # H5 - Kaggle Heart Prediction
]

# Column rename maps per dataset -> common 6-feature schema
# Harmonized features: age, sex, trestbps (systolic BP), diaBP (diastolic BP),
#                      chol (cholesterol), fbs (fasting blood glucose)
COLUMN_MAPPINGS: List[Dict[str, str]] = [
    # H1 - Framingham
    {
        "male": "sex",
        "age": "age",
        "totChol": "chol",
        "sysBP": "trestbps",
        "diaBP": "diaBP",
        "glucose": "fbs",
        "TenYearCHD": "target",
    },
    # H2 - IEEE Comprehensive Heart Disease
    {
        "Age": "age",
        "Sex": "sex",
        "SBP": "trestbps",
        "DBP": "diaBP",
        "Cholesterol": "chol",
        "FastingBS": "fbs",
        "HeartDisease": "target",
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
    # H4 - FIC Pakistan
    {
        "Age": "age",
        "Gender": "sex",
        "Systolic BP": "trestbps",
        "Diastolic BP": "diaBP",
        "Cholestrol": "chol",
        "FBS": "fbs",
        "DEATH_EVENT": "target",
    },
    # H5 - Kaggle Heart Prediction
    {
        "Age": "age",
        "Sex": "sex",
        "RestingBP": "trestbps",
        "Cholesterol": "chol",
        "FastingBS": "fbs",
        "HeartDisease": "target",
    },
]

# Harmonized 6-feature set (same order for every client)
FINAL_FEATURES = ["age", "sex", "trestbps", "diaBP", "chol", "fbs"]
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

def load_and_preprocess_data(
    data_dir: str = ".",
    val_size: float = 0.10,
    test_size: float = 0.20,
    random_state: int = 42,
) -> Tuple[
    Optional[List[Tuple[np.ndarray, np.ndarray]]],
    Optional[List[Tuple[np.ndarray, np.ndarray]]],
    Optional[List[str]],
]:
    """Load and harmonize all five cardiovascular datasets.

    The preprocessing pipeline for each client is:
        1. Binarisation of the target column.
        2. Stratified 70/10/20 train/validation/test split.
        3. IQR-based outlier capping (bounds from the training fold only).
        4. Median imputation for remaining missing values (medians from the
           training fold only) - both 3. and 4. are fit on the training
           fold and applied unchanged to val/test, preventing leakage.
        5. SMOTE oversampling on the training fold only.
        6. StandardScaler normalization (fit on train, applied to val and test).

    Note: the Flower simulation uses only training and test folds. The
    validation fold is stored in the returned ``client_val_datasets`` list
    and can be used for hyper-parameter tuning outside the FL loop.

    Parameters
    ----------
    data_dir:
        Directory that contains the five CSV files.
    val_size:
        Fraction of each client's data reserved for validation (default 0.10).
    test_size:
        Fraction of each client's data reserved for testing (default 0.20).
    random_state:
        Random seed for reproducibility.

    Returns
    -------
    client_train_datasets : list of (X_train, y_train) arrays (post-SMOTE).
    client_test_datasets  : list of (X_test,  y_test)  arrays.
    filenames             : list of dataset file names (same order).

    All three return values are ``None`` if loading fails for any dataset.
    """
    import os

    print("--- Loading and Preprocessing Datasets for 5 Clients ---")
    print(f"    Features: {FINAL_FEATURES}")
    print(f"    Split: 70% train / {int(val_size*100)}% val / {int(test_size*100)}% test")

    client_train_datasets: List[Tuple[np.ndarray, np.ndarray]] = []
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

            # Harmonise 'sex' string labels -> binary (1=Male, 0=Female)
            if df["sex"].dtype == object:
                sex_map = {"Male": 1, "Female": 0, "male": 1, "female": 0,
                           "M": 1, "F": 0, "1": 1, "0": 0}
                df["sex"] = (
                    df["sex"].astype(str).str.strip().str.capitalize()
                    .map({"Male": 1, "Female": 0})
                    .fillna(pd.to_numeric(df["sex"], errors="coerce"))
                )

            # Coerce all columns to numeric
            for col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

            # Binarise target (any positive class -> 1). Deterministic
            # thresholding, so doing this before the split introduces no
            # train/test leakage.
            df[TARGET_COLUMN] = (df[TARGET_COLUMN] > 0).astype(int)
            y_full = df[TARGET_COLUMN].values

            # Step 1: Stratified 70/10/20 split, performed on the raw
            # (pre-capping, pre-imputation) frame so that every subsequent
            # preprocessing statistic (IQR bounds, medians, scaler) is fit
            # on the training fold only and merely applied to val/test -
            # matching the paper's stated no-leakage preprocessing protocol.
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

            # Step 2: IQR-based outlier capping - bounds computed from the
            # training fold only, applied identically to train/val/test.
            iqr_bounds = _iqr_bounds(df_train, FINAL_FEATURES)
            df_train = _apply_iqr_cap(df_train, iqr_bounds)
            df_val = _apply_iqr_cap(df_val, iqr_bounds)
            df_test = _apply_iqr_cap(df_test, iqr_bounds)

            # Step 3: Median imputation - medians computed from the
            # (capped) training fold only, applied identically to
            # train/val/test, preventing any cross-fold leakage.
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

            # Step 4: SMOTE on training fold only
            X_train_res, y_train_res = _apply_smote(X_train, y_train, random_state)

            # Step 5: StandardScaler (fit on pre-SMOTE train to avoid leakage)
            scaler = StandardScaler()
            scaler.fit(X_train)
            X_train_scaled = scaler.transform(X_train_res)
            X_val_scaled = scaler.transform(X_val)
            X_test_scaled = scaler.transform(X_test)

            client_train_datasets.append((X_train_scaled, y_train_res))
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
            return None, None, None
        except Exception as exc:
            print(f"ERROR processing {filename}: {exc}")
            return None, None, None

    return client_train_datasets, client_test_datasets, FILENAMES


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
