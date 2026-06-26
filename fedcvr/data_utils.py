"""
<<<<<<< HEAD
data_utils.py – Data loading, harmonisation, and preprocessing.

Five publicly available cardiovascular datasets are harmonised to a common
10-feature schema and split into per-client training/test pairs that simulate
the non-IID, institutionally siloed nature of real healthcare data.

Datasets
--------
Client 0  –  Framingham Heart Study     (framingham.csv)
Client 1  –  UCI Cleveland Heart Disease (cleveland.csv)
Client 2  –  FIC Pakistan               (fic_pakistan.csv)
Client 3  –  Heart Disease Prediction   (heart_disease_prediction.csv)
Client 4  –  Hungarian                  (Hungarian-98-10.csv)
=======
data_utils.py - Data loading, harmonisation, and preprocessing.

Five publicly available cardiovascular datasets are harmonised to a common
6-feature schema and split into per-client training/validation/test partitions
(70/10/20) that simulate the non-IID, institutionally siloed nature of real
healthcare data.

Preprocessing pipeline (applied per client, in order)
------------------------------------------------------
1. IQR-based outlier capping: values below Q1 - 1.5*IQR or above
   Q3 + 1.5*IQR are clipped to the fence values.
2. Median imputation for missing values.
3. Binarisation of the target column (any positive class -> 1).
4. SMOTE oversampling applied to the training fold only to address
   class imbalance.
5. StandardScaler normalisation (fit on training fold, applied to all folds).

Datasets
--------
Client H1  -  Framingham Heart Study           (framingham.csv)
Client H2  -  IEEE Comprehensive Heart Disease  (ieee_chd.csv)
Client H3  -  UCI Cleveland Heart Disease       (cleveland.csv)
Client H4  -  FIC Pakistan                      (fic_pakistan.csv)
Client H5  -  Kaggle Heart Prediction           (kaggle_heart.csv)
>>>>>>> 3d539aa >>>>>>>

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
<<<<<<< HEAD
    "framingham.csv",
    "cleveland.csv",
    "fic_pakistan.csv",
    "heart_disease_prediction.csv",
    "Hungarian-98-10.csv",
]

# Column rename maps per dataset → common schema
COLUMN_MAPPINGS: List[Dict[str, str]] = [
    # Framingham
=======
    "framingham.csv",       # H1 - Framingham Heart Study
    "ieee_chd.csv",         # H2 - IEEE Comprehensive Heart Disease
    "cleveland.csv",        # H3 - UCI Cleveland Heart Disease
    "fic_pakistan.csv",     # H4 - FIC Pakistan
    "kaggle_heart.csv",     # H5 - Kaggle Heart Prediction
]

# Column rename maps per dataset -> common 6-feature schema
# Harmonised features: age, sex, trestbps (systolic BP), diaBP (diastolic BP),
#                      chol (cholesterol), fbs (fasting blood glucose)
COLUMN_MAPPINGS: List[Dict[str, str]] = [
    # H1 - Framingham
>>>>>>> 3d539aa >>>>>>>
    {
        "male": "sex",
        "age": "age",
        "totChol": "chol",
        "sysBP": "trestbps",
<<<<<<< HEAD
        "diabetes": "fbs",
        "TenYearCHD": "target",
    },
    # Cleveland
    {
        "num": "target",
        "trestbps": "trestbps",
        "chol": "chol",
        "fbs": "fbs",
        "restecg": "restecg",
        "thalach": "thalach",
        "exang": "exang",
        "oldpeak": "oldpeak",
        "cp": "cp",
        "sex": "sex",
        "age": "age",
    },
    # FIC Pakistan
    {
        "Age": "age",
        "Gender": "sex",
        "Chest pain": "cp",
=======
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
>>>>>>> 3d539aa >>>>>>>
        "Cholestrol": "chol",
        "FBS": "fbs",
        "Mortality": "target",
    },
<<<<<<< HEAD
    # Heart Disease Prediction (Kaggle)
    {
        "Age": "age",
        "Gender": "sex",
        "Chest Pain Type": "cp",
        "Blood Pressure": "trestbps",
        "Cholesterol": "chol",
        "Blood Sugar": "fbs",
        "Heart Rate": "thalach",
        "Exercise Induced Angina": "exang",
        "Heart Disease": "target",
    },
    # Hungarian
    {
        "num": "target",
        "trestbps": "trestbps",
        "chol": "chol",
        "fbs": "fbs",
        "restecg": "restecg",
        "thalach": "thalach",
        "exang": "exang",
        "oldpeak": "oldpeak",
        "cp": "cp",
        "sex": "sex",
        "age": "age",
    },
]

# Harmonised feature set (same order for every client)
FINAL_FEATURES = [
    "age", "sex", "cp", "trestbps", "chol",
    "fbs", "restecg", "thalach", "exang", "oldpeak",
]
=======
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

# Harmonised 6-feature set (same order for every client)
FINAL_FEATURES = ["age", "sex", "trestbps", "diaBP", "chol", "fbs"]
>>>>>>> 3d539aa >>>>>>>
TARGET_COLUMN = "target"


# ---------------------------------------------------------------------------
<<<<<<< HEAD
=======
# Helper: IQR-based outlier capping
# ---------------------------------------------------------------------------

def _iqr_cap(df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    """Clip each numeric feature to [Q1 - 1.5*IQR, Q3 + 1.5*IQR]."""
    df = df.copy()
    for col in features:
        if col not in df.columns:
            continue
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
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
>>>>>>> 3d539aa >>>>>>>
# Public API
# ---------------------------------------------------------------------------

def load_and_preprocess_data(
    data_dir: str = ".",
<<<<<<< HEAD
    test_size: float = 0.2,
=======
    val_size: float = 0.10,
    test_size: float = 0.20,
>>>>>>> 3d539aa >>>>>>>
    random_state: int = 42,
) -> Tuple[
    Optional[List[Tuple[np.ndarray, np.ndarray]]],
    Optional[List[Tuple[np.ndarray, np.ndarray]]],
    Optional[List[str]],
]:
    """Load and harmonise all five cardiovascular datasets.

<<<<<<< HEAD
=======
    The preprocessing pipeline for each client is:
        1. IQR-based outlier capping on numeric features.
        2. Median imputation for remaining missing values.
        3. Binarisation of the target column.
        4. Stratified 70/10/20 train/validation/test split.
        5. SMOTE oversampling on the training fold only.
        6. StandardScaler normalisation (fit on train, applied to val and test).

    Note: the Flower simulation uses only training and test folds. The
    validation fold is stored in the returned ``client_val_datasets`` list
    and can be used for hyper-parameter tuning outside the FL loop.

>>>>>>> 3d539aa >>>>>>>
    Parameters
    ----------
    data_dir:
        Directory that contains the five CSV files.
<<<<<<< HEAD
    test_size:
        Fraction of each client's data to use as a local test set.
=======
    val_size:
        Fraction of each client's data reserved for validation (default 0.10).
    test_size:
        Fraction of each client's data reserved for testing (default 0.20).
>>>>>>> 3d539aa >>>>>>>
    random_state:
        Random seed for reproducibility.

    Returns
    -------
<<<<<<< HEAD
    client_train_datasets : list of (X_train, y_train) arrays, one per client.
    client_test_datasets  : list of (X_test,  y_test)  arrays, one per client.
    filenames             : list of dataset file names (same order as above).
=======
    client_train_datasets : list of (X_train, y_train) arrays (post-SMOTE).
    client_test_datasets  : list of (X_test,  y_test)  arrays.
    filenames             : list of dataset file names (same order).
>>>>>>> 3d539aa >>>>>>>

    All three return values are ``None`` if loading fails for any dataset.
    """
    import os

    print("--- Loading and Preprocessing Datasets for 5 Clients ---")
<<<<<<< HEAD
=======
    print(f"    Features: {FINAL_FEATURES}")
    print(f"    Split: 70% train / {int(val_size*100)}% val / {int(test_size*100)}% test")
>>>>>>> 3d539aa >>>>>>>

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

<<<<<<< HEAD
            # Ensure every feature exists (fill missing ones with NaN for imputation)
=======
            # Ensure every feature column exists (fill missing with NaN for imputation)
>>>>>>> 3d539aa >>>>>>>
            for col in FINAL_FEATURES:
                if col not in df.columns:
                    df[col] = np.nan

<<<<<<< HEAD
            df = df[FINAL_FEATURES + [TARGET_COLUMN]]

            # Harmonise 'sex' string labels → binary
            if df["sex"].dtype == object:
                sex_map = {"Male": 1, "Female": 0, "male": 1, "female": 0}
                df["sex"] = (
                    df["sex"].str.strip().str.capitalize().map(sex_map).fillna(df["sex"])
                )

            # Coerce all columns to numeric; impute with median
            for col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            df.fillna(df.median(numeric_only=True), inplace=True)
            df.fillna(0, inplace=True)

            # Binarise target (any positive class → 1)
            df[TARGET_COLUMN] = (df[TARGET_COLUMN] > 0).astype(int)

            X = df[FINAL_FEATURES].values
            y = df[TARGET_COLUMN].values

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            client_train_datasets.append((X_train_scaled, y_train))
            client_test_datasets.append((X_test_scaled, y_test))

            print(
                f"  Client {i} ({filename}): {len(df)} samples — "
                f"train={len(y_train)}, test={len(y_test)}, "
                f"pos_rate={y.mean():.1%}"
=======
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

            # Step 1: IQR-based outlier capping (on numeric features only)
            df = _iqr_cap(df, FINAL_FEATURES)

            # Step 2: Median imputation
            df.fillna(df.median(numeric_only=True), inplace=True)
            df.fillna(0, inplace=True)

            # Step 3: Binarise target (any positive class -> 1)
            df[TARGET_COLUMN] = (df[TARGET_COLUMN] > 0).astype(int)

            X = df[FINAL_FEATURES].values.astype(np.float32)
            y = df[TARGET_COLUMN].values.astype(np.int64)

            # Step 4: Stratified 70/10/20 split
            # First split off the test set (20%)
            X_trainval, X_test, y_trainval, y_test = train_test_split(
                X, y,
                test_size=test_size,
                random_state=random_state,
                stratify=y,
            )
            # Then split validation from the remaining 80% -> val is 10/80 = 12.5%
            val_fraction_of_trainval = val_size / (1.0 - test_size)
            X_train, X_val, y_train, y_val = train_test_split(
                X_trainval, y_trainval,
                test_size=val_fraction_of_trainval,
                random_state=random_state,
                stratify=y_trainval,
            )

            # Step 5: SMOTE on training fold only
            X_train_res, y_train_res = _apply_smote(X_train, y_train, random_state)

            # Step 6: StandardScaler (fit on pre-SMOTE train to avoid leakage)
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
>>>>>>> 3d539aa >>>>>>>
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
