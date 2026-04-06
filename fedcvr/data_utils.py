"""
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
    "framingham.csv",
    "cleveland.csv",
    "fic_pakistan.csv",
    "heart_disease_prediction.csv",
    "Hungarian-98-10.csv",
]

# Column rename maps per dataset → common schema
COLUMN_MAPPINGS: List[Dict[str, str]] = [
    # Framingham
    {
        "male": "sex",
        "age": "age",
        "totChol": "chol",
        "sysBP": "trestbps",
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
        "Cholestrol": "chol",
        "FBS": "fbs",
        "Mortality": "target",
    },
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
TARGET_COLUMN = "target"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_and_preprocess_data(
    data_dir: str = ".",
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[
    Optional[List[Tuple[np.ndarray, np.ndarray]]],
    Optional[List[Tuple[np.ndarray, np.ndarray]]],
    Optional[List[str]],
]:
    """Load and harmonise all five cardiovascular datasets.

    Parameters
    ----------
    data_dir:
        Directory that contains the five CSV files.
    test_size:
        Fraction of each client's data to use as a local test set.
    random_state:
        Random seed for reproducibility.

    Returns
    -------
    client_train_datasets : list of (X_train, y_train) arrays, one per client.
    client_test_datasets  : list of (X_test,  y_test)  arrays, one per client.
    filenames             : list of dataset file names (same order as above).

    All three return values are ``None`` if loading fails for any dataset.
    """
    import os

    print("--- Loading and Preprocessing Datasets for 5 Clients ---")

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

            # Ensure every feature exists (fill missing ones with NaN for imputation)
            for col in FINAL_FEATURES:
                if col not in df.columns:
                    df[col] = np.nan

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
