"""
evaluation.py - Centralized, threshold-calibrated evaluation. Selects the
F1-maximizing decision threshold on the validation fold, then reports
metrics on the held-out test fold at that threshold (never the reverse).
See README "Evaluation Protocol" for the pooled/macro/per-client views.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Dict, List, Tuple

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .model import Net


def load_model_from_ndarrays(ndarrays: List[np.ndarray], input_features: int) -> Net:
    """Reconstruct a ``Net`` from a Flower strategy's final ndarray weights
    (e.g. ``parameters_to_ndarrays(strategy.final_weights)``)."""
    model = Net(input_features=input_features)
    state_dict = model.state_dict()
    new_state = OrderedDict(
        (key, torch.tensor(val)) for key, val in zip(state_dict.keys(), ndarrays)
    )
    model.load_state_dict(new_state, strict=True)
    model.eval()
    return model


def predict_proba(model: Net, X: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        outputs = model(torch.tensor(X, dtype=torch.float32))
    return outputs.numpy().flatten()


def evaluate_arrays(
    model: Net, X: np.ndarray, y: np.ndarray, threshold: float = 0.5
) -> Dict[str, float]:
    """Compute accuracy/precision/recall/F1/AUC at a given decision threshold."""
    probs = predict_proba(model, X)
    preds = (probs >= threshold).astype(int)
    metrics = {
        "accuracy": float(accuracy_score(y, preds)),
        "precision": float(precision_score(y, preds, zero_division=0)),
        "recall": float(recall_score(y, preds, zero_division=0)),
        "f1_score": float(f1_score(y, preds, zero_division=0)),
        "threshold": float(threshold),
    }
    metrics["auc"] = float(roc_auc_score(y, probs)) if len(set(y.tolist())) > 1 else float("nan")
    return metrics


def find_best_threshold(
    model: Net, X_val: np.ndarray, y_val: np.ndarray, n_steps: int = 99
) -> float:
    """Scan thresholds in (0, 1) and return the one maximizing validation F1."""
    probs = predict_proba(model, X_val)
    thresholds = np.linspace(0.01, 0.99, n_steps)
    best_threshold, best_f1 = 0.5, -1.0
    for t in thresholds:
        preds = (probs >= t).astype(int)
        f1 = f1_score(y_val, preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_threshold = f1, float(t)
    return best_threshold


def pooled_arrays(
    client_datasets: List[Tuple[np.ndarray, np.ndarray]]
) -> Tuple[np.ndarray, np.ndarray]:
    """Concatenate a list of per-client (X, y) arrays into pooled arrays,
    matching the paper's "aggregated held-out test sets... micro-averaged
    by sample volume" evaluation protocol."""
    X_all = np.concatenate([X for X, _ in client_datasets], axis=0)
    y_all = np.concatenate([y for _, y in client_datasets], axis=0)
    return X_all, y_all


def calibrated_final_evaluation(
    final_ndarrays: List[np.ndarray],
    input_features: int,
    client_val_data: List[Tuple[np.ndarray, np.ndarray]],
    client_test_data: List[Tuple[np.ndarray, np.ndarray]],
    client_names: List[str] = None,
) -> Dict:
    """Load the final global model and report four views of test
    performance: "pooled" (micro-average, single global threshold),
    "macro" (unweighted per-client mean, same global threshold),
    "macro_local_threshold" (each client's own threshold), and
    "per_client" (both thresholds, per site)."""
    model = load_model_from_ndarrays(final_ndarrays, input_features)

    X_val, y_val = pooled_arrays(client_val_data)
    global_threshold = find_best_threshold(model, X_val, y_val)

    labels = client_names or [f"H{i+1}" for i in range(len(client_test_data))]

    macro_metrics = ["accuracy", "precision", "recall", "f1_score", "auc"]
    per_client = {}
    for label, (X_val_c, y_val_c), (X_test_c, y_test_c) in zip(
        labels, client_val_data, client_test_data
    ):
        local_threshold = find_best_threshold(model, X_val_c, y_val_c)
        per_client[label] = {
            "global_threshold": evaluate_arrays(model, X_test_c, y_test_c, threshold=global_threshold),
            "local_threshold": evaluate_arrays(model, X_test_c, y_test_c, threshold=local_threshold),
        }

    def _macro(view: str) -> Dict[str, float]:
        m = {
            k: float(np.nanmean([v[view][k] for v in per_client.values()]))
            for k in macro_metrics
        }
        return m

    macro = _macro("global_threshold")
    macro["threshold"] = global_threshold
    macro_local = _macro("local_threshold")
    macro_local["threshold"] = float("nan")  # per-client, no single value

    X_test, y_test = pooled_arrays(client_test_data)
    pooled = evaluate_arrays(model, X_test, y_test, threshold=global_threshold)

    return {
        "pooled": pooled,
        "macro": macro,
        "macro_local_threshold": macro_local,
        "per_client": per_client,
    }
