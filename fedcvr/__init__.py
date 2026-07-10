"""
FedCVR - Federated Cardiovascular Risk Prediction
==================================================
A privacy-preserving federated learning framework combining:
  - Client-side Differential Privacy (Opacus / Gaussian mechanism, DP-SGD)
  - Adaptive server-side moment estimation (DP-FedAdam, eta=0.1, tau=1e-3)
  - Per-client RDP accounting for fairness auditing
  - Baselines for benchmarking (FedProx, FedCluster, FedAdagrad, FedYogi)

Paper: "Interpretable Differentially Private Federated Learning for
        Cardiovascular Risk Prediction: Mechanistic Transparency and
        Fairness Auditing"
        Engineering Applications of Artificial Intelligence (submitted)
"""

from .model import Net
from .client import FedCVRClient, build_client
from .strategy import FedCVRStrategy
from .baselines import FedAdagradStrategy, FedYogiStrategy, FedClusterStrategy
from .data_utils import (
    load_and_preprocess_data,
    aggregate_metrics_fn,
    cluster_clients_by_distribution,
)
from .rdp_accountant import RDPAccountant, audit_paper_scenario

__all__ = [
    "Net",
    "FedCVRClient",
    "build_client",
    "FedCVRStrategy",
    "FedAdagradStrategy",
    "FedYogiStrategy",
    "FedClusterStrategy",
    "load_and_preprocess_data",
    "aggregate_metrics_fn",
    "cluster_clients_by_distribution",
    "RDPAccountant",
    "audit_paper_scenario",
]
