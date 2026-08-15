"""
FedCVR - Federated Cardiovascular Risk Prediction

Privacy-preserving federated learning: client-side DP-SGD (Opacus),
server-side DP-FedAdam, per-client RDP accounting, fuzzy fairness
rebalancing, and FedProx/FedCluster/FedAdagrad/FedYogi baselines.
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
from .rdp_accountant import RDPAccountant
from .fuzzy_fairness import FuzzyFairnessController

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
    "FuzzyFairnessController",
]
