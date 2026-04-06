"""
FedCVR – Federated Cardiovascular Risk Prediction
==================================================
A secure federated learning framework combining:
  - Client-side Differential Privacy (Opacus / Gaussian mechanism)
  - Proximal regularisation (FedProx-style)
  - Adaptive server-side moment estimation (FedAdam-style)

Repository: https://github.com/rodrigo-tertulino/fedcvr
Paper: iSys – Brazilian Journal of Information Systems, 2025
"""

from .model import Net
from .client import FedCVRClient
from .strategy import FedCVRStrategy
from .data_utils import load_and_preprocess_data, aggregate_metrics_fn

__all__ = [
    "Net",
    "FedCVRClient",
    "FedCVRStrategy",
    "load_and_preprocess_data",
    "aggregate_metrics_fn",
]
