"""
<<<<<<< HEAD
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
=======
FedCVR - Federated Cardiovascular Risk Prediction
==================================================
A privacy-preserving federated learning framework combining:
  - Client-side Differential Privacy (Opacus / Gaussian mechanism, DP-SGD)
  - Proximal regularisation (FedProx-style, mu=0.1)
  - Adaptive server-side moment estimation (DP-FedAdam, eta=0.1, tau=1e-3)
  - Per-client RDP accounting for fairness auditing

Paper: "Interpretable Differentially Private Federated Learning for
        Cardiovascular Risk Prediction: Mechanistic Transparency and
        Fairness Auditing"
        Engineering Applications of Artificial Intelligence (submitted)
"""

from .model import Net
from .client import FedCVRClient, build_client
from .strategy import FedCVRStrategy
from .data_utils import load_and_preprocess_data, aggregate_metrics_fn
from .rdp_accountant import RDPAccountant, audit_paper_scenario
>>>>>>> 3d539aa (fix: align code with paper (architecture, features, datasets, hyperparams))

__all__ = [
    "Net",
    "FedCVRClient",
<<<<<<< HEAD
    "FedCVRStrategy",
    "load_and_preprocess_data",
    "aggregate_metrics_fn",
=======
    "build_client",
    "FedCVRStrategy",
    "load_and_preprocess_data",
    "aggregate_metrics_fn",
    "RDPAccountant",
    "audit_paper_scenario",
>>>>>>> 3d539aa (fix: align code with paper (architecture, features, datasets, hyperparams))
]
