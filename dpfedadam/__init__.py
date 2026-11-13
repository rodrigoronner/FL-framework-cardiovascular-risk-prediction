"""
DP-FedAdam - privacy-budget auditing and allocation for differentially
private federated cardiovascular risk prediction (ASOC-D-26-13458, R1).

Modules: data loading and harmonization (data_utils), the network
(model), calibrated evaluation (evaluation), per-site Renyi accounting
of the executed Poisson-subsampled Gaussian mechanism (rdp_accountant),
and the lot-size allocation rules (epsilon_matched, proportional_fairness,
fuzzy_fairness). Federated training is simulated in
experiments/revision_runner.py; the fuzzy headroom policy is in
experiments/headroom_study.py.
"""

from .data_utils import get_partition_metadata, get_train_sizes, load_and_preprocess_data
from .epsilon_matched import EpsilonMatchedController
from .fuzzy_fairness import FuzzyFairnessController
from .model import Net
from .proportional_fairness import ProportionalFairnessController
from .rdp_accountant import RDPAccountant

__all__ = [
    "EpsilonMatchedController",
    "FuzzyFairnessController",
    "Net",
    "ProportionalFairnessController",
    "RDPAccountant",
    "get_partition_metadata",
    "get_train_sizes",
    "load_and_preprocess_data",
]
