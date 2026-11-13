"""
rdp_accountant.py - Per-client Renyi Differential Privacy (RDP) accounting.
Wraps Opacus's RDP accounting utilities to expose per-client effective
epsilon at a given delta, given (noise_multiplier, batch_size, n_train,
n_rounds, local_epochs, delta). See README "Privacy Budget Accounting".

Usage
-----
    from dpfedadam.rdp_accountant import RDPAccountant

    accountant = RDPAccountant(noise_multiplier=1.1, max_grad_norm=1.0, batch_size=32, delta=1e-5)
    eps = accountant.compute_epsilon(n_train=211, n_rounds=100, local_epochs=5)
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np


_EPS_CACHE: dict = {}

class RDPAccountant:
    """Per-client RDP privacy accountant based on Opacus utilities.

    Parameters
    ----------
    noise_multiplier : float
        Ratio of the Gaussian noise standard deviation to the l2 sensitivity
        (max_grad_norm). Corresponds to sigma in the paper.
    max_grad_norm : float
        Per-sample gradient clipping norm (l2 sensitivity). Default 1.0.
    batch_size : int
        Mini-batch size used during DP-SGD training. Default 32.
    delta : float
        Target failure probability for the (epsilon, delta)-DP guarantee.
        Set to 1e-5 in all paper experiments.
    """

    def __init__(
        self,
        noise_multiplier: float = 1.1,
        max_grad_norm: float = 1.0,
        batch_size: int = 32,
        delta: float = 1e-5,
    ) -> None:
        self.noise_multiplier = noise_multiplier
        self.max_grad_norm = max_grad_norm
        self.batch_size = batch_size
        self.delta = delta

    @staticmethod
    def steps_per_epoch(n_train: int, batch_size: int) -> int:
        """Lots per local epoch = len(DataLoader) = ceil(n_train / L)."""
        return int(np.ceil(n_train / batch_size))

    @staticmethod
    def sample_rate(n_train: int, batch_size: int) -> float:
        """Poisson inclusion probability used by Opacus: 1 / len(DataLoader)."""
        return 1.0 / RDPAccountant.steps_per_epoch(n_train, batch_size)

    def compute_epsilon(
        self,
        n_train: int,
        n_rounds: int,
        local_epochs: int = 5,
    ) -> float:
        """Effective (epsilon, delta)-DP guarantee for one client after
        ``n_rounds`` rounds of ``local_epochs`` local epochs.

        The accounted mechanism is exactly the one the client executes
        (experiments/revision_runner.py -> Opacus ``make_private``): Poisson subsampling
        in which every one of the ``n_train`` real records is included in
        a lot independently with probability q = 1 / ceil(n_train / L),
        ceil(n_train / L) lots per local epoch, composed with the
        Renyi-DP analysis of the subsampled Gaussian mechanism
        (Mironov et al., 2019). ``n_train`` is the number of real
        training records; the pipeline performs no oversampling."""
        try:
            from opacus.accountants import RDPAccountant as OpacusRDP
        except ImportError:
            raise ImportError(
                "Opacus is required for RDP accounting. "
                "Install with: pip install opacus"
            )

        sample_rate = self.sample_rate(n_train, self.batch_size)
        steps_per_epoch = self.steps_per_epoch(n_train, self.batch_size)
        total_steps = n_rounds * local_epochs * steps_per_epoch
        # The budget depends on the data only through steps_per_epoch, so
        # identical (sigma, delta, steps per epoch, horizon) give identical
        # results; memoize them (the lot-size searches call this often).
        key = (float(self.noise_multiplier), float(self.delta), int(steps_per_epoch),
               int(n_rounds), int(local_epochs))
        if key in _EPS_CACHE:
            return _EPS_CACHE[key]

        accountant = OpacusRDP()
        accountant.history = [(self.noise_multiplier, sample_rate, total_steps)]

        epsilon = float(accountant.get_epsilon(delta=self.delta))
        _EPS_CACHE[key] = epsilon
        return epsilon

    def audit_all_clients(
        self,
        n_train_per_client: List[int],
        n_rounds: int,
        local_epochs: int = 5,
        client_labels: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """Compute effective epsilon for all clients and report the spread.

        Parameters
        ----------
        n_train_per_client : list of int
            Training set sizes for each client.
        n_rounds : int
            Total number of federated communication rounds.
        local_epochs : int
            Local SGD epochs per round.
        client_labels : list of str, optional
            Human-readable labels for each client. Defaults to H1, H2, ...

        Returns
        -------
        dict mapping client label -> effective epsilon
        """
        if client_labels is None:
            client_labels = [f"H{i+1}" for i in range(len(n_train_per_client))]

        results: Dict[str, float] = {}
        print(
            f"\nRDP Privacy Audit  |  sigma={self.noise_multiplier}, "
            f"delta={self.delta:.0e}, rounds={n_rounds}, epochs={local_epochs}"
        )
        print("-" * 60)
        for label, n_train in zip(client_labels, n_train_per_client):
            eps = self.compute_epsilon(n_train, n_rounds, local_epochs)
            results[label] = eps
            print(f"  {label:4s}  n_train={n_train:5d}  effective epsilon={eps:.2f}")

        eps_values = list(results.values())
        spread = max(eps_values) / min(eps_values) if min(eps_values) > 0 else float("inf")
        print("-" * 60)
        print(
            f"  Spread (max/min): {spread:.2f}x  "
            f"[min={min(eps_values):.2f}, max={max(eps_values):.2f}]"
        )
        print(
            "  Note: clients with smaller n_train incur higher effective "
            "epsilon (weaker privacy guarantee).\n"
        )
        return results
