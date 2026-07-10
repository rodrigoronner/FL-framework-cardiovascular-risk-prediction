"""
rdp_accountant.py - Per-client Renyi Differential Privacy (RDP) accounting.

Implements the RDP accountant described in Section 3.3 of the paper:
"Interpretable Differentially Private Federated Learning for Cardiovascular
Risk Prediction: Mechanistic Transparency and Fairness Auditing."

Background
----------
Opacus internally uses RDP accounting (Mironov, 2017) to track privacy
expenditure. This module wraps Opacus's accounting utilities to expose
per-client effective epsilon values at a given delta, enabling the fairness
audit described in Section 4.4 of the paper.

The effective epsilon per client depends on:
  - noise_multiplier (sigma): higher sigma -> lower epsilon (stronger privacy)
  - n_train: number of training samples per client; smaller datasets require
    more passes over the data per epoch, which amplifies the privacy cost
  - batch_size: affects the sampling rate q = batch_size / n_train
  - n_rounds * local_epochs: total number of gradient steps
  - delta: target failure probability (set to 1e-5 in the paper)

Per-client epsilon spread
-------------------------
The paper reports a 2.8x spread in effective epsilon across the five
hospital clients (H1-H5), driven by dataset-size imbalance:
  n_train ranges from 644 (H3, UCI Cleveland) to 2,967 (H1, Framingham).
Smaller datasets yield higher effective epsilon (weaker privacy guarantees)
because the same number of gradient steps corresponds to more epochs and a
higher sampling rate relative to the dataset size.

Usage
-----
    from fedcvr.rdp_accountant import RDPAccountant

    accountant = RDPAccountant(
        noise_multiplier=1.1,
        max_grad_norm=1.0,
        batch_size=32,
        delta=1e-5,
    )

    # Compute effective epsilon after T rounds x E local epochs per client
    eps = accountant.compute_epsilon(
        n_train=2967,     # H1 - Framingham training samples
        n_rounds=100,
        local_epochs=5,
    )
    print(f"H1 effective epsilon: {eps:.2f}")
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np


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

    def compute_epsilon(
        self,
        n_train: int,
        n_rounds: int,
        local_epochs: int = 5,
    ) -> float:
        """Compute the effective (epsilon, delta)-DP guarantee for one client.

        Uses Opacus's ``privacy_analysis`` module (PRV accountant under the
        hood) to convert RDP curves to (eps, delta)-DP.

        Parameters
        ----------
        n_train : int
            Number of training samples for this client (post-SMOTE if SMOTE
            was applied, since accounting should reflect actual gradient steps).
        n_rounds : int
            Total number of federated communication rounds.
        local_epochs : int
            Number of local SGD epochs per federated round.

        Returns
        -------
        float
            Effective epsilon at the configured delta.

        Raises
        ------
        ImportError
            If Opacus is not installed.
        """
        try:
            from opacus.accountants.utils import get_noise_multiplier
            from opacus.accountants import RDPAccountant as OpacusRDP
        except ImportError:
            raise ImportError(
                "Opacus is required for RDP accounting. "
                "Install with: pip install opacus"
            )

        # Sampling rate: fraction of dataset per mini-batch
        sample_rate = self.batch_size / n_train

        # Total number of gradient steps = rounds x local_epochs x steps_per_epoch
        steps_per_epoch = int(np.ceil(n_train / self.batch_size))
        total_steps = n_rounds * local_epochs * steps_per_epoch

        accountant = OpacusRDP()
        accountant.history = [(self.noise_multiplier, sample_rate, total_steps)]

        epsilon = accountant.get_epsilon(delta=self.delta)
        return float(epsilon)

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
            Training set sizes for each client (H1 through H5).
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


# ---------------------------------------------------------------------------
# Convenience function for quick audits
# ---------------------------------------------------------------------------

def audit_paper_scenario(n_rounds: int = 100, local_epochs: int = 5) -> Dict[str, float]:
    """Reproduce the per-client epsilon audit from Table 4 of the paper.

    Uses the training set sizes reported in the paper for the five hospital
    clients (H1-H5) under the FedCVR+DP configuration (sigma=1.1, delta=1e-5).

    Parameters
    ----------
    n_rounds : int
        Number of federated rounds (default 100, as used in the paper).
    local_epochs : int
        Local SGD epochs per round (default 5, as used in the paper).

    Returns
    -------
    dict mapping client label -> effective epsilon
    """
    # Training set sizes (70% of each dataset, before SMOTE)
    # H1: Framingham 4238 * 0.7 ~ 2967
    # H2: IEEE-CHD   1190 * 0.7 ~  833
    # H3: Cleveland   920 * 0.7 ~  644
    # H4: FIC Pakistan 1000 * 0.7 ~ 700
    # H5: Kaggle HD   1025 * 0.7 ~  718
    n_train_per_client = [2967, 833, 644, 700, 718]
    client_labels = ["H1 (Framingham)", "H2 (IEEE-CHD)", "H3 (Cleveland)",
                     "H4 (FIC-PK)", "H5 (Kaggle)"]

    accountant = RDPAccountant(
        noise_multiplier=1.1,
        max_grad_norm=1.0,
        batch_size=32,
        delta=1e-5,
    )
    return accountant.audit_all_clients(
        n_train_per_client=n_train_per_client,
        n_rounds=n_rounds,
        local_epochs=local_epochs,
        client_labels=client_labels,
    )


if __name__ == "__main__":
    audit_paper_scenario()
