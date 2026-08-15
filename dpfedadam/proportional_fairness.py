"""
proportional_fairness.py - Non-fuzzy ablation baseline for
dpfedadam.fuzzy_fairness.FuzzyFairnessController. Recommends a per-client
DP-SGD batch size scaled linearly by dataset size alone (no epsilon
feedback, no fuzzy inference), to isolate what the Mamdani controller's
fuzzy-logic machinery adds over the simplest reasonable reallocation
heuristic. Returns the same FairnessRebalanceReport type as the fuzzy
controller for direct, drop-in comparison.

Usage
-----
    from dpfedadam.proportional_fairness import ProportionalFairnessController
    from dpfedadam.rdp_accountant import RDPAccountant

    controller = ProportionalFairnessController()
    accountant = RDPAccountant(noise_multiplier=2.5, max_grad_norm=1.0, delta=1e-5)
    report = controller.rebalance(
        accountant=accountant, n_train_per_client=[211, 833, 644, 717],
        n_rounds=100, local_epochs=1, client_labels=["H1", "H2", "H3", "H4"],
    )
    print(report)
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from .fuzzy_fairness import ClientRebalanceResult, FairnessRebalanceReport


class ProportionalFairnessController:
    """Non-fuzzy ablation baseline: batch size is scaled linearly by
    n_train relative to the federation mean, with no epsilon feedback.

    L_k = clip(round(base_batch_size * n_train_k / mean(n_train)),
               min_batch_size, max_batch_size)

    This is the simplest heuristic consistent with "give smaller clients
    smaller batches, larger clients larger batches" and is used to test
    whether the Mamdani fuzzy controller's specific inference mechanism
    (epsilon-aware, rule-based) adds value over a size-only linear rule.
    """

    def recommend_multiplier(self, n_train: int, mean_n_train: float) -> float:
        """Return the batch-size multiplier for a single client, as the
        ratio of its dataset size to the federation mean."""
        return float(n_train) / float(mean_n_train)

    def rebalance(
        self,
        accountant,
        n_train_per_client: List[int],
        n_rounds: int,
        local_epochs: int = 5,
        base_batch_size: int = 32,
        min_batch_size: int = 8,
        max_batch_size: int = 128,
        client_labels: Optional[List[str]] = None,
    ) -> FairnessRebalanceReport:
        """Compute the baseline per-client epsilon (at ``base_batch_size``),
        derive a size-proportional per-client batch size, and recompute
        epsilon under the adjusted batch sizes via the same ``accountant``.
        Mirrors FuzzyFairnessController.rebalance()'s signature and return
        type exactly, for a direct ablation comparison.
        """
        if client_labels is None:
            client_labels = [f"H{i+1}" for i in range(len(n_train_per_client))]

        mean_n_train = float(np.mean(n_train_per_client))

        results: List[ClientRebalanceResult] = []
        for label, n_train in zip(client_labels, n_train_per_client):
            accountant.batch_size = base_batch_size
            baseline_eps = accountant.compute_epsilon(n_train, n_rounds, local_epochs)

            mult = self.recommend_multiplier(n_train, mean_n_train)
            adjusted_batch = int(
                np.clip(round(base_batch_size * mult), min_batch_size, max_batch_size)
            )
            adjusted_batch = max(1, min(adjusted_batch, n_train))

            accountant.batch_size = adjusted_batch
            adjusted_eps = accountant.compute_epsilon(n_train, n_rounds, local_epochs)

            results.append(
                ClientRebalanceResult(
                    label=label,
                    n_train=n_train,
                    baseline_epsilon=baseline_eps,
                    baseline_batch_size=base_batch_size,
                    recommended_multiplier=mult,
                    adjusted_batch_size=adjusted_batch,
                    adjusted_epsilon=adjusted_eps,
                )
            )

        accountant.batch_size = base_batch_size  # restore
        return FairnessRebalanceReport(clients=results)
