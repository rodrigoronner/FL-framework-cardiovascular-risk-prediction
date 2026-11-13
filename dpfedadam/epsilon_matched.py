"""
epsilon_matched.py - Non-fuzzy reference allocator: exhaustive search.

For a target budget eps*, each client receives the LARGEST integer lot
size L in [min_batch_size, max_batch_size] whose accounted epsilon does
not exceed eps*. Because epsilon is monotone in L at fixed (sigma, E, T),
this is the allocation that equalizes per-client epsilon as tightly as
the integer grid allows, at a cost of at most (max - min + 1) accountant
calls per client. It is the like-for-like baseline against which the
Mamdani controller (dpfedadam.fuzzy_fairness) must be judged: same
worst-case epsilon, no fuzzy inference.
"""

from __future__ import annotations

from typing import List, Optional

from .fuzzy_fairness import ClientRebalanceResult, FairnessRebalanceReport


class EpsilonMatchedController:
    def __init__(self, target_epsilon: float) -> None:
        self.target_epsilon = float(target_epsilon)

    def rebalance(
        self,
        accountant,
        n_train_per_client: List[int],
        n_rounds: int,
        local_epochs: int = 1,
        base_batch_size: int = 32,
        min_batch_size: int = 8,
        max_batch_size: int = 128,
        client_labels: Optional[List[str]] = None,
    ) -> FairnessRebalanceReport:
        if client_labels is None:
            client_labels = [f"H{i+1}" for i in range(len(n_train_per_client))]
        results: List[ClientRebalanceResult] = []
        for label, n_train in zip(client_labels, n_train_per_client):
            accountant.batch_size = base_batch_size
            baseline_eps = accountant.compute_epsilon(n_train, n_rounds, local_epochs)
            best_l, best_eps = min_batch_size, None
            for lot in range(min_batch_size, min(max_batch_size, n_train) + 1):
                accountant.batch_size = lot
                eps = accountant.compute_epsilon(n_train, n_rounds, local_epochs)
                if eps <= self.target_epsilon + 1e-9:
                    best_l, best_eps = lot, eps
            if best_eps is None:
                accountant.batch_size = best_l
                best_eps = accountant.compute_epsilon(n_train, n_rounds, local_epochs)
            results.append(ClientRebalanceResult(
                label=label, n_train=n_train, baseline_epsilon=baseline_eps,
                baseline_batch_size=base_batch_size,
                recommended_multiplier=best_l / base_batch_size,
                adjusted_batch_size=best_l, adjusted_epsilon=best_eps,
            ))
        accountant.batch_size = base_batch_size
        return FairnessRebalanceReport(clients=results)
