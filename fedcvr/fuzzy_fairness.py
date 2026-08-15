"""
fuzzy_fairness.py - Mamdani fuzzy controller recommending a per-client
DP-SGD batch size from (dataset size, effective epsilon), to narrow the
per-client privacy-fairness spread reported by the RDP accountant. See
README "Fuzzy fairness rebalancing" for the rule base and rationale.

Usage
-----
    from fedcvr.fuzzy_fairness import FuzzyFairnessController
    from fedcvr.rdp_accountant import RDPAccountant

    controller = FuzzyFairnessController()
    accountant = RDPAccountant(noise_multiplier=1.1, max_grad_norm=1.0, delta=1e-5)
    report = controller.rebalance(
        accountant=accountant, n_train_per_client=[211, 833, 644, 717],
        n_rounds=100, local_epochs=5, client_labels=["H1", "H2", "H3", "H4"],
    )
    print(report)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

try:
    import skfuzzy as fuzz
    from skfuzzy import control as fuzzy_ctrl
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "scikit-fuzzy is required for fedcvr.fuzzy_fairness. "
        "Install with: pip install scikit-fuzzy"
    ) from exc


@dataclass
class ClientRebalanceResult:
    label: str
    n_train: int
    baseline_epsilon: float
    baseline_batch_size: int
    recommended_multiplier: float
    adjusted_batch_size: int
    adjusted_epsilon: float


@dataclass
class FairnessRebalanceReport:
    clients: List[ClientRebalanceResult]

    @property
    def baseline_spread(self) -> float:
        eps = [c.baseline_epsilon for c in self.clients]
        return max(eps) / min(eps) if min(eps) > 0 else float("inf")

    @property
    def adjusted_spread(self) -> float:
        eps = [c.adjusted_epsilon for c in self.clients]
        return max(eps) / min(eps) if min(eps) > 0 else float("inf")

    def __repr__(self) -> str:  # pragma: no cover - convenience only
        lines = [
            f"{'Client':6s} {'n_train':>8s} {'L_base':>7s} {'eps_base':>9s} "
            f"{'mult':>6s} {'L_adj':>6s} {'eps_adj':>8s}"
        ]
        for c in self.clients:
            lines.append(
                f"{c.label:6s} {c.n_train:8d} {c.baseline_batch_size:7d} "
                f"{c.baseline_epsilon:9.2f} {c.recommended_multiplier:6.2f} "
                f"{c.adjusted_batch_size:6d} {c.adjusted_epsilon:8.2f}"
            )
        lines.append(
            f"Spread: {self.baseline_spread:.2f}x -> {self.adjusted_spread:.2f}x"
        )
        return "\n".join(lines)


class FuzzyFairnessController:
    """Mamdani fuzzy controller recommending a per-client batch-size
    multiplier from (dataset size, effective epsilon), to narrow the
    per-client privacy-fairness spread reported by the RDP accountant.

    Parameters
    ----------
    n_train_universe : (float, float)
        Min/max of the dataset-size universe of discourse.
    epsilon_universe : (float, float)
        Min/max of the effective-epsilon universe of discourse.
    multiplier_universe : (float, float)
        Min/max of the batch-size multiplier applied to the base batch size.
    """

    def __init__(
        self,
        n_train_universe: tuple = (0, 3200),
        epsilon_universe: tuple = (0, 20),
        multiplier_universe: tuple = (0.4, 2.2),
    ) -> None:
        n_lo, n_hi = n_train_universe
        e_lo, e_hi = epsilon_universe
        m_lo, m_hi = multiplier_universe

        n_train = fuzzy_ctrl.Antecedent(np.linspace(n_lo, n_hi, 321), "n_train")
        epsilon = fuzzy_ctrl.Antecedent(np.linspace(e_lo, e_hi, 201), "epsilon")
        multiplier = fuzzy_ctrl.Consequent(np.linspace(m_lo, m_hi, 181), "multiplier")

        # Dataset-size membership functions: small / medium / large
        n_train["small"] = fuzz.trimf(n_train.universe, [n_lo, n_lo, n_lo + 0.45 * (n_hi - n_lo)])
        n_train["medium"] = fuzz.trimf(
            n_train.universe,
            [n_lo + 0.15 * (n_hi - n_lo), n_lo + 0.45 * (n_hi - n_lo), n_lo + 0.75 * (n_hi - n_lo)],
        )
        n_train["large"] = fuzz.trimf(n_train.universe, [n_lo + 0.55 * (n_hi - n_lo), n_hi, n_hi])

        # Effective-epsilon membership functions: low / moderate / high
        epsilon["low"] = fuzz.trimf(epsilon.universe, [e_lo, e_lo, e_lo + 0.45 * (e_hi - e_lo)])
        epsilon["moderate"] = fuzz.trimf(
            epsilon.universe,
            [e_lo + 0.15 * (e_hi - e_lo), e_lo + 0.45 * (e_hi - e_lo), e_lo + 0.75 * (e_hi - e_lo)],
        )
        epsilon["high"] = fuzz.trimf(epsilon.universe, [e_lo + 0.55 * (e_hi - e_lo), e_hi, e_hi])

        # Output membership functions: decrease / keep / increase batch size
        multiplier["decrease"] = fuzz.trimf(multiplier.universe, [m_lo, m_lo, 1.0])
        multiplier["keep"] = fuzz.trimf(multiplier.universe, [0.75, 1.0, 1.25])
        multiplier["increase"] = fuzz.trimf(multiplier.universe, [1.0, m_hi, m_hi])

        # 3x3 rule base over (n_train, epsilon) -> multiplier.
        rules = [
            fuzzy_ctrl.Rule(n_train["small"] & epsilon["high"], multiplier["decrease"]),
            fuzzy_ctrl.Rule(n_train["small"] & epsilon["moderate"], multiplier["decrease"]),
            fuzzy_ctrl.Rule(n_train["small"] & epsilon["low"], multiplier["keep"]),
            fuzzy_ctrl.Rule(n_train["medium"] & epsilon["high"], multiplier["decrease"]),
            fuzzy_ctrl.Rule(n_train["medium"] & epsilon["moderate"], multiplier["keep"]),
            fuzzy_ctrl.Rule(n_train["medium"] & epsilon["low"], multiplier["increase"]),
            fuzzy_ctrl.Rule(n_train["large"] & epsilon["high"], multiplier["keep"]),
            fuzzy_ctrl.Rule(n_train["large"] & epsilon["moderate"], multiplier["keep"]),
            fuzzy_ctrl.Rule(n_train["large"] & epsilon["low"], multiplier["increase"]),
        ]

        self._n_train_var = n_train
        self._epsilon_var = epsilon
        self._multiplier_var = multiplier
        self._system = fuzzy_ctrl.ControlSystem(rules)

    def recommend_multiplier(self, n_train: int, epsilon: float) -> float:
        """Return the defuzzified (centroid) batch-size multiplier for a
        single client, given its training-set size and current effective
        epsilon."""
        n_lo, n_hi = self._n_train_var.universe.min(), self._n_train_var.universe.max()
        e_lo, e_hi = self._epsilon_var.universe.min(), self._epsilon_var.universe.max()

        sim = fuzzy_ctrl.ControlSystemSimulation(self._system)
        sim.input["n_train"] = float(np.clip(n_train, n_lo, n_hi))
        sim.input["epsilon"] = float(np.clip(epsilon, e_lo, e_hi))
        sim.compute()
        return float(sim.output["multiplier"])

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
        derive a fuzzy-recommended per-client batch size, and recompute
        epsilon under the adjusted batch sizes via the same ``accountant``.

        Parameters
        ----------
        accountant : fedcvr.rdp_accountant.RDPAccountant
            Pre-configured with (noise_multiplier, max_grad_norm, delta).
            Its own ``batch_size`` attribute is overridden per-call below.
        """
        if client_labels is None:
            client_labels = [f"H{i+1}" for i in range(len(n_train_per_client))]

        results: List[ClientRebalanceResult] = []
        for label, n_train in zip(client_labels, n_train_per_client):
            accountant.batch_size = base_batch_size
            baseline_eps = accountant.compute_epsilon(n_train, n_rounds, local_epochs)

            mult = self.recommend_multiplier(n_train, baseline_eps)
            adjusted_batch = int(
                np.clip(round(base_batch_size * mult), min_batch_size, max_batch_size)
            )
            # Never let the adjusted batch exceed the client's own dataset size.
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


