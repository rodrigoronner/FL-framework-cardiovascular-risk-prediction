"""
baselines.py - Competing server-side aggregation strategies benchmarked
alongside FedCVRStrategy: FedAdagrad and FedYogi (FedOpt template, Reddi
et al. 2021) subclass ``_AdaptiveServerStrategy`` (fedcvr.strategy) and
supply their own second-moment update rule; FedCluster runs independent
FedAvg within each of k pre-computed client clusters (KMeans over each
client's mean feature vector, see data_utils.cluster_clients_by_distribution).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg

from .strategy import _AdaptiveServerStrategy, _partition_key


class FedAdagradStrategy(_AdaptiveServerStrategy):
    """FedAvg + FedAdagrad server optimiser (Reddi et al., 2021).

    Accumulates squared pseudo-gradients (second moment only - no decay,
    no first-moment/momentum term), so it adapts per-coordinate step sizes
    but lacks the cross-round noise smoothing that momentum provides:

        v_t = v_{t-1} + Δ_t²
        w_{t+1} = w_t + η · Δ_t / (√v_t + τ)
    """

    def __init__(self, *, eta: float = 0.1, tau: float = 1e-3, **kwargs) -> None:
        super().__init__(**kwargs)
        self.eta = eta
        self.tau = tau
        self._v: Optional[List[np.ndarray]] = None

    def _init_state(self, aggregated_ndarrays: List[np.ndarray]) -> None:
        self._v = [np.zeros_like(p) for p in aggregated_ndarrays]

    def _server_step(
        self, delta: List[np.ndarray], server_round: int
    ) -> List[np.ndarray]:
        self._v = [v + d ** 2 for v, d in zip(self._v, delta)]
        return [
            w + self.eta * d / (np.sqrt(v) + self.tau)
            for w, d, v in zip(self._current_weights, delta, self._v)
        ]


class FedYogiStrategy(_AdaptiveServerStrategy):
    """FedAvg + FedYogi server optimiser (Reddi et al., 2021; Zaheer et al.,
    2018). Maintains both moments, but the second moment uses a sign-based
    additive rule instead of Adam's exponential moving average:

        m_t = β₁ · m_{t-1} + (1 − β₁) · Δ_t
        v_t = v_{t-1} − (1 − β₂) · sign(v_{t-1} − Δ_t²) · Δ_t²
        w_{t+1} = w_t + η · m_t / (√|v_t| + τ)

    ``|v_t|`` guards against the sign-based rule driving v_t negative.
    """

    def __init__(
        self,
        *,
        eta: float = 0.1,
        beta_1: float = 0.9,
        beta_2: float = 0.999,
        tau: float = 1e-3,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.eta = eta
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.tau = tau
        self._m: Optional[List[np.ndarray]] = None
        self._v: Optional[List[np.ndarray]] = None

    def _init_state(self, aggregated_ndarrays: List[np.ndarray]) -> None:
        self._m = [np.zeros_like(p) for p in aggregated_ndarrays]
        self._v = [np.zeros_like(p) for p in aggregated_ndarrays]

    def _server_step(
        self, delta: List[np.ndarray], server_round: int
    ) -> List[np.ndarray]:
        self._m = [
            self.beta_1 * m_prev + (1.0 - self.beta_1) * d
            for m_prev, d in zip(self._m, delta)
        ]
        self._v = [
            v_prev - (1.0 - self.beta_2) * np.sign(v_prev - d ** 2) * (d ** 2)
            for v_prev, d in zip(self._v, delta)
        ]
        return [
            w + self.eta * m / (np.sqrt(np.abs(v)) + self.tau)
            for w, m, v in zip(self._current_weights, self._m, self._v)
        ]


class FedClusterStrategy(FedAvg):
    """Clustered FL baseline: independent FedAvg per pre-computed client
    cluster (no adaptive optimizer, no proximal term). A client only ever
    receives and contributes to its own cluster's model, via
    ``configure_fit``/``configure_evaluate`` overrides that substitute
    per-cluster parameters."""

    def __init__(self, *, cluster_assignment: Dict[str, int], **kwargs) -> None:
        super().__init__(**kwargs)
        self.cluster_assignment = cluster_assignment
        self.n_clusters = len(set(cluster_assignment.values()))
        self._cluster_weights: Dict[int, Optional[List[np.ndarray]]] = {
            c: None for c in range(self.n_clusters)
        }
        self.client_metrics_history: Dict[int, Dict[str, Dict]] = {}
        self.final_weights: Optional[Parameters] = None

    def _cluster_of(self, client: ClientProxy) -> int:
        return self.cluster_assignment[_partition_key(client)]

    def _cluster_parameters(self, cluster: int, fallback: Parameters) -> Parameters:
        weights = self._cluster_weights[cluster]
        return fallback if weights is None else ndarrays_to_parameters(weights)

    # ------------------------------------------------------------------
    # Broadcast: route each client its own cluster's current parameters
    # ------------------------------------------------------------------

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        base_instructions = super().configure_fit(server_round, parameters, client_manager)
        return [
            (client, FitIns(self._cluster_parameters(self._cluster_of(client), parameters), fit_ins.config))
            for client, fit_ins in base_instructions
        ]

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        base_instructions = super().configure_evaluate(server_round, parameters, client_manager)
        return [
            (client, EvaluateIns(self._cluster_parameters(self._cluster_of(client), parameters), eval_ins.config))
            for client, eval_ins in base_instructions
        ]

    # ------------------------------------------------------------------
    # Fit aggregation – independent FedAvg per cluster
    # ------------------------------------------------------------------

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        if not results:
            return None, {}

        by_cluster: Dict[int, List[Tuple[ClientProxy, FitRes]]] = {
            c: [] for c in range(self.n_clusters)
        }
        for client, fit_res in results:
            by_cluster[self._cluster_of(client)].append((client, fit_res))

        for cluster, cluster_results in by_cluster.items():
            if not cluster_results:
                continue
            total_examples = sum(fit_res.num_examples for _, fit_res in cluster_results)
            first_ndarrays = parameters_to_ndarrays(cluster_results[0][1].parameters)
            weighted_sum = [np.zeros_like(p) for p in first_ndarrays]
            for _, fit_res in cluster_results:
                weight = fit_res.num_examples / total_examples
                ndarrays = parameters_to_ndarrays(fit_res.parameters)
                weighted_sum = [acc + weight * p for acc, p in zip(weighted_sum, ndarrays)]
            self._cluster_weights[cluster] = weighted_sum

        # Representative "global" parameters for logging: largest cluster.
        largest_cluster = max(
            range(self.n_clusters),
            key=lambda c: sum(1 for cc in self.cluster_assignment.values() if cc == c),
        )
        updated_parameters = ndarrays_to_parameters(self._cluster_weights[largest_cluster])
        self.final_weights = updated_parameters
        return updated_parameters, {}

    # ------------------------------------------------------------------
    # Evaluate aggregation – collect per-client metrics
    # ------------------------------------------------------------------

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ):
        aggregated_loss, aggregated_metrics = super().aggregate_evaluate(
            server_round, results, failures
        )

        if results:
            self.client_metrics_history[server_round] = {
                _partition_key(client_proxy): {"loss": res.loss, **res.metrics}
                for client_proxy, res in results
            }

        return aggregated_loss, aggregated_metrics
