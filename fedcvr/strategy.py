"""
strategy.py - DP-FedAdam server-side aggregation strategy. ``DPFedAdamStrategy``
extends Flower's ``FedAvg`` with a DP-FedAdam server optimizer applying
bias-corrected first/second-moment estimation to the aggregated
pseudo-gradient (delta = avg_client_update - current_global_weights):

    delta_t = FedAvg(client_updates) - w_t
    m_t = b1*m_{t-1} + (1-b1)*delta_t
    v_t = b2*v_{t-1} + (1-b2)*delta_t^2
    w_{t+1} = w_t + eta * m_hat_t / (sqrt(v_hat_t) + tau)

``_AdaptiveServerStrategy`` factors out the plumbing shared by every
stateful server optimizer here and in ``fedcvr.baselines``
(FedAdagrad/FedYogi): bootstrapping w_0 from round 1's plain FedAvg
result, and per-client evaluation metric logging.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from flwr.common import (
    EvaluateRes,
    FitRes,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg


def _partition_key(client: ClientProxy) -> str:
    """Stable per-client dataset-index key ("0".."N-1"), robust to Flower
    versions where ``ClientProxy.cid`` is a large random node id rather
    than the sequential index passed to ``client_fn``. Newer Flower
    simulation backends (``RayActorClientProxy``) expose the real
    sequential index as ``.partition_id``; older ones set ``cid`` itself
    to that index, so this falls back to ``cid`` when ``partition_id``
    is absent."""
    return str(getattr(client, "partition_id", client.cid))


class _AdaptiveServerStrategy(FedAvg):
    """Shared bootstrap + per-client metric logging for stateful server
    optimisers built on top of FedAvg's weighted-average aggregation."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._current_weights: Optional[List[np.ndarray]] = None

        # Metric history: {round: {cid: {metric: value}}}
        self.client_metrics_history: Dict[int, Dict[str, Dict]] = {}
        # Final aggregated model weights (Parameters object)
        self.final_weights: Optional[Parameters] = None

    def _init_state(self, aggregated_ndarrays: List[np.ndarray]) -> None:
        """Initialise optimiser-specific state (moment vectors, ...)."""
        raise NotImplementedError

    def _server_step(
        self, delta: List[np.ndarray], server_round: int
    ) -> List[np.ndarray]:
        """Compute the next global weights from the pseudo-gradient delta."""
        raise NotImplementedError

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        # Standard FedAvg weighted average
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )

        if aggregated_parameters is None:
            return aggregated_parameters, aggregated_metrics

        aggregated_ndarrays = parameters_to_ndarrays(aggregated_parameters)

        # Bootstrap w_0 from round 1's plain FedAvg result (no prior
        # global model to diff against yet); moment vectors start at zero.
        if self._current_weights is None:
            self._current_weights = [np.array(p, copy=True) for p in aggregated_ndarrays]
            self._init_state(aggregated_ndarrays)
            updated_parameters = ndarrays_to_parameters(self._current_weights)
            self.final_weights = updated_parameters
            return updated_parameters, aggregated_metrics

        # Pseudo-gradient: difference between aggregated update and current weights
        delta = [
            agg - cur
            for agg, cur in zip(aggregated_ndarrays, self._current_weights)
        ]

        self._current_weights = self._server_step(delta, server_round)
        updated_parameters = ndarrays_to_parameters(self._current_weights)
        self.final_weights = updated_parameters

        return updated_parameters, aggregated_metrics

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


class DPFedAdamStrategy(_AdaptiveServerStrategy):
    """FedAvg + Adam-style server optimiser + per-client metric logging.

    Parameters
    ----------
    eta   : Server learning rate (η). ``eta=0.0`` disables the Adam
            optimiser entirely, reducing to plain FedAvg (used to emulate
            the FedAvg / FedProx baselines).
    beta_1: Exponential decay for 1st moment (β₁).
    beta_2: Exponential decay for 2nd moment (β₂).
    tau   : Numerical stability constant (ε).
    **kwargs: Forwarded verbatim to ``FedAvg.__init__``.
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

        # Moment vectors – initialized on first aggregation call
        self._m: Optional[List[np.ndarray]] = None
        self._v: Optional[List[np.ndarray]] = None

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        if self.eta == 0.0:
            # eta=0 disables the Adam step, reducing to plain FedAvg
            # (used to emulate the FedAvg/FedProx baselines).
            aggregated_parameters, aggregated_metrics = FedAvg.aggregate_fit(
                self, server_round, results, failures
            )
            if aggregated_parameters is not None:
                self.final_weights = aggregated_parameters
            return aggregated_parameters, aggregated_metrics

        return super().aggregate_fit(server_round, results, failures)

    def _init_state(self, aggregated_ndarrays: List[np.ndarray]) -> None:
        self._m = [np.zeros_like(p) for p in aggregated_ndarrays]
        self._v = [np.zeros_like(p) for p in aggregated_ndarrays]

    def _server_step(
        self, delta: List[np.ndarray], server_round: int
    ) -> List[np.ndarray]:
        t = server_round

        # Moment updates
        self._m = [
            self.beta_1 * m_prev + (1.0 - self.beta_1) * d
            for m_prev, d in zip(self._m, delta)
        ]
        self._v = [
            self.beta_2 * v_prev + (1.0 - self.beta_2) * (d ** 2)
            for v_prev, d in zip(self._v, delta)
        ]

        # Bias-corrected moments
        m_hat = [m / (1.0 - self.beta_1 ** t) for m in self._m]
        v_hat = [v / (1.0 - self.beta_2 ** t) for v in self._v]

        # Adam parameter update
        return [
            w + self.eta * mh / (np.sqrt(vh) + self.tau)
            for w, mh, vh in zip(self._current_weights, m_hat, v_hat)
        ]
