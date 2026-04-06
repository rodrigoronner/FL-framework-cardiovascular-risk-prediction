"""
strategy.py – FedCVR server-side aggregation strategy.

``FedCVRStrategy`` extends Flower's ``FedAvg`` with an Adam-style server
optimiser that applies bias-corrected first- and second-moment estimation to
the aggregated pseudo-gradient (Δ = avg_client_update − current_global_weights).

Server update rule (per round t)
---------------------------------
    Δ_t  = FedAvg(client_updates) − w_t          # pseudo-gradient
    m_t  = β₁ · m_{t-1} + (1 − β₁) · Δ_t        # 1st moment
    v_t  = β₂ · v_{t-1} + (1 − β₂) · Δ_t²       # 2nd moment
    m̂_t  = m_t / (1 − β₁ᵗ)                       # bias correction
    v̂_t  = v_t / (1 − β₂ᵗ)                       # bias correction
    w_{t+1} = w_t + η · m̂_t / (√v̂_t + ε)        # parameter update

Default hyper-parameters match those used in the paper experiments:
    η = 0.01,  β₁ = 0.9,  β₂ = 0.999,  ε = 1e-8

The class also stores per-round per-client evaluation metrics so that
results can be inspected or exported after the simulation.
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


class FedCVRStrategy(FedAvg):
    """FedAvg + Adam-style server optimiser + per-client metric logging.

    Parameters
    ----------
    eta   : Server learning rate (η).
    beta_1: Exponential decay for 1st moment (β₁).
    beta_2: Exponential decay for 2nd moment (β₂).
    tau   : Numerical stability constant (ε).
    **kwargs: Forwarded verbatim to ``FedAvg.__init__``.
    """

    def __init__(
        self,
        *,
        eta: float = 0.01,
        beta_1: float = 0.9,
        beta_2: float = 0.999,
        tau: float = 1e-8,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.eta = eta
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.tau = tau

        # Moment vectors – initialised on first aggregation call
        self._m: Optional[List[np.ndarray]] = None
        self._v: Optional[List[np.ndarray]] = None
        self._current_weights: Optional[List[np.ndarray]] = None

        # Metric history: {round: {cid: {metric: value}}}
        self.client_metrics_history: Dict[int, Dict[str, Dict]] = {}
        # Final aggregated model weights (Parameters object)
        self.final_weights: Optional[Parameters] = None

    # ------------------------------------------------------------------
    # Fit aggregation – apply server Adam optimiser
    # ------------------------------------------------------------------

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

        # On the very first round, bootstrap moment vectors and current weights
        if self._current_weights is None:
            self._current_weights = [np.zeros_like(p) for p in aggregated_ndarrays]
            self._m = [np.zeros_like(p) for p in aggregated_ndarrays]
            self._v = [np.zeros_like(p) for p in aggregated_ndarrays]

        # Pseudo-gradient: difference between aggregated update and current weights
        delta = [
            agg - cur
            for agg, cur in zip(aggregated_ndarrays, self._current_weights)
        ]

        # Moment updates
        t = server_round
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
        new_weights = [
            w + self.eta * mh / (np.sqrt(vh) + self.tau)
            for w, mh, vh in zip(self._current_weights, m_hat, v_hat)
        ]

        # Persist updated weights for the next round
        self._current_weights = new_weights
        updated_parameters = ndarrays_to_parameters(new_weights)
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
                client_proxy.cid: {"loss": res.loss, **res.metrics}
                for client_proxy, res in results
            }

        return aggregated_loss, aggregated_metrics
