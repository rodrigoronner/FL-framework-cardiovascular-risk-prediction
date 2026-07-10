"""
strategy.py – FedCVR server-side aggregation strategy.

``FedCVRStrategy`` extends Flower's ``FedAvg`` with an Adam-style server
an optimizer that applies bias-corrected first- and second-moment estimation to
the aggregated pseudo-gradient (Δ = avg_client_update − current_global_weights).

Server update rule (per round t)
---------------------------------
    Δ_t  = FedAvg(client_updates) − w_t          # pseudo-gradient
    m_t  = β₁ · m_{t-1} + (1 − β₁) · Δ_t        # 1st moment
    v_t  = β₂ · v_{t-1} + (1 − β₂) · Δ_t²       # 2nd moment
    m̂_t  = m_t / (1 − β₁ᵗ)                       # bias correction
    v̂_t  = v_t / (1 − β₂ᵗ)                       # bias correction
    w_{t+1} = w_t + η · m̂_t / (√v̂_t + ε)        # parameter update

Default hyperparameters match those used in the paper experiments:
    η = 0.1,  β₁ = 0.9,  β₂ = 0.999,  τ = 1e-3

The class also stores per-round per-client evaluation metrics so that
results can be inspected or exported after the simulation.

``_AdaptiveServerStrategy`` factors out the plumbing shared by every
stateful server-side optimizer evaluated in the paper's benchmark
(``FedCVRStrategy`` here, and ``FedAdagradStrategy`` / ``FedYogiStrategy``
in ``fedcvr.baselines``): bootstrapping w_0 from the first round's plain
FedAvg result (Algorithm 1 requires w_0 as input and zero-initializes only
the moment vectors, never the model itself), and per-client evaluation
metric logging. Subclasses only implement the per-round update rule.
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

        # On the very first round there is no previously-tracked global
        # model to diff against. Algorithm 1 requires w_0 (the actual
        # initial model) as input and zero-initialises only the moment
        # vectors - so we bootstrap w_0 from this round's plain FedAvg
        # result rather than from zeros, and start applying the optimiser
        # update from the following round.
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
                client_proxy.cid: {"loss": res.loss, **res.metrics}
                for client_proxy, res in results
            }

        return aggregated_loss, aggregated_metrics


class FedCVRStrategy(_AdaptiveServerStrategy):
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
            # eta == 0 disables the server-side Adam optimizer entirely:
            # the server aggregation step is a simple weighted average
            # (used to emulate the FedAvg / FedProx baselines, per the
            # paper's description of those methods).
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
