"""
experiments/momentum_noise_check.py
====================================
Empirical check of the first-moment variance identity, and of what it
does NOT imply.

A stream of i.i.d. zero-mean pseudo-gradients (pure DP noise, no signal)
is passed through three server rules for T rounds:

  (a) stateless     w_{t+1} = w_t + eta * d_t
  (b) momentum only w_{t+1} = w_t + eta * m_t,  m_t = b1 m_{t-1} + (1-b1) d_t
  (c) full Adam     w_{t+1} = w_t + eta * m_hat_t / (sqrt(v_hat_t) + tau)

Measured over many coordinates and repetitions:

  * Var(m_t) / Var(d_t) per round, against the closed form
    (1-b1)/(1+b1) * (1 - b1^(2t));
  * Var(w_T - w_0), the noise that actually reaches the model.

The first quantity matches the closed form. The second shows that the
per-step attenuation does not carry over to the trajectory: every noise
draw enters the iterate with total weight (1-b1) * sum_k b1^k -> 1, so
momentum (b) accumulates essentially the same variance as the stateless
rule (a). Rule (c) is reported for completeness; its displacement is set
by the sqrt(v_hat) normalization, not by the injected noise scale.

Usage
-----
    python -m experiments.momentum_noise_check
"""
import os

import numpy as np
import pandas as pd

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


def simulate(beta_1, beta_2=0.9614, tau=1.28e-3, eta=0.168, T=100, dim=3585, reps=200, noise_sd=0.01, seed=0):
    rng = np.random.default_rng(seed)
    d = rng.normal(0.0, noise_sd, size=(reps, T, dim))
    m = np.zeros((reps, dim)); v = np.zeros((reps, dim))
    w_a = np.zeros((reps, dim)); w_b = np.zeros((reps, dim)); w_c = np.zeros((reps, dim))
    ratio = []
    for t in range(1, T + 1):
        dt = d[:, t - 1]
        m = beta_1 * m + (1 - beta_1) * dt
        v = beta_2 * v + (1 - beta_2) * dt ** 2
        ratio.append(m.var() / dt.var())
        w_a += eta * dt
        w_b += eta * m
        w_c += eta * (m / (1 - beta_1 ** t)) / (np.sqrt(v / (1 - beta_2 ** t)) + tau)
    predicted = [(1 - beta_1) / (1 + beta_1) * (1 - beta_1 ** (2 * t)) for t in range(1, T + 1)]
    return np.array(ratio), np.array(predicted), w_a.var(), w_b.var(), w_c.var()


def main():
    rows, curves = [], []
    for beta_1 in (0.5, 0.7817, 0.9):
        ratio, pred, va, vb, vc = simulate(beta_1)
        rows.append({"beta_1": beta_1,
                     "stationary_ratio_predicted": (1 - beta_1) / (1 + beta_1),
                     "stationary_ratio_observed": ratio[-20:].mean(),
                     "max_rel_error_over_rounds": float(np.max(np.abs(ratio - pred) / pred)),
                     "attenuation_factor": (1 + beta_1) / (1 - beta_1),
                     "trajectory_var_momentum_over_stateless": vb / va,
                     "trajectory_var_adam_over_stateless": vc / va})
        curves.append(pd.DataFrame({"beta_1": beta_1, "round": np.arange(1, len(ratio) + 1),
                                    "observed": ratio, "predicted": pred}))
    out = os.path.join(ROOT, "results_revision")
    os.makedirs(out, exist_ok=True)
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(out, "momentum_noise_check.csv"), index=False)
    pd.concat(curves).to_csv(os.path.join(out, "momentum_noise_check_curves.csv"), index=False)
    pd.set_option("display.width", 200)
    print(res.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
