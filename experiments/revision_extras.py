"""Deterministic companion analyses for the revision (no training).

  1. Sensitivity of the fuzzy allocation to its membership functions,
     output universe and rule base (privacy axis).
  2. Firing strength of every rule for every site.
  3. DP-noise variance entering the aggregated pseudo-gradient under each
     allocation (heterogeneous-lot version of the aggregated-noise formula).
  4. Accounting-horizon table (T = 14 versus T = 100).
  5. Effect of H4's subsampling on its budget.
  6. Stability of the train-fold feature ranking at H1.

Usage:  python -m experiments.revision_extras
"""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "results_revision")

from dpfedadam.fuzzy_fairness import FuzzyFairnessController  # noqa: E402
from dpfedadam.rdp_accountant import RDPAccountant  # noqa: E402
from experiments.revision_runner import allocate  # noqa: E402

N = [211, 700, 642, 717]
SIGMA, T, E, DELTA, BASE_L = 2.5, 100, 1, 1e-5, 32


def eps(n, lot, sigma=SIGMA, rounds=T, epochs=E):
    return RDPAccountant(noise_multiplier=sigma, batch_size=lot, delta=DELTA).compute_epsilon(n, rounds, epochs)


def fuzzy_lots(**kw):
    base = [eps(n, BASE_L) for n in N]
    ctl = FuzzyFairnessController.for_federation(N, base, **kw)
    lots = [int(np.clip(round(BASE_L * ctl.recommend_multiplier(n, e)), 8, 128)) for n, e in zip(N, base)]
    return lots, [eps(n, l) for n, l in zip(N, lots)]


def sensitivity():
    rows = []

    def add(kind, name, **kw):
        try:
            lots, e = fuzzy_lots(**kw)
        except Exception as exc:  # a rule base that leaves a site uncovered
            rows.append(dict(kind=kind, variant=name, lots="n/a", worst_eps=np.nan, spread=np.nan, note=str(exc)[:60]))
            return
        rows.append(dict(kind=kind, variant=name, lots=json.dumps(lots), worst_eps=max(e), spread=max(e) / min(e), note=""))

    add("reference", "adopted controller")
    for name, bp in (("narrow overlap", (0.25, 0.45, 0.65, 0.65)), ("wide overlap", (0.05, 0.45, 0.45, 0.85)),
                     ("peak shifted left", (0.10, 0.35, 0.45, 0.65)), ("peak shifted right", (0.25, 0.55, 0.65, 0.85))):
        add("membership functions", name, breakpoints=bp)
    for mu in ((0.5, 2.0), (0.3, 2.5), (0.6, 1.8), (0.4, 3.0)):
        add("output universe", f"multiplier in [{mu[0]}, {mu[1]}]", multiplier_universe=mu)
    terms = ["decrease", "keep", "increase"]
    ref = FuzzyFairnessController().rule_table
    for key, out in ref.items():
        i = terms.index(out)
        for j in (i - 1, i + 1):
            if 0 <= j < 3:
                add("rule base", f"{key[0]} & {key[1]}: {out} -> {terms[j]}", rule_overrides={key: terms[j]})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "fuzzy_sensitivity.csv"), index=False)
    return df


def firing():
    import skfuzzy as fuzz
    base = [eps(n, BASE_L) for n in N]
    ctl = FuzzyFairnessController.for_federation(N, base)
    rows = []
    for lab, n, e in zip(("H1", "H2", "H3", "H4"), N, base):
        mn = {t: float(fuzz.interp_membership(ctl._n_train_var.universe, ctl._n_train_var[t].mf, n)) for t in ("small", "medium", "large")}
        me = {t: float(fuzz.interp_membership(ctl._epsilon_var.universe, ctl._epsilon_var[t].mf, e)) for t in ("low", "moderate", "high")}
        r = dict(site=lab, n_train=n, eps_base=e, multiplier=ctl.recommend_multiplier(n, e))
        r.update({f"mu_size_{k}": v for k, v in mn.items()})
        r.update({f"mu_eps_{k}": v for k, v in me.items()})
        for (a, b), o in ctl.rule_table.items():
            r[f"{a}&{b}->{o}"] = min(mn[a], me[b])
        rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "fuzzy_rule_firing.csv"), index=False)
    return df


def noise_variance():
    """Per-round DP-noise variance (per coordinate, in units of
    (eta_c * sigma * C)^2) in the aggregated pseudo-gradient:
    sum_k p_k^2 * S_k / Lbar_k^2, S_k = ceil(n_k / L_k) steps and
    Lbar_k = n_k / S_k the expected lot size used as divisor."""
    p = np.array(N) / sum(N)
    rows = []
    for arm in ("fixed", "fuzzy", "search", "proportional"):
        a = allocate(arm, N, SIGMA, T, E)
        S = np.array([math.ceil(n / l) for n, l in zip(N, a["lots"])])
        Lbar = np.array(N) / S
        v = S / Lbar ** 2
        rows.append(dict(arm=arm, lots=json.dumps(a["lots"]), steps=json.dumps(S.tolist()),
                         per_site_var=json.dumps([round(float(x), 6) for x in v]),
                         aggregated_var=float(np.sum(p ** 2 * v)), worst_eps=max(a["eps"]),
                         spread=max(a["eps"]) / min(a["eps"]), eps=json.dumps([round(x, 2) for x in a["eps"]])))
    df = pd.DataFrame(rows)
    df["aggregated_var_rel_fixed"] = df.aggregated_var / df.aggregated_var.iloc[0]
    f = np.array(json.loads(df.per_site_var.iloc[0]))
    df["per_site_var_rel_fixed"] = [json.dumps([round(float(x), 2) for x in np.array(json.loads(v)) / f]) for v in df.per_site_var]
    df.to_csv(os.path.join(OUT, "noise_variance_by_allocation.csv"), index=False)
    return df


def horizon():
    rows = [dict(sigma=s, epochs=e, eps_T14=eps(211, 32, s, 14, e), eps_T100=eps(211, 32, s, 100, e))
            for s, e in ((2.5, 1), (1.1, 5))]
    df = pd.DataFrame(rows)
    df["ratio"] = df.eps_T100 / df.eps_T14
    df.to_csv(os.path.join(OUT, "accounting_horizon.csv"), index=False)
    return df


def h4_full():
    n_full = int(round(70000 * 0.7))
    return dict(n_train_full=n_full, eps_fixed=eps(n_full, 32), eps_h1=eps(211, 32))


def feature_stability():
    from dpfedadam import data_utils as du
    sel = []
    for seed in range(20):
        meta = du.get_partition_metadata(os.path.join(ROOT, "data"), random_state=seed)
        sel.append(list(meta[0]["h1_selected_features"]))
    if not sel[0]:
        return None
    flat = pd.Series([f for s in sel for f in s]).value_counts()
    pair = [len(set(a) & set(b)) for i, a in enumerate(sel) for b in sel[i + 1:]]
    flat.rename("times_selected_of_20").to_csv(os.path.join(OUT, "h1_feature_selection_stability.csv"))
    return dict(distinct=len(flat), always=int((flat == 20).sum()), mean_pairwise_overlap=float(np.mean(pair)),
                counts=flat.to_dict())


N_GRADED = [70, 141, 140, 560, 322, 320, 239, 478]


def graded_federation():
    """Graded 8-client federation (sizes identical in every partition):
    allocation of each rule, rule firing, and a grid over the controller's
    output universe to see whether tuning it removes the spread."""
    import skfuzzy as fuzz
    rows = []
    for arm in ("fixed", "fuzzy", "search", "proportional"):
        a = allocate(arm, N_GRADED, SIGMA, T, E)
        rows.append(dict(kind="arm", variant=arm, lots=json.dumps(a["lots"]), worst_eps=max(a["eps"]),
                         spread=max(a["eps"]) / min(a["eps"]), eps=json.dumps([round(x, 2) for x in a["eps"]])))
    base = [eps(n, BASE_L) for n in N_GRADED]
    best = None
    for lo in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7):
        for hi in (1.5, 1.8, 2.2, 2.6, 3.0, 3.5, 4.0):
            ctl = FuzzyFairnessController.for_federation(N_GRADED, base, multiplier_universe=(lo, hi))
            lots = [max(1, min(n, int(np.clip(round(BASE_L * ctl.recommend_multiplier(n, e)), 8, 128))))
                    for n, e in zip(N_GRADED, base)]
            e2 = [eps(n, l) for n, l in zip(N_GRADED, lots)]
            r = dict(kind="output universe", variant=f"[{lo}, {hi}]", lots=json.dumps(lots), worst_eps=max(e2),
                     spread=max(e2) / min(e2), eps=json.dumps([round(x, 2) for x in e2]))
            rows.append(r)
            if best is None or r["spread"] < best["spread"]:
                best = r
    ctl = FuzzyFairnessController.for_federation(N_GRADED, base)
    fired = {}
    for n, e in zip(N_GRADED, base):
        mn = {t: float(fuzz.interp_membership(ctl._n_train_var.universe, ctl._n_train_var[t].mf, n)) for t in ("small", "medium", "large")}
        me = {t: float(fuzz.interp_membership(ctl._epsilon_var.universe, ctl._epsilon_var[t].mf, e)) for t in ("low", "moderate", "high")}
        for key in ctl.rule_table:
            w = min(mn[key[0]], me[key[1]])
            if w > 0.01:
                fired[f"{key[0]} & {key[1]}"] = max(fired.get(f"{key[0]} & {key[1]}", 0.0), w)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "graded_allocation.csv"), index=False)
    pd.Series(fired, name="max_firing_strength").to_csv(os.path.join(OUT, "graded_rule_firing.csv"))
    return df, best, fired


if __name__ == "__main__":
    pd.set_option("display.width", 250, "display.max_columns", 40, "display.max_colwidth", 60)
    print(sensitivity().to_string()); print()
    print(firing().round(3).T.to_string()); print()
    print(noise_variance().to_string()); print()
    print(horizon().to_string()); print()
    print(h4_full()); print()
    print(feature_stability())
    g, best, fired = graded_federation()
    print(g[g.kind == "arm"].to_string()); print("best output universe:", best); print(fired)
