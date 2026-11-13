"""
experiments/headroom_study.py
=============================
When does it pay for the larger clients to spend their privacy headroom?

Every client must take at least 12 lots per epoch, which caps its budget
at eps* = f(12) = 5.977 (sigma = 2.5, T = 100, E = 1, delta = 1e-5).
Clients above the cap under the base lot size (n <= 352) shrink to the
cap. A client below the cap may keep its base lot (frac = 0, "no harm"),
or move a fraction frac of the way to the largest lot the cap allows
(frac = 1 is exact equalization). The policies decide frac per federation:

  P0     always 0          (never spend headroom)
  P05    always 0.5
  P1     always 1          (always equalize)
  PT     crisp threshold   (spend all if constrained share >= X; X tuned on
                            the development federations, validation AUC)
  PF     fuzzy controller  (rules fixed in headroom_study_protocol.md
                            before any fuzzy run)

The protocol, including the success criteria, is in
experiments/headroom_study_protocol.md.

Usage
-----
    python -m experiments.revision_runner --suite headroom_study
    python -m experiments.revision_runner --suite headroom_study_fuzzy
    python -m experiments.headroom_study            # features + analysis
"""
from __future__ import annotations

import json
import math
import os
import sys
from typing import Dict, List

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "results_revision")

CAP_EPS = 5.9773          # = f(12 lots per epoch) = 5.97726..., rounded up
PARTITIONS = range(10)
TRAIN_SEEDS = range(3)
SPLIT_SEED = 2026

T3 = [1 / 3] * 3
FEDERATIONS: Dict[str, List[List[float]]] = {
    "F01": [[1], [1], [1], [1]],
    "F02": [[0.5, 0.5], [1], [1], [1]],
    "F03": [[1], [0.3, 0.7], [1], [1]],
    "F04": [[0.5, 0.5], [0.5, 0.5], [1], [1]],
    "F05": [[0.5, 0.5], [1], [0.5, 0.5], [1]],
    "F06": [[1], T3, [1], [1]],
    "F07": [[1], T3, T3, [1]],
    "F08": [[0.5, 0.5], [0.15, 0.15, 0.7], [1], [0.15, 0.15, 0.7]],
    "F09": [[1], [1], [0.25, 0.75], [0.25, 0.75]],
    "F10": [[1], [0.2] * 5, [1], [1]],
    "F11": [[0.5, 0.5], [1 / 3, 2 / 3], [1 / 3, 2 / 3], [1 / 3, 2 / 3]],
    "F12": [[1], [1 / 6] * 6, [1 / 6] * 6, [1]],
}


def split():
    """F01 (the four-site federation, already observed) goes to development;
    five more development federations are drawn at random with a fixed seed;
    the remaining six are the confirmation set."""
    rest = sorted(k for k in FEDERATIONS if k != "F01")
    rng = np.random.RandomState(SPLIT_SEED)
    dev = ["F01"] + sorted(rng.choice(rest, 5, replace=False).tolist())
    conf = [k for k in rest if k not in dev]
    return dev, conf


def client_sizes(ratios) -> List[int]:
    from dpfedadam.data_utils import load_and_preprocess_data
    from experiments.revision_runner import _shard_ratios
    train, _, _, _ = load_and_preprocess_data(os.path.join(ROOT, "data"), random_state=0, verbose=False)
    return [len(y) for _, y in _shard_ratios(train, ratios, 0)]


def allocation(n, frac):
    from experiments.revision_runner import allocate
    return allocate("frac", n, 2.5, 100, 1, CAP_EPS, frac)


def noise_var(n, lots) -> float:
    """Per-round DP-noise variance of the aggregated update, Eq. (7) units."""
    p = np.array(n) / sum(n)
    S = np.array([math.ceil(a / b) for a, b in zip(n, lots)])
    Lbar = np.array(n) / S
    return float(np.sum(p ** 2 * S / Lbar ** 2))


def features(n) -> Dict[str, float]:
    a0, a1 = allocation(n, 0.0), allocation(n, 1.0)
    base = a0["base_eps"]
    constrained = sum(k for k, e in zip(n, base) if e > CAP_EPS)
    return {"clients": len(n), "constrained_share": constrained / sum(n),
            "noise_reduction": 1.0 - noise_var(n, a1["lots"]) / noise_var(n, a0["lots"])}


# ---------------------------------------------------------------------------
# Policies
# ---------------------------------------------------------------------------

def fuzzy_frac(constrained_share: float, noise_reduction: float) -> float:
    """Mamdani controller; universes, terms and rules as in the protocol."""
    import skfuzzy as fuzz
    from skfuzzy import control as ctrl

    c = ctrl.Antecedent(np.linspace(0, 0.5, 101), "c")
    r = ctrl.Antecedent(np.linspace(0, 1, 101), "r")
    f = ctrl.Consequent(np.linspace(0, 1, 101), "f")
    c["low"] = fuzz.trimf(c.universe, [0, 0, 0.2])
    c["medium"] = fuzz.trimf(c.universe, [0.1, 0.25, 0.4])
    c["high"] = fuzz.trimf(c.universe, [0.3, 0.5, 0.5])
    r["small"] = fuzz.trimf(r.universe, [0, 0, 0.5])
    r["moderate"] = fuzz.trimf(r.universe, [0.3, 0.6, 0.9])
    r["large"] = fuzz.trimf(r.universe, [0.7, 1, 1])
    f["none"] = fuzz.trimf(f.universe, [0, 0, 0.5])
    f["half"] = fuzz.trimf(f.universe, [0.25, 0.5, 0.75])
    f["all"] = fuzz.trimf(f.universe, [0.5, 1, 1])
    rules = [
        ctrl.Rule(c["low"] & r["small"], f["none"]),
        ctrl.Rule(c["low"] & r["moderate"], f["none"]),
        ctrl.Rule(c["low"] & r["large"], f["half"]),
        ctrl.Rule(c["medium"] & r["small"], f["none"]),
        ctrl.Rule(c["medium"] & r["moderate"], f["half"]),
        ctrl.Rule(c["medium"] & r["large"], f["all"]),
        ctrl.Rule(c["high"] & r["small"], f["half"]),
        ctrl.Rule(c["high"] & r["moderate"], f["all"]),
        ctrl.Rule(c["high"] & r["large"], f["all"]),
    ]
    sim = ctrl.ControlSystemSimulation(ctrl.ControlSystem(rules))
    sim.input["c"] = float(np.clip(constrained_share, 0, 0.5))
    sim.input["r"] = float(np.clip(noise_reduction, 0, 1))
    sim.compute()
    return round(float(sim.output["f"]), 3)


def study_table() -> pd.DataFrame:
    dev, conf = split()
    rows = []
    for k, ratios in FEDERATIONS.items():
        n = client_sizes(ratios)
        ft = features(n)
        fz = fuzzy_frac(ft["constrained_share"], ft["noise_reduction"])
        row = {"federation": k, "set": "development" if k in dev else "confirmation", "sizes": json.dumps(n), **ft,
               "fuzzy_frac": fz}
        for name, fr in (("0", 0.0), ("0.5", 0.5), ("1", 1.0), ("fuzzy", fz)):
            a = allocation(n, fr)
            row[f"lots_{name}"] = json.dumps(a["lots"])
            row[f"eps_mean_{name}"] = float(np.mean(a["eps"]))
            row[f"rises_{name}"] = int(sum(e > b + 1e-9 for e, b in zip(a["eps"], a["base_eps"])))
        row["eps_worst"] = max(allocation(n, 1.0)["eps"])
        rows.append(row)
    return pd.DataFrame(rows)


def study_configs(arms=("0", "0.5", "1")) -> List[Dict]:
    """Run configurations; the fuzzy arm reads its fraction from the rules."""
    out = []
    table = None
    for k, ratios in FEDERATIONS.items():
        for arm in arms:
            if arm == "fuzzy":
                if table is None:
                    table = study_table().set_index("federation")
                frac = float(table.loc[k, "fuzzy_frac"])
            else:
                frac = float(arm)
            for p in PARTITIONS:
                for t in TRAIN_SEEDS:
                    out.append(dict(suite="headroom_study" + ("_fuzzy" if arm == "fuzzy" else ""),
                                    partition_seed=p, train_seed=t, allocation="frac", target_eps=CAP_EPS,
                                    headroom_frac=frac, shard_ratios=ratios, tag=f"{k}|{arm}",
                                    hp="fedadam_default"))
    return out


# ---------------------------------------------------------------------------
# Analysis (pre-specified in the protocol)
# ---------------------------------------------------------------------------

def _blocks(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    df = df.copy()
    df[["federation", "arm"]] = df["tag"].str.split("|", expand=True)
    return df.groupby(["federation", "partition_seed", "arm"])[metric].mean().unstack("arm")


def tune_threshold(dev_blocks: pd.DataFrame, table: pd.DataFrame) -> float:
    """X minimizing mean development regret on validation AUC; ties go to
    the larger X (less spending)."""
    fed_means = dev_blocks.groupby(level="federation")[["0", "0.5", "1"]].mean()
    share = table.set_index("federation")["constrained_share"].reindex(fed_means.index)
    vals = sorted(share.values)
    cands = [0.0] + [(a + b) / 2 for a, b in zip(vals, vals[1:])] + [1.01]
    best = None
    for x in cands:
        pick = np.where(share.values >= x, "1", "0")
        regret = float(np.mean([fed_means.loc[f, ["0", "0.5", "1"]].max() - fed_means.loc[f, a]
                                for f, a in zip(fed_means.index, pick)]))
        if best is None or regret < best[1] - 1e-12 or (abs(regret - best[1]) <= 1e-12 and x > best[0]):
            best = (x, regret)
    return best[0]


def analyse() -> None:
    from scipy import stats

    table = study_table()
    pd.set_option("display.width", 250, "display.max_columns", 40, "display.max_colwidth", 60)
    print(table[["federation", "set", "clients", "constrained_share", "noise_reduction", "fuzzy_frac",
                 "eps_mean_0", "eps_mean_1", "eps_mean_fuzzy", "rises_1", "rises_fuzzy"]].round(3).to_string(index=False))
    path = os.path.join(OUT, "headroom_study.csv")
    if not os.path.exists(path):
        return
    runs = pd.read_csv(path)
    fz = os.path.join(OUT, "headroom_study_fuzzy.csv")
    if os.path.exists(fz):
        runs = pd.concat([runs, pd.read_csv(fz)], ignore_index=True)
    dev, conf = split()
    val = _blocks(runs, "val_macro_auc_sites")
    test = _blocks(runs, "macro_auc_sites")
    x = tune_threshold(val.loc[dev], table)
    share = table.set_index("federation")["constrained_share"]
    print(f"\nCrisp threshold tuned on development (validation AUC): spend all if constrained share >= {x:.3f}")

    t = test.loc[conf].copy()
    feds = t.index.get_level_values("federation")
    t["PT"] = [t.loc[(f, p), "1" if share[f] >= x else "0"] for f, p in t.index]
    pol = {"P0": "0", "P05": "0.5", "P1": "1", "PT": "PT"}
    if "fuzzy" in t:
        pol["PF"] = "fuzzy"
    tb = table.set_index("federation")
    eps_of = {"P0": "eps_mean_0", "P05": "eps_mean_0.5", "P1": "eps_mean_1", "PF": "eps_mean_fuzzy"}
    summary = []
    fed_best = t[["0", "0.5", "1"]].groupby(level="federation").mean().max(axis=1)
    for name, col in pol.items():
        fed_mean = t[col].groupby(level="federation").mean()
        eps = [tb.loc[f, eps_of[name]] if name != "PT" else tb.loc[f, "eps_mean_1" if share[f] >= x else "eps_mean_0"]
               for f in fed_mean.index]
        summary.append({"policy": name, "test_auc": t[col].mean(), "regret": float((fed_best - fed_mean).mean()),
                        "mean_client_eps": float(np.mean(eps))})
    print("\nConfirmation federations (test AUC; 6 federations x 10 partitions):")
    print(pd.DataFrame(summary).round(4).to_string(index=False))
    print("\nPer federation (test AUC):")
    print(t.groupby(level="federation").mean().round(4).to_string())

    def paired(a, b, label):
        d = (t[a] - t[b]).values
        n = len(d)
        se = d.std(ddof=1) / np.sqrt(n)
        lo1 = d.mean() - stats.t.ppf(0.95, n - 1) * se
        lo2, hi2 = d.mean() - stats.t.ppf(0.975, n - 1) * se, d.mean() + stats.t.ppf(0.975, n - 1) * se
        w = stats.wilcoxon(d, alternative="greater").pvalue
        print(f"  {label:28s} diff {d.mean():+.4f}  95% CI [{lo2:+.4f}, {hi2:+.4f}]  one-sided lower {lo1:+.4f}"
              f"  Wilcoxon one-sided p {w:.4f}  wins {int((d > 0).sum())}/{n}")
        return lo1, w

    if "fuzzy" in t:
        print("\nPre-specified criteria for the fuzzy policy:")
        s1, _ = paired("fuzzy", "1", "S1 fuzzy - always equalize")
        _, s3 = paired("fuzzy", "0", "S3 fuzzy - never spend")
        e = {r["policy"]: r["mean_client_eps"] for r in summary}
        print(f"  S1 non-inferior to P1 (lower bound > -0.005): {s1 > -0.005}")
        print(f"  S2 spends less privacy than P1: {e['PF'] < e['P1']} ({e['PF']:.3f} vs {e['P1']:.3f})")
        print(f"  S3 better than P0 (one-sided Wilcoxon p < 0.05): {s3 < 0.05}")
        print("\nSecondary: fuzzy versus the crisp threshold rule")
        paired("fuzzy", "PT", "fuzzy - crisp threshold")


if __name__ == "__main__":
    analyse()
