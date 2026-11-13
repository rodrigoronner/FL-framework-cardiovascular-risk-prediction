"""
experiments/revision_analysis.py
=================================
Statistical analysis of the runs produced by ``revision_runner``.

Unit of analysis. Each arm is run on the same 20 partitions x 3 training
seeds. Runs that share a partition are not independent, so the three
training seeds are averaged first and every test is PAIRED over
partitions (n = 20 unless stated).

Reported per comparison: mean paired difference, 95% t-interval, paired
t-test, Wilcoxon signed-rank test, Cohen's d_z, and Holm-adjusted
p-values within each family. Multi-arm families additionally get a
Friedman test and Nemenyi critical difference on mean ranks.

Also produced: membership-inference audit (loss-threshold attack),
subgroup recall (sex, age band), and mean ROC curves.

Usage
-----
    python -m experiments.revision_analysis
"""

from __future__ import annotations

import glob
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
OUT = os.path.join(ROOT, "results_revision")
SITES = ["H1", "H2", "H3", "H4"]


def load(suite: str) -> pd.DataFrame:
    df = pd.read_csv(os.path.join(OUT, f"{suite}.csv"))
    for col in ("target_eps", "hp", "tag"):
        if col in df:
            df[col] = df[col].fillna("")
    return df


def by_partition(df: pd.DataFrame, arm_col: str, value: str) -> pd.DataFrame:
    """partition x arm table, training seeds averaged."""
    return df.groupby(["partition_seed", arm_col])[value].mean().unstack(arm_col)


def holm(pvals):
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (len(p) - rank) * p[idx])
        adj[idx] = min(1.0, running)
    return adj


def paired(table: pd.DataFrame, a: str, b: str) -> dict:
    d = (table[a] - table[b]).dropna().values
    n = len(d)
    se = d.std(ddof=1) / np.sqrt(n)
    half = stats.t.ppf(0.975, n - 1) * se
    try:
        w_p = stats.wilcoxon(d).pvalue
    except ValueError:
        w_p = np.nan
    return {"comparison": f"{a} - {b}", "n": n, "mean_diff": d.mean(),
            "ci_low": d.mean() - half, "ci_high": d.mean() + half,
            "t_p": stats.ttest_1samp(d, 0.0).pvalue, "wilcoxon_p": w_p,
            "cohen_dz": d.mean() / d.std(ddof=1) if d.std(ddof=1) > 0 else np.nan,
            "wins": int((d > 0).sum())}


def family(table: pd.DataFrame, pairs, title: str) -> pd.DataFrame:
    res = pd.DataFrame([paired(table, a, b) for a, b in pairs])
    res["t_p_holm"] = holm(res["t_p"])
    res["wilcoxon_p_holm"] = holm(res["wilcoxon_p"])
    res.insert(0, "family", title)
    return res


def friedman_nemenyi(table: pd.DataFrame, title: str) -> dict:
    t = table.dropna()
    k, n = t.shape[1], t.shape[0]
    stat, p = stats.friedmanchisquare(*[t[c].values for c in t.columns])
    ranks = t.rank(axis=1, ascending=False).mean()
    q = stats.studentized_range.ppf(0.95, k, np.inf) / np.sqrt(2)
    cd = q * np.sqrt(k * (k + 1) / (6.0 * n))
    return {"family": title, "n_blocks": n, "k": k, "friedman_chi2": stat, "friedman_p": p,
            "nemenyi_cd_0.05": cd, "mean_ranks": json.dumps(ranks.round(3).to_dict())}


def describe(df: pd.DataFrame, arm_col: str, cols) -> pd.DataFrame:
    """Mean and SD over partitions (training seeds averaged first)."""
    out = {}
    for c in cols:
        t = by_partition(df, arm_col, c)
        out[c + "_mean"] = t.mean()
        out[c + "_sd"] = t.std(ddof=1)
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------
# Prediction-level analyses
# ---------------------------------------------------------------------------

def _npz(suite: str, run_id: str):
    return np.load(os.path.join(OUT, "preds", suite, run_id + ".npz"))


def mia_audit(df: pd.DataFrame, suite: str) -> pd.DataFrame:
    """Loss-threshold membership inference (Yeom et al.): members are a
    site's training records, non-members its test records, the attack
    score is the negative per-record cross-entropy under the final model.
    Returns attack AUC and best balanced advantage max(TPR - FPR)."""
    from sklearn.metrics import roc_auc_score, roc_curve

    rows = []
    for _, r in df.iterrows():
        z = _npz(suite, r["run_id"])
        for s in SITES:
            if f"train_p_{s}" not in z:
                continue
            def nll(p, y):
                p = np.clip(p, 1e-7, 1 - 1e-7)
                return -(y * np.log(p) + (1 - y) * np.log(1 - p))
            score = -np.concatenate([nll(z[f"train_p_{s}"], z[f"train_y_{s}"]),
                                     nll(z[f"test_p_{s}"], z[f"test_y_{s}"])])
            member = np.concatenate([np.ones(len(z[f"train_y_{s}"])), np.zeros(len(z[f"test_y_{s}"]))])
            fpr, tpr, _ = roc_curve(member, score)
            rows.append({"run_id": r["run_id"], "site": s, "attack_auc": roc_auc_score(member, score),
                         "advantage": float(np.max(tpr - fpr))})
    return pd.DataFrame(rows)


def subgroup_recall(df: pd.DataFrame, suite: str) -> pd.DataFrame:
    """Recall at the run's global decision threshold by sex and age band,
    pooled over the four test folds."""
    rows = []
    for _, r in df.iterrows():
        z = _npz(suite, r["run_id"])
        p = np.concatenate([z[f"test_p_{s}"] for s in SITES])
        y = np.concatenate([z[f"test_y_{s}"] for s in SITES])
        agesex = np.concatenate([z[f"test_agesex_{s}"] for s in SITES])
        age = 18.0 + 82.0 * agesex[:, 0]
        pred = p >= r["pooled_threshold"]
        groups = {"male": agesex[:, 1] > 0.5, "female": agesex[:, 1] <= 0.5,
                  "age<55": age < 55, "age>=55": age >= 55}
        for g, mask in groups.items():
            pos = mask & (y == 1)
            rows.append({"run_id": r["run_id"], "group": g, "n_pos": int(pos.sum()),
                         "recall": float(pred[pos].mean()) if pos.sum() else np.nan})
    return pd.DataFrame(rows)


def mean_roc(df: pd.DataFrame, suite: str, grid=np.linspace(0, 1, 101)) -> dict:
    from sklearn.metrics import roc_curve

    curves = []
    for _, r in df.iterrows():
        z = _npz(suite, r["run_id"])
        p = np.concatenate([z[f"test_p_{s}"] for s in SITES])
        y = np.concatenate([z[f"test_y_{s}"] for s in SITES])
        fpr, tpr, _ = roc_curve(y, p)
        curves.append(np.interp(grid, fpr, tpr))
    c = np.array(curves)
    return {"fpr": grid, "tpr_mean": c.mean(0), "tpr_sd": c.std(0, ddof=1)}


# ---------------------------------------------------------------------------

def main() -> None:
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    pd.set_option("display.float_format", lambda v: f"{v:.4f}")
    tests, omnibus = [], []
    auc_cols = ["macro_auc"] + [f"auc_{s}" for s in SITES]

    # ---- (1) allocation arms under DP ------------------------------------
    main_df = load("main_default")
    print("=" * 100, "\n(1) ALLOCATION ARMS, sigma=2.5, E=1, T=100, DP-FedAdam (literature-default server hyperparameters)\n", "=" * 100)
    first = main_df[(main_df.partition_seed == 0) & (main_df.train_seed == 0)]
    print(first[["allocation", "lots", "eps", "eps_spread", "eps_worst"]].to_string(index=False))
    print("\nmax |audited eps - eps reported by the Opacus accountant that trained|:",
          main_df["eps_max_abs_diff"].max())
    arms = ["fixed", "fuzzy", "search", "proportional"]
    print(describe(main_df, "allocation", auc_cols + ["macro_f1_score", "eps_worst", "eps_spread"]).loc[arms].T)
    t = by_partition(main_df, "allocation", "macro_auc")
    fam = family(t, [("fuzzy", "fixed"), ("search", "fixed"), ("proportional", "fixed"),
                     ("fuzzy", "search"), ("fuzzy", "proportional"), ("search", "proportional")],
                 "allocation / macro AUC")
    tests.append(fam); print("\n", fam.drop(columns="family").to_string(index=False))
    omnibus.append(friedman_nemenyi(t[arms], "allocation / macro AUC"))
    for s in SITES:
        tests.append(family(by_partition(main_df, "allocation", f"auc_{s}"),
                            [("fuzzy", "fixed"), ("search", "fixed"), ("fuzzy", "search")],
                            f"allocation / AUC {s}"))
    print("\nPer-site paired differences:")
    print(pd.concat(tests[1:])[["family", "comparison", "mean_diff", "ci_low", "ci_high", "t_p_holm"]].to_string(index=False))
    main_df.groupby(["partition_seed", "train_seed", "allocation"])[auc_cols].mean().unstack().to_csv(
        os.path.join(OUT, "main_default_individual_runs_wide.csv"))

    # ---- (1b) sensitivity: configuration tuned for the first submission ----
    tuned_df = load("main")
    print("\n", "=" * 100, "\n(1b) SENSITIVITY: same arms, server hyperparameters of the first submission\n", "=" * 100)
    print(describe(tuned_df, "allocation", auc_cols).loc[arms].T)
    fam = family(by_partition(tuned_df, "allocation", "macro_auc"),
                 [("fuzzy", "fixed"), ("search", "fixed"), ("proportional", "fixed"), ("fuzzy", "search")],
                 "allocation (first-submission hyperparameters) / macro AUC")
    tests.append(fam); print(fam.drop(columns="family").to_string(index=False))

    # ---- (2) server optimizers under DP ----------------------------------
    if os.path.exists(os.path.join(OUT, "dp_optimizers.csv")):
        dpo = load("dp_optimizers")
        dpo["arm"] = dpo.aggregator
        tuned = tuned_df[tuned_df.allocation == "fixed"].assign(arm="fedadam(first-submission hp)")
        chk = dpo[dpo.arm == "fedadam"].merge(main_df[main_df.allocation == "fixed"], on=["partition_seed", "train_seed"])
        print("consistency: |AUC(dp_optimizers fedadam) - AUC(main_default fixed)| max =",
              (chk.macro_auc_x - chk.macro_auc_y).abs().max())
        both = pd.concat([dpo, tuned])
        print("\n", "=" * 100, "\n(2) SERVER OPTIMIZERS UNDER DP (fixed L=32, sigma=2.5)\n", "=" * 100)
        print(describe(both, "arm", auc_cols).T)
        t = by_partition(both, "arm", "macro_auc")
        fam = family(t, [("fedadam", a) for a in ("fedavg", "fedprox", "fedadagrad", "fedyogi", "fedadam(first-submission hp)")],
                     "DP optimizers / macro AUC")
        tests.append(fam); print(fam.drop(columns="family").to_string(index=False))
        omnibus.append(friedman_nemenyi(t[["fedavg", "fedprox", "fedadagrad", "fedyogi", "fedadam"]], "DP optimizers / macro AUC"))

    # ---- (3) no-DP references --------------------------------------------
    if os.path.exists(os.path.join(OUT, "nodp.csv")):
        nd = load("nodp")
        nd["arm"] = np.where(nd.scope == "federated", nd.aggregator, nd.scope)
        nd["arm"] = nd["arm"].replace({"fedadam": "fedadam(first-submission hp)"})
        nd = pd.concat([nd, load("nodp_default").assign(arm="fedadam")])
        nd["src"] = np.where(nd.suite == "nodp_default", "nodp_default", "nodp")
        print("\n", "=" * 100, "\n(3) NO-DP REFERENCES (E=5): federated optimizers, pooled-centralized, local-only\n", "=" * 100)
        print(describe(nd, "arm", auc_cols).T)
        t = by_partition(nd, "arm", "macro_auc")
        fam = family(t, [("fedadam", a) for a in ("fedavg", "fedprox", "fedadagrad", "fedyogi", "pooled", "local")],
                     "no-DP / macro AUC")
        tests.append(fam); print(fam.drop(columns="family").to_string(index=False))
        omnibus.append(friedman_nemenyi(t[["fedavg", "fedprox", "fedadagrad", "fedyogi", "fedadam"]], "no-DP optimizers / macro AUC"))
        for s in SITES:
            tests.append(family(by_partition(nd, "arm", f"auc_{s}"), [("fedadam", "local"), ("fedadam", "pooled")],
                                f"no-DP / AUC {s}"))
        print(pd.concat(tests[-4:])[["family", "comparison", "mean_diff", "ci_low", "ci_high", "t_p_holm"]].to_string(index=False))

    # ---- (4) stronger privacy targets ------------------------------------
    if os.path.exists(os.path.join(OUT, "eps_targets.csv")):
        et = load("eps_targets")
        print("\n", "=" * 100, "\n(4) PRIVACY TARGETS (sigma calibrated on the fixed allocation's worst case)\n", "=" * 100)
        for tag, g in et.groupby("tag"):
            d = describe(g, "allocation", ["macro_auc", "eps_worst", "eps_spread"])
            d.insert(0, "sigma", g["sigma"].iloc[0])
            print(tag); print(d)
            tests.append(family(by_partition(g, "allocation", "macro_auc"),
                                [("fuzzy", "fixed"), ("search", "fixed"), ("fuzzy", "search")], f"{tag} / macro AUC"))
        print(pd.concat(tests[-3:])[["family", "comparison", "mean_diff", "ci_low", "ci_high", "t_p_holm"]].to_string(index=False))

    # ---- (5) scalability ---------------------------------------------------
    if os.path.exists(os.path.join(OUT, "scalability.csv")):
        sc = load("scalability")
        print("\n", "=" * 100, "\n(5) SCALABILITY: each site split into k clients\n", "=" * 100)
        for k, g in sc.groupby("shards"):
            print(f"{4 * k} clients"); print(describe(g, "allocation", ["macro_auc_sites", "eps_worst", "eps_spread"]))
            tests.append(family(by_partition(g, "allocation", "macro_auc_sites"),
                                [("fuzzy", "fixed"), ("search", "fixed"), ("fuzzy", "search")], f"{4 * k} clients / macro AUC"))
        print(pd.concat(tests[-2:])[["family", "comparison", "mean_diff", "ci_low", "ci_high", "t_p_holm"]].to_string(index=False))

    # ---- (5b) graded federation ---------------------------------------------
    if os.path.exists(os.path.join(OUT, "graded.csv")):
        gr = load("graded")
        print("\n", "=" * 100, "\n(5b) GRADED FEDERATION: 8 clients of graded size\n", "=" * 100)
        print(describe(gr, "allocation", ["macro_auc_sites", "macro_auc", "eps_worst", "eps_spread"]))
        tg = by_partition(gr, "allocation", "macro_auc_sites")
        tests.append(family(tg, [("fuzzy", "fixed"), ("search", "fixed"), ("proportional", "fixed"),
                                 ("fuzzy", "search"), ("fuzzy", "proportional"), ("search", "proportional")],
                            "graded 8 clients / macro AUC"))
        print(tests[-1][["comparison", "mean_diff", "ci_low", "ci_high", "wins", "t_p_holm", "wilcoxon_p_holm"]].to_string(index=False))
        omnibus.append(friedman_nemenyi(tg, "graded 8 clients / macro AUC"))

    # ---- (5c) headroom: no-harm and half-headroom rules ---------------------
    #      Same partitions and training seeds as main_default / graded, and
    #      training is deterministic given (partition, seed, lots), so the
    #      new arms pair with the arms already run.
    for new, base_suite, fam, col in (("headroom", "main_default", "headroom 4 sites", "macro_auc"),
                                      ("graded_headroom", "graded", "headroom graded", "macro_auc_sites")):
        if not os.path.exists(os.path.join(OUT, f"{new}.csv")):
            continue
        hd = pd.concat([load(base_suite), load(new)], ignore_index=True)
        print("\n", "=" * 100, f"\n(5c) {fam.upper()}: no-harm / half / exact / fuzzy / fixed\n", "=" * 100)
        hd["eps_mean"] = hd["eps"].apply(lambda s: float(np.mean(json.loads(s))))
        print(describe(hd, "allocation", [col, "eps_worst", "eps_mean", "eps_spread"]).round(4))
        th = by_partition(hd, "allocation", col)
        tests.append(family(th, [("noharm", "fixed"), ("half", "fixed"), ("search", "noharm"), ("half", "noharm"),
                                 ("search", "half"), ("fuzzy", "noharm")], f"{fam} / macro AUC"))
        print(tests[-1][["comparison", "mean_diff", "ci_low", "ci_high", "wins", "t_p_holm", "wilcoxon_p_holm"]].to_string(index=False))
        sites = sorted(c[4:] for c in hd.columns if c.startswith("auc_H") and hd[c].notna().all())
        print(hd.groupby("allocation")[[f"auc_{s}" for s in sites]].mean().round(3).T.to_string())
        omnibus.append(friedman_nemenyi(th[["fixed", "noharm", "half", "search", "fuzzy"]], f"{fam} / macro AUC"))

    # ---- (5d) no-harm at a second cap (the size-proportional worst case) ----
    if os.path.exists(os.path.join(OUT, "noharm_targets.csv")):
        nt = load("noharm_targets")
        for tag, base_suite, fam, col in (("cap_4.72", "main_default", "second cap 4 sites", "macro_auc"),
                                          ("cap_7.08", "graded", "second cap graded", "macro_auc_sites")):
            b = load(base_suite)
            d = pd.concat([b[b.allocation.isin(["fixed", "proportional"])],
                           nt[nt.tag == tag].assign(allocation="noharm")], ignore_index=True)
            d["eps_mean"] = d["eps"].apply(lambda s: float(np.mean(json.loads(s))))
            print("\n", "=" * 100, f"\n(5d) {fam.upper()}: no-harm vs size-proportional\n", "=" * 100)
            print(describe(d, "allocation", [col, "eps_worst", "eps_mean", "eps_spread"]).round(4))
            tests.append(family(by_partition(d, "allocation", col),
                                [("noharm", "fixed"), ("proportional", "fixed"), ("noharm", "proportional")],
                                f"{fam} / macro AUC"))
            print(tests[-1][["comparison", "mean_diff", "ci_low", "ci_high", "wins", "t_p_holm", "wilcoxon_p_holm"]].to_string(index=False))
            sites = sorted(c[4:] for c in d.columns if c.startswith("auc_H") and d[c].notna().all())
            print(d.groupby("allocation")[[f"auc_{s}" for s in sites]].mean().round(3).T.to_string())

    # ---- (6) membership inference, subgroups ------------------------------
    print("\n", "=" * 100, "\n(6) MEMBERSHIP-INFERENCE AUDIT (loss threshold) AND SUBGROUP RECALL\n", "=" * 100)
    mia_parts = []
    for arm in ("fixed", "fuzzy", "search"):
        m = mia_audit(main_df[main_df.allocation == arm], "main_default").assign(arm=f"DP {arm}")
        mia_parts.append(m)
    if os.path.exists(os.path.join(OUT, "nodp.csv")):
        mia_parts.append(mia_audit(nd[nd.arm == "fedadam"], "nodp_default").assign(arm="no-DP fedadam"))
        mia_parts.append(mia_audit(nd[nd.arm == "local"], "nodp").assign(arm="no-DP local-only"))
    mia = pd.concat(mia_parts)
    mia.to_csv(os.path.join(OUT, "mia_audit.csv"), index=False)
    print(mia.groupby(["arm", "site"])[["attack_auc", "advantage"]].agg(["mean", "std"]).round(4))
    sub_parts = [subgroup_recall(main_df[main_df.allocation == a], "main_default").assign(arm=f"DP {a}") for a in ("fixed", "fuzzy", "search")]
    if os.path.exists(os.path.join(OUT, "nodp.csv")):
        sub_parts.append(subgroup_recall(nd[nd.arm == "fedadam"], "nodp_default").assign(arm="no-DP fedadam"))
    sub = pd.concat(sub_parts)
    sub.to_csv(os.path.join(OUT, "subgroup_recall.csv"), index=False)
    print(sub.groupby(["arm", "group"])["recall"].agg(["mean", "std"]).unstack("group").round(3))

    pd.concat(tests).to_csv(os.path.join(OUT, "statistical_tests.csv"), index=False)
    pd.DataFrame(omnibus).to_csv(os.path.join(OUT, "friedman_nemenyi.csv"), index=False)
    print("\nFriedman / Nemenyi:"); print(pd.DataFrame(omnibus).to_string(index=False))


if __name__ == "__main__":
    main()
