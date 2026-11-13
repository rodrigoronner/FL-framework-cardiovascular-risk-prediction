"""Build every figure and numeric table of the revised manuscript from the
CSV files in results_revision/ (nothing is typed by hand).

Usage:  python -m experiments.revision_report --out <manuscript/R1>
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
RES = os.path.join(ROOT, "results_revision")
SITES = ["H1", "H2", "H3", "H4"]
ARMS = ["fixed", "fuzzy", "search", "proportional"]
ARM_LABEL = {"fixed": "Fixed ($L{=}32$)", "fuzzy": "Fuzzy controller",
             "search": "Exact equalization", "proportional": "Size-proportional",
             "noharm": "No harm (cap only)", "half": "Half headroom"}
ARM_PLOT = {"fixed": "Fixed", "fuzzy": "Fuzzy", "search": "Exact equalization", "proportional": "Size-proportional"}
COL = {"fixed": "#D55E00", "fuzzy": "#0072B2", "search": "#009E73", "proportional": "#AA4499", "nodp": "#333333"}
MARK = {"fixed": "s", "fuzzy": "o", "search": "^", "proportional": "D"}
LS = {"fixed": "--", "fuzzy": "-", "search": "-.", "proportional": ":"}
OPT_LABEL = {"fedavg": "FedAvg", "fedprox": "FedProx", "fedadagrad": "FedAdagrad", "fedyogi": "FedYogi",
             "fedadam": "FedAdam"}


def load(name):
    return pd.read_csv(os.path.join(RES, f"{name}.csv"))


def part(df, by, col):
    """Mean over training seeds per partition -> table partition x arm."""
    return df.groupby([by, "partition_seed"])[col].mean().unstack(0)


def ms(x, d=3):
    return f"${np.mean(x):.{d}f}\\pm{np.std(x, ddof=1):.{d}f}$"


def pfmt(p):
    return "$<0.001$" if p < 0.001 else f"${p:.3f}$"


def sgn(x, d=3):
    return f"${x:+.{d}f}$"


def write(out, name, text):
    with open(os.path.join(out, "tables", name + ".tex"), "w") as fh:
        fh.write(text + "\n")


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

def tables(out):
    tests = load("statistical_tests")
    fr = load("friedman_nemenyi")
    main = load("main_default")
    nums = {}

    def test(fam, comp):
        return tests[(tests.family == fam) & (tests.comparison == comp)].iloc[0]

    # --- privacy audit by allocation --------------------------------------
    nv = load("noise_variance_by_allocation").set_index("arm")
    n_train = json.loads(main.n_train.iloc[0])
    rows = []
    for i, s in enumerate(SITES):
        cells = [s, f"{n_train[i]}"]
        for a in ARMS:
            L = json.loads(nv.lots[a])[i]
            S = json.loads(nv.steps[a])[i]
            e = json.loads(nv.eps[a])[i]
            cells += [str(L), str(S), f"${e:.2f}$"]
        rows.append(" & ".join(cells) + r" \\")
    foot = [r"\midrule", r"\multicolumn{2}{l}{Worst-case $\varepsilon$} & " + " & ".join(
        rf"\multicolumn{{3}}{{r}}{{${nv.worst_eps[a]:.2f}$}}" for a in ARMS) + r" \\",
        r"\multicolumn{2}{l}{Spread $\varepsilon_{\max}/\varepsilon_{\min}$} & " + " & ".join(
            rf"\multicolumn{{3}}{{r}}{{${nv.spread[a]:.2f}\times$}}" for a in ARMS) + r" \\",
        r"\multicolumn{2}{l}{Aggregated DP-noise variance (Eq.~\ref{eq:noise_var}), relative to fixed} & " + " & ".join(
            rf"\multicolumn{{3}}{{r}}{{${nv.aggregated_var_rel_fixed[a]:.2f}$}}" for a in ARMS) + r" \\"]
    write(out, "audit_rows", "\n".join(rows + foot))

    # --- allocation arms: utility ------------------------------------------
    rows = []
    for a in ARMS:
        d = main[main.allocation == a]
        g = d.groupby("partition_seed").mean(numeric_only=True)
        cells = [ARM_LABEL[a], f"${d.eps_worst.iloc[0]:.2f}$", f"${d.eps_spread.iloc[0]:.2f}\\times$",
                 ms(g.macro_auc), ms(g.macro_f1_score), ms(g.macro_recall)] + [ms(g[f"auc_{s}"]) for s in SITES]
        rows.append(" & ".join(cells) + r" \\")
        nums[a] = g.macro_auc.mean()
    hd = load("headroom")
    rows.append(r"\midrule")
    rows.append(r"\multicolumn{10}{@{}l}{\textit{Cap-based rules at the same worst case (Section~\ref{sec:allocation})}} \\")
    for a in ("noharm", "half"):
        d = hd[hd.allocation == a]
        g = d.groupby("partition_seed").mean(numeric_only=True)
        cells = [ARM_LABEL[a], f"${d.eps_worst.iloc[0]:.2f}$", f"${d.eps_spread.iloc[0]:.2f}\\times$",
                 ms(g.macro_auc), ms(g.macro_f1_score), ms(g.macro_recall)] + [ms(g[f"auc_{s}"]) for s in SITES]
        rows.append(" & ".join(cells) + r" \\")
    write(out, "alloc_rows", "\n".join(rows))

    def paired_rows(fam, comps):
        r = []
        for c, lab in comps:
            t = test(fam, c)
            r.append(f"{lab} & {sgn(t.mean_diff, 4)} & $[{t.ci_low:+.4f},\\,{t.ci_high:+.4f}]$ & ${t.cohen_dz:+.2f}$ & "
                     f"{int(t.wins)}/{int(t.n)} & {pfmt(t.t_p_holm)} & {pfmt(t.wilcoxon_p_holm)} \\\\")
        return r
    comps = [("fuzzy - fixed", "Fuzzy $-$ fixed"), ("search - fixed", "Exact equalization $-$ fixed"),
             ("proportional - fixed", "Size-proportional $-$ fixed"), ("fuzzy - search", "Fuzzy $-$ exact equalization"),
             ("fuzzy - proportional", "Fuzzy $-$ size-proportional"),
             ("search - proportional", "Exact equalization $-$ size-proportional")]
    rows = [r"\multicolumn{7}{l}{\textit{Primary configuration (literature-default server hyperparameters)}} \\"]
    rows += paired_rows("allocation / macro AUC", comps)
    rows += [r"\midrule", r"\multicolumn{7}{l}{\textit{Cap-based rules (no harm, half headroom), same partitions and seeds}} \\"]
    rows += paired_rows("headroom 4 sites / macro AUC",
                        [("noharm - fixed", "No harm $-$ fixed"), ("half - fixed", "Half headroom $-$ fixed"),
                         ("search - noharm", "Exact equalization $-$ no harm"),
                         ("half - noharm", "Half headroom $-$ no harm"),
                         ("fuzzy - noharm", "Fuzzy $-$ no harm"),
                         ("search - half", "Exact equalization $-$ half headroom")])
    rows += [r"\midrule", r"\multicolumn{7}{l}{\textit{Sensitivity: server hyperparameters of the first submission}} \\"]
    rows += paired_rows("allocation (first-submission hyperparameters) / macro AUC", comps[:4])
    write(out, "paired_rows", "\n".join(rows))

    rows = []
    for s in SITES:
        cells = [s]
        for c in ("fuzzy - fixed", "search - fixed", "fuzzy - search"):
            t = test(f"allocation / AUC {s}", c)
            cells += [f"{sgn(t.mean_diff)} $[{t.ci_low:+.3f},\\,{t.ci_high:+.3f}]$", pfmt(t.t_p_holm)]
        rows.append(" & ".join(cells) + r" \\")
    write(out, "persite_rows", "\n".join(rows))

    # --- sensitivity suite summary (first-submission hp) -------------------
    tuned = load("main")
    rows = []
    for a in ARMS:
        g = tuned[tuned.allocation == a].groupby("partition_seed").mean(numeric_only=True)
        rows.append(f"{ARM_LABEL[a]} & {ms(g.macro_auc)} & {ms(g.macro_f1_score)} \\\\")
    write(out, "tuned_rows", "\n".join(rows))

    # --- server optimizers with and without DP ------------------------------
    dpo = load("dp_optimizers")
    nd = load("nodp")
    nd["arm"] = np.where(nd.scope == "federated", nd.aggregator, nd.scope)
    ndd = load("nodp_default").assign(arm="fedadam")
    nd = pd.concat([nd[nd.arm != "fedadam"], ndd])
    rk_dp = json.loads(fr[fr.family == "DP optimizers / macro AUC"].mean_ranks.iloc[0])
    rk_nd = json.loads(fr[fr.family == "no-DP optimizers / macro AUC"].mean_ranks.iloc[0])
    rows = []
    for o in ("fedavg", "fedprox", "fedadagrad", "fedyogi", "fedadam"):
        g = dpo[dpo.aggregator == o].groupby("partition_seed").mean(numeric_only=True)
        h = nd[nd.arm == o].groupby("partition_seed").mean(numeric_only=True)
        cells = [OPT_LABEL[o], ms(h.macro_auc), f"{rk_nd[o]:.2f}", ms(g.macro_auc), ms(g.macro_f1_score),
                 ms(g.macro_recall), f"{rk_dp[o]:.2f}"]
        if o == "fedadam":
            cells += ["n/a", "n/a"]
        else:
            t = test("DP optimizers / macro AUC", f"fedadam - {o}")
            cells += [f"{sgn(t.mean_diff)} $[{t.ci_low:+.3f},\\,{t.ci_high:+.3f}]$", pfmt(t.wilcoxon_p_holm)]
        rows.append(" & ".join(cells) + r" \\")
    write(out, "optim_rows", "\n".join(rows))

    # --- references: local-only, pooled, federated, + DP -------------------
    rows = []

    def ref_row(label, g):
        return " & ".join([label, ms(g.macro_auc)] + [ms(g[f"auc_{s}"]) for s in SITES]) + r" \\"
    rows.append(r"\multicolumn{6}{l}{\textit{Without differential privacy ($E=5$)}} \\")
    for arm, lab in (("local", "Local-only network (one per site)"), ("pooled", "Pooled network (centralized)"),
                     ("fedadam", "Federated, FedAdam")):
        rows.append(ref_row(lab, nd[nd.arm == arm].groupby("partition_seed").mean(numeric_only=True)))
    rows.append(r"\midrule")
    rows.append(r"\multicolumn{6}{l}{\textit{With differential privacy ($\sigma=2.5$, $E=1$)}} \\")
    for a in ("fixed", "fuzzy"):
        rows.append(ref_row(f"Federated, DP-FedAdam, {ARM_LABEL[a][0].lower() + ARM_LABEL[a][1:]}",
                            main[main.allocation == a].groupby("partition_seed").mean(numeric_only=True)))
    rows.append(r"\midrule")
    cells = ["Federated $-$ local-only (no DP)"]
    t = test("no-DP / macro AUC", "fedadam - local")
    cells.append(f"{sgn(t.mean_diff)}{'$^{*}$' if t.t_p_holm < 0.05 else ''}")
    for s in SITES:
        t = test(f"no-DP / AUC {s}", "fedadam - local")
        cells.append(f"{sgn(t.mean_diff)}{'$^{*}$' if t.t_p_holm < 0.05 else ''}")
    rows.append(" & ".join(cells) + r" \\")
    write(out, "refs_rows", "\n".join(rows))

    # --- epsilon targets ------------------------------------------------------
    et = load("eps_targets")
    rows = []
    for tag, tgt in (("target_eps_1.0", 1), ("target_eps_3.0", 3), ("target_eps_8.0", 8)):
        d = et[et.tag == tag]
        for j, a in enumerate(("fixed", "fuzzy", "search")):
            da = d[d.allocation == a]
            g = da.groupby("partition_seed").mean(numeric_only=True)
            cells = [f"\\multirow{{3}}{{*}}{{{tgt}}}" if j == 0 else "",
                     f"\\multirow{{3}}{{*}}{{{da.sigma.iloc[0]:.2f}}}" if j == 0 else "",
                     ARM_LABEL[a], f"${da.eps_worst.iloc[0]:.2f}$", f"${da.eps_spread.iloc[0]:.2f}\\times$", ms(g.macro_auc)]
            if a == "fixed":
                cells += ["n/a", "n/a"]
            else:
                t = test(f"{tag} / macro AUC", f"{a} - fixed")
                cells += [f"{sgn(t.mean_diff)} $[{t.ci_low:+.3f},\\,{t.ci_high:+.3f}]$", pfmt(t.t_p_holm)]
            rows.append(" & ".join(cells) + r" \\")
        if tgt != 8:
            rows.append(r"\midrule")
    write(out, "eps_rows", "\n".join(rows))

    # --- scalability ----------------------------------------------------------
    sc = load("scalability")
    rows = []
    for k, fam in ((2, "8 clients"), (4, "16 clients")):
        d = sc[sc.shards == k]
        for j, a in enumerate(("fixed", "fuzzy", "search")):
            da = d[d.allocation == a]
            g = da.groupby("partition_seed").mean(numeric_only=True)
            cells = [f"\\multirow{{3}}{{*}}{{{4 * k}}}" if j == 0 else "", ARM_LABEL[a],
                     f"${da.eps_worst.iloc[0]:.2f}$", f"${da.eps_spread.iloc[0]:.2f}\\times$", ms(g.macro_auc)]
            if a == "fixed":
                cells += ["n/a", "n/a"]
            else:
                t = test(f"{fam} / macro AUC", f"{a} - fixed")
                cells += [f"{sgn(t.mean_diff)} $[{t.ci_low:+.3f},\\,{t.ci_high:+.3f}]$", pfmt(t.t_p_holm)]
            rows.append(" & ".join(cells) + r" \\")
        if k == 2:
            rows.append(r"\midrule")
    write(out, "scale_rows", "\n".join(rows))

    # --- membership inference + subgroup recall ----------------------------
    mia = load("mia_audit")
    rows = []
    for arm, lab in (("no-DP local-only", "Local-only, no DP"), ("no-DP fedadam", "Federated FedAdam, no DP"),
                     ("DP fixed", "DP-FedAdam, fixed"), ("DP fuzzy", "DP-FedAdam, fuzzy"),
                     ("DP search", "DP-FedAdam, exact equalization")):
        d = mia[mia.arm == arm]
        cells = [lab]
        for s in SITES:
            x = d[d.site == s]
            cells.append(f"${x.attack_auc.mean():.3f}$ / ${x.advantage.mean():.3f}$")
        rows.append(" & ".join(cells) + r" \\")
    write(out, "mia_rows", "\n".join(rows))

    sub = load("subgroup_recall")
    rows = []
    for arm, lab in (("no-DP fedadam", "Federated FedAdam, no DP"), ("DP fixed", "DP-FedAdam, fixed"),
                     ("DP fuzzy", "DP-FedAdam, fuzzy"), ("DP search", "DP-FedAdam, exact equalization")):
        d = sub[sub.arm == arm].groupby("group").recall.mean()
        rows.append(f"{lab} & ${d['female']:.3f}$ & ${d['male']:.3f}$ & ${d['male'] - d['female']:+.3f}$ & "
                    f"${d['age<55']:.3f}$ & ${d['age>=55']:.3f}$ & ${d['age>=55'] - d['age<55']:+.3f}$ \\\\")
    write(out, "subgroup_rows", "\n".join(rows))

    # --- fuzzy sensitivity --------------------------------------------------
    fs = load("fuzzy_sensitivity")
    rows = []
    for kind in ("reference", "membership functions", "output universe"):
        for _, r in fs[fs.kind == kind].iterrows():
            lots = ", ".join(str(x) for x in json.loads(r.lots))
            rows.append(f"{kind.capitalize()} & {r.variant.capitalize()} & {lots} & ${r.worst_eps:.2f}$ & ${r.spread:.2f}\\times$ \\\\")
    rb = fs[fs.kind == "rule base"]
    ref = fs.iloc[0]
    same = rb[rb.lots == ref.lots]
    rows.append(f"Rule base & {len(same)} of {len(rb)} single-rule changes (rules that do not fire) & "
                f"{', '.join(str(x) for x in json.loads(ref.lots))} & ${ref.worst_eps:.2f}$ & ${ref.spread:.2f}\\times$ \\\\")
    for _, r in rb[rb.lots != ref.lots].iterrows():
        v = r.variant.replace("&", r"\&").replace("->", r"$\rightarrow$")
        lots = ", ".join(str(x) for x in json.loads(r.lots))
        rows.append(f"Rule base & {v} & {lots} & ${r.worst_eps:.2f}$ & ${r.spread:.2f}\\times$ \\\\")
    write(out, "sens_rows", "\n".join(rows))

    # --- appendix: per-partition results ------------------------------------
    t = part(main, "allocation", "macro_auc")[ARMS]
    rows = [f"{p} & " + " & ".join(f"{t.loc[p, a]:.4f}" for a in ARMS)
            + f" & {t.loc[p, 'fuzzy'] - t.loc[p, 'fixed']:+.4f} & {t.loc[p, 'search'] - t.loc[p, 'fixed']:+.4f} \\\\"
            for p in t.index]
    rows.append(r"\midrule")
    rows.append("Mean & " + " & ".join(f"{t[a].mean():.4f}" for a in ARMS)
                + f" & {(t.fuzzy - t.fixed).mean():+.4f} & {(t.search - t.fixed).mean():+.4f} \\\\")
    rows.append("SD & " + " & ".join(f"{t[a].std(ddof=1):.4f}" for a in ARMS)
                + f" & {(t.fuzzy - t.fixed).std(ddof=1):.4f} & {(t.search - t.fixed).std(ddof=1):.4f} \\\\")
    write(out, "partition_rows", "\n".join(rows))
    # --- graded federation -----------------------------------------------
    if os.path.exists(os.path.join(RES, "graded.csv")):
        gr = load("graded")
        ga = load("graded_allocation")
        fam = "graded 8 clients / macro AUC"
        rows = []
        for a in ARMS:
            d = gr[gr.allocation == a]
            g = d.groupby("partition_seed").mean(numeric_only=True)
            lots = ", ".join(str(x) for x in json.loads(ga[(ga.kind == "arm") & (ga.variant == a)].lots.iloc[0]))
            cells = [ARM_LABEL[a], lots, f"${d.eps_worst.iloc[0]:.2f}$", f"${d.eps_spread.iloc[0]:.2f}\\times$",
                     ms(g.macro_auc_sites)]
            if a == "fixed":
                cells += ["n/a", "n/a"]
            else:
                t = test(fam, f"{a} - fixed")
                cells += [f"{sgn(t.mean_diff)} $[{t.ci_low:+.3f},\\,{t.ci_high:+.3f}]$", pfmt(t.t_p_holm)]
            rows.append(" & ".join(cells) + r" \\")
        u = ga[ga.kind == "output universe"].sort_values("spread").iloc[0]
        rows.append(r"\midrule")
        rows.append(f"Fuzzy, best output universe {u.variant} & {', '.join(str(x) for x in json.loads(u.lots))} & "
                    f"${u.worst_eps:.2f}$ & ${u.spread:.2f}\\times$ & -- & -- & -- \\\\")
        write(out, "graded_rows", "\n".join(rows))
    return nums


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def figures(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve

    plt.rcParams.update({"font.size": 8, "font.family": "DejaVu Sans", "axes.spines.top": False,
                         "axes.spines.right": False, "axes.linewidth": 0.6, "axes.edgecolor": "#444444",
                         "xtick.major.width": 0.6, "ytick.major.width": 0.6, "axes.grid": True,
                         "grid.color": "#E6E6E6", "grid.linewidth": 0.5, "axes.axisbelow": True,
                         "legend.frameon": False, "pdf.fonttype": 42, "savefig.bbox": "tight",
                         "axes.titlesize": 8, "axes.titleweight": "bold", "axes.titlelocation": "left"})
    fig_dir = os.path.join(out, "figures")
    main = load("main_default")
    nv = load("noise_variance_by_allocation").set_index("arm")

    # Fig. per-site epsilon ---------------------------------------------------
    fig, ax = plt.subplots(figsize=(3.4, 2.3))
    arms = ["fixed", "fuzzy", "search"]
    w = 0.26
    for j, a in enumerate(arms):
        e = json.loads(nv.eps[a])
        x = np.arange(4) + (j - 1) * w
        ax.bar(x, e, w - 0.03, color=COL[a], label=ARM_PLOT[a], hatch={"fixed": "", "fuzzy": "//", "search": ".."}[a],
               edgecolor="white", linewidth=0.4)
        for xi, ei in zip(x, e):
            ax.text(xi, ei + 0.12, f"{ei:.1f}", ha="center", va="bottom", fontsize=6, color="#222222")
    n = json.loads(main.n_train.iloc[0])
    ax.set_xticks(np.arange(4))
    ax.set_xticklabels([f"{s}\n$n$={k}" for s, k in zip(SITES, n)])
    ax.set_ylabel(r"Privacy budget $\varepsilon$ ($\delta=10^{-5}$)")
    ax.set_ylim(0, 9.6)
    ax.grid(axis="x", visible=False)
    ax.legend(ncol=3, loc="upper right", fontsize=6.5, handlelength=1.2, columnspacing=0.9, bbox_to_anchor=(1.02, 1.12))
    fig.savefig(os.path.join(fig_dir, "fig_eps_per_site.pdf"))
    plt.close(fig)

    def npz(suite, rid):
        return np.load(os.path.join(RES, "preds", suite, rid + ".npz"))

    # Fig. convergence ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    rounds = np.arange(1, 101)
    for a in ("fixed", "fuzzy", "search"):
        d = main[main.allocation == a]
        H = pd.DataFrame([npz("main_default", r).get("history_macro_auc") for r in d.run_id])
        H["p"] = d.partition_seed.values
        Hp = H.groupby("p").mean().values
        m, s = Hp.mean(0), Hp.std(0, ddof=1)
        ci = 1.96 * s / math.sqrt(len(Hp))
        ax.fill_between(rounds, m - ci, m + ci, color=COL[a], alpha=0.18, linewidth=0)
        ax.plot(rounds, m, color=COL[a], lw=1.3, ls=LS[a], label=ARM_PLOT[a])
        ax.text(101.5, m[-1] + {"fixed": -0.004, "fuzzy": -0.003, "search": 0.004}[a], f"{m[-1]:.3f}", color="#222222",
                fontsize=6.5, va="center")
    ax.set_xlabel("Communication round")
    ax.set_ylabel("Macro AUC (test folds)")
    ax.set_xlim(1, 100)
    ax.set_ylim(0.5, 0.82)
    ax.legend(loc="lower right", fontsize=6.5)
    fig.savefig(os.path.join(fig_dir, "fig_convergence.pdf"))
    plt.close(fig)

    # Fig. mean ROC per site ---------------------------------------------------
    grid = np.linspace(0, 1, 201)
    nd = load("nodp_default")
    series = [("nodp", "FedAdam, no DP", nd, "nodp_default", ":"),
              ("fixed", "DP-FedAdam, fixed", main[main.allocation == "fixed"], "main_default", "--"),
              ("fuzzy", "DP-FedAdam, fuzzy", main[main.allocation == "fuzzy"], "main_default", "-")]
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.1), sharey=True)
    for ax, s in zip(axes, SITES):
        for key, lab, d, suite, ls in series:
            curves, aucs = [], []
            for p, dp_ in d.groupby("partition_seed"):
                c = []
                for rid in dp_.run_id:
                    z = npz(suite, rid)
                    fpr, tpr, _ = roc_curve(z[f"test_y_{s}"], z[f"test_p_{s}"])
                    c.append(np.interp(grid, fpr, tpr))
                curves.append(np.mean(c, 0))
            curves = np.array(curves)
            m = curves.mean(0)
            lo, hi = np.percentile(curves, [2.5, 97.5], axis=0)
            ax.fill_between(grid, lo, hi, color=COL[key], alpha=0.13, linewidth=0)
            ax.plot(grid, m, color=COL[key], lw=1.2, ls=ls, label=f"{lab}")
        ax.plot([0, 1], [0, 1], color="#BBBBBB", lw=0.6)
        ax.set_title(s)
        ax.set_xlabel("False-positive rate")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.005)
        ax.set_aspect("equal")
    axes[0].set_ylabel("True-positive rate")
    axes[0].legend(loc="lower right", fontsize=5.8, handlelength=1.6)
    fig.savefig(os.path.join(fig_dir, "fig_roc_mean.pdf"))
    plt.close(fig)

    # Fig. privacy-utility ------------------------------------------------------
    et = load("eps_targets")
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    for a in ("fixed", "fuzzy", "search"):
        xs, ys, es = [], [], []
        for tag in ("target_eps_1.0", "target_eps_3.0", "target_eps_8.0"):
            d = et[(et.tag == tag) & (et.allocation == a)]
            g = d.groupby("partition_seed").macro_auc.mean()
            xs.append(d.eps_worst.iloc[0]); ys.append(g.mean()); es.append(1.96 * g.std(ddof=1) / math.sqrt(len(g)))
        ax.errorbar(xs, ys, yerr=es, color=COL[a], marker=MARK[a], ms=4.5, lw=1.2, ls=LS[a], capsize=2,
                    label=ARM_PLOT[a], markeredgecolor="white", markeredgewidth=0.6)
    ref = nd.groupby("partition_seed").macro_auc.mean().mean()
    ax.axhline(ref, color=COL["nodp"], lw=0.8, ls=":")
    ax.text(0.72, ref + 0.004, f"FedAdam without DP ({ref:.3f})", fontsize=6.3, color="#222222")
    ax.set_xscale("log")
    ax.set_xticks([0.75, 1, 2, 3, 6, 8])
    ax.set_xticklabels(["0.75", "1", "2", "3", "6", "8"])
    ax.minorticks_off()
    ax.set_xlabel(r"Worst-case $\varepsilon$ across sites (log scale)")
    ax.set_ylabel("Macro AUC")
    ax.set_ylim(0.64, 0.83)
    ax.legend(loc="lower right", fontsize=6.5)
    fig.savefig(os.path.join(fig_dir, "fig_privacy_utility.pdf"))
    plt.close(fig)

    # Fig. paired differences ---------------------------------------------------
    t = part(main, "allocation", "macro_auc")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.3), gridspec_kw={"width_ratios": [1.25, 1]})
    ax = axes[0]
    order = (t.fuzzy - t.fixed).sort_values().index
    x = np.arange(len(order))
    for a in ("fuzzy", "search", "proportional"):
        ax.plot(x, (t[a] - t.fixed).loc[order].values, ls="none", marker=MARK[a], ms=4, color=COL[a], label=ARM_PLOT[a],
                markeredgecolor="white", markeredgewidth=0.5)
    ax.axhline(0, color="#444444", lw=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([str(p) for p in order], fontsize=6)
    ax.set_xlabel("Data partition (sorted by the fuzzy difference)")
    ax.set_ylabel("Macro AUC minus fixed allocation")
    ax.set_title("(a) Paired difference per partition")
    ax.legend(fontsize=6.3, loc="upper left", ncol=3, columnspacing=0.8, handletextpad=0.2)
    ax.set_ylim(-0.03, 0.05)
    ax = axes[1]
    tests = load("statistical_tests")
    w = 0.36
    for j, (a, comp) in enumerate((("fuzzy", "fuzzy - fixed"), ("search", "search - fixed"))):
        md, lo, hi = [], [], []
        for s in SITES:
            r = tests[(tests.family == f"allocation / AUC {s}") & (tests.comparison == comp)].iloc[0]
            md.append(r.mean_diff); lo.append(r.mean_diff - r.ci_low); hi.append(r.ci_high - r.mean_diff)
        xx = np.arange(4) + (j - 0.5) * w
        ax.bar(xx, md, w - 0.04, color=COL[a], label=ARM_PLOT[a], hatch={"fuzzy": "//", "search": ".."}[a],
               edgecolor="white", linewidth=0.4)
        ax.errorbar(xx, md, yerr=[lo, hi], fmt="none", ecolor="#222222", lw=0.8, capsize=2)
    ax.axhline(0, color="#444444", lw=0.7)
    ax.set_xticks(np.arange(4))
    ax.set_xticklabels(SITES)
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("AUC minus fixed allocation")
    ax.set_title("(b) Per-site difference (95% CI)")
    ax.legend(fontsize=6.3, loc="upper center", ncol=2)
    ax.set_ylim(-0.035, 0.075)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "fig_paired.pdf"))
    plt.close(fig)

    # Fig. momentum check -------------------------------------------------------
    mc = load("momentum_noise_check_curves")
    fig, ax = plt.subplots(figsize=(3.4, 2.3))
    cols = ["#0072B2", "#D55E00", "#009E73"]
    for c, (b, d) in zip(cols, mc.groupby("beta_1")):
        ax.plot(d["round"], d.predicted, color=c, lw=1.2, label=rf"$\beta_1={b:.2f}$")
        ax.plot(d["round"][::4], d.observed[::4], ls="none", marker="o", ms=2.6, color=c, markeredgecolor="white",
                markeredgewidth=0.3)
    ax.set_xlabel("Communication round $t$")
    ax.set_ylabel(r"$\mathrm{Var}(n_t)\,/\,\mathrm{Var}(e_t)$")
    ax.set_ylim(0, 0.4)
    ax.set_xlim(0, 60)
    ax.legend(fontsize=6.5, title="line: Eq. (identity); dots: simulation", title_fontsize=6)
    fig.savefig(os.path.join(fig_dir, "fig_momentum_check.pdf"))
    plt.close(fig)


# ---------------------------------------------------------------------------
# Pre-registered headroom study
# ---------------------------------------------------------------------------

def headroom_study_outputs(out):
    """Tables and figure of the headroom study (experiments/headroom_study.py)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy import stats

    from experiments.headroom_study import _blocks, split, study_table, tune_threshold

    runs = pd.concat([load("headroom_study"), load("headroom_study_fuzzy")], ignore_index=True)
    table = study_table()
    dev, conf = split()
    val = _blocks(runs, "val_macro_auc_sites")
    test = _blocks(runs, "macro_auc_sites")
    x = tune_threshold(val.loc[dev], table)
    tb = table.set_index("federation")
    share = tb["constrained_share"]

    # design table -----------------------------------------------------------
    rows = []
    for f in dev + conf:
        r = tb.loc[f]
        n = json.loads(r.sizes)
        tm = test.loc[f].mean()
        rows.append(f"{f} & {'Dev.' if f in dev else 'Conf.'} & {len(n)} & {min(n)}--{max(n)} & ${r.constrained_share:.2f}$ & "
                    f"${r.noise_reduction:.2f}$ & ${r.fuzzy_frac:.2f}$ & ${tm['0']:.3f}$ & ${tm['0.5']:.3f}$ & ${tm['1']:.3f}$ & "
                    f"${tm['fuzzy']:.3f}$ \\\\")
        if f == dev[-1]:
            rows.append(r"\midrule")
    write(out, "hs_design_rows", "\n".join(rows))

    # summary on the confirmation federations --------------------------------
    t = test.loc[conf].copy()
    t["PT"] = [t.loc[(f, p), "1" if share[f] >= x else "0"] for f, p in t.index]
    eps_col = {"0": "eps_mean_0", "0.5": "eps_mean_0.5", "1": "eps_mean_1", "fuzzy": "eps_mean_fuzzy"}
    fed_best = t[["0", "0.5", "1"]].groupby(level="federation").mean().max(axis=1)
    label = {"0": "Never spend (no harm)", "0.5": "Always half", "1": "Always equalize",
             "PT": f"Crisp threshold ($c\\ge{x:.2f}$)", "fuzzy": "Fuzzy policy"}
    rows, pts = [], {}
    for col in ("0", "0.5", "1", "PT", "fuzzy"):
        fm = t[col].groupby(level="federation").mean()
        eps = np.mean([tb.loc[f, eps_col["1" if share[f] >= x else "0"]] if col == "PT" else tb.loc[f, eps_col[col]]
                       for f in fm.index])
        d = t[col].values
        cells = [label[col], f"${d.mean():.4f}$", f"${float((fed_best - fm).mean()):.4f}$", f"${eps:.2f}$"]
        for ref in ("1", "0"):
            if col == ref or (col == "PT" and ref == "1" and x <= share[conf].min()):
                cells += ["n/a"]
                continue
            dd = (t[col] - t[ref]).values
            h = stats.t.ppf(0.975, len(dd) - 1) * dd.std(ddof=1) / np.sqrt(len(dd))
            cells.append(f"${dd.mean():+.4f}$ $[{dd.mean() - h:+.4f},\\,{dd.mean() + h:+.4f}]$")
        rows.append(" & ".join(cells) + r" \\")
        pts[col] = (eps, d.mean(), stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d)))
    write(out, "hs_summary_rows", "\n".join(rows))

    # numbers quoted in the text ---------------------------------------------
    def pair(a, b, frame):
        dd = (frame[a] - frame[b]).values
        h = stats.t.ppf(0.975, len(dd) - 1) * dd.std(ddof=1) / np.sqrt(len(dd))
        lo1 = dd.mean() - stats.t.ppf(0.95, len(dd) - 1) * dd.std(ddof=1) / np.sqrt(len(dd))
        return dict(diff=dd.mean(), lo=dd.mean() - h, hi=dd.mean() + h, lo1=lo1, wins=int((dd > 0).sum()), n=len(dd),
                    p_w=stats.wilcoxon(dd).pvalue, p_w1=stats.wilcoxon(dd, alternative="greater").pvalue)
    allb = test.loc[dev + conf]
    nums = {"threshold": x, "conf": {k: pair(*k.split("|"), t) for k in ("fuzzy|1", "fuzzy|0", "fuzzy|0.5", "0.5|1", "0.5|0")},
            "all": {k: pair(*k.split("|"), allb) for k in ("fuzzy|1", "fuzzy|0", "fuzzy|0.5", "0.5|0", "1|0")},
            "dev": {k: pair(*k.split("|"), test.loc[dev]) for k in ("fuzzy|1", "0.5|1", "1|0")}}
    with open(os.path.join(out, "tables", "hs_numbers.json"), "w") as fh:
        json.dump(nums, fh, indent=1, default=float)

    # figure ----------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    style = {"0": ("#D55E00", "s"), "0.5": ("#AA4499", "D"), "1": ("#009E73", "^"), "fuzzy": ("#0072B2", "o")}
    short = {"0": "Never spend", "0.5": "Always half", "1": "Always equalize\n(= crisp threshold)", "fuzzy": "Fuzzy policy"}
    for col in ("0", "0.5", "1", "fuzzy"):
        e, m, h = pts[col]
        c, mk = style[col]
        ax.errorbar([e], [m], yerr=[h], color=c, marker=mk, ms=6, capsize=2.5, lw=1.1, markeredgecolor="white",
                    markeredgewidth=0.6, ls="none")
        pos = {"0": (e + 0.025, m, "left"), "0.5": (e - 0.03, m + 0.0012, "right"),
               "1": (e + 0.03, m - 0.0012, "left"), "fuzzy": (e + 0.02, m + 0.0047, "left")}[col]
        ax.text(pos[0], pos[1], short[col], fontsize=6.3, ha=pos[2], va="center", color="#222222")
    ax.set_xlabel("Mean client budget $\\varepsilon$ (privacy spent)")
    ax.set_ylabel("Macro AUC, confirmation federations")
    ax.set_xlim(5.35, 6.45)
    ax.set_ylim(0.772, 0.791)
    fig.savefig(os.path.join(out, "figures", "fig_headroom_study.pdf"))
    plt.close(fig)
    return nums


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    for sub in ("tables", "figures"):
        os.makedirs(os.path.join(args.out, sub), exist_ok=True)
    print(tables(args.out))
    figures(args.out)
    print(json.dumps(headroom_study_outputs(args.out), indent=1, default=float))
    print("done ->", args.out)
