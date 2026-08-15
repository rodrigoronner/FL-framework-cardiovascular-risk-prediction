"""
experiments/run_comparison.py
=============================
Benchmark comparison across 6 FL strategies (FedAvg, FedProx, FedCluster,
FedAdagrad, FedYogi, DP-FedAdam), all under the same client-side configuration
so differences are attributable to server-side aggregation design alone.
Pass --no_dp for the no-privacy stage. See run_dp_sensitivity.py /
run_dp_sensitivity_fuzzy.py / run_privacy_budget_test.py for the DP-only
privacy-utility analyses.

Usage
-----
    python -m experiments.run_comparison --data_dir ./data --rounds 100
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from typing import Dict, Optional

import flwr as fl
import matplotlib
matplotlib.use("Agg")  # non-interactive backend; the native macOS backend's
                        # plt.show() blocks forever with no display attached.
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd

warnings.filterwarnings("ignore")

# Allow running as a script from the repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from flwr.common import parameters_to_ndarrays

from dpfedadam.baselines import FedAdagradStrategy, FedClusterStrategy, FedYogiStrategy
from dpfedadam.client import build_client
from dpfedadam.data_utils import (
    aggregate_metrics_fn,
    cluster_clients_by_distribution,
    load_and_preprocess_data,
)
from dpfedadam.evaluation import calibrated_final_evaluation
from dpfedadam.strategy import DPFedAdamStrategy


# ---------------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------------

# Shared DP-SGD configuration applied identically to every method.
DP_CONFIG: Dict = {"noise_multiplier": 1.1, "max_grad_norm": 1.0}

SCENARIOS: Dict[str, Dict] = {
    "FedAvg": {
        "mu": 0.0,
        "strategy_cls": DPFedAdamStrategy,
        "strategy_kwargs": {"eta": 0.0},   # disable Adam on server (plain FedAvg)
        "linestyle": "-",
        "color": "tab:blue",
    },
    "FedProx (μ=0.01)": {
        "mu": 0.01,
        "strategy_cls": DPFedAdamStrategy,
        "strategy_kwargs": {"eta": 0.0},   # proximal client, plain server
        "linestyle": "--",
        "color": "tab:orange",
    },
    "FedCluster (k=2)": {
        "mu": 0.0,
        "strategy_cls": FedClusterStrategy,
        "strategy_kwargs": {},             # cluster_assignment injected in run()
        "linestyle": "-.",
        "color": "tab:purple",
    },
    "FedAdagrad": {
        "mu": 0.0,
        "strategy_cls": FedAdagradStrategy,
        "strategy_kwargs": {"eta": 0.1, "tau": 1e-3},
        "linestyle": "--",
        "color": "tab:brown",
    },
    "FedYogi": {
        "mu": 0.0,
        "strategy_cls": FedYogiStrategy,
        "strategy_kwargs": {"eta": 0.1, "beta_1": 0.9, "beta_2": 0.999, "tau": 1e-3},
        "linestyle": ":",
        "color": "tab:pink",
    },
    "DP-FedAdam (ours)": {
        "mu": 0.0,
        "strategy_cls": DPFedAdamStrategy,
        "strategy_kwargs": {"eta": 0.1},   # Algorithm 2 has no proximal term; only the Adam server optimiser differs from FedAvg
        "linestyle": "-",
        "color": "tab:green",
    },
}


# ---------------------------------------------------------------------------
# Main simulation runner
# ---------------------------------------------------------------------------

def run(
    data_dir: str = ".",
    num_rounds: int = 100,
    out_dir: str = "results",
    hyperparams_path: Optional[str] = None,
    seed: int = 42,
    use_dp: bool = True,
) -> None:
    os.makedirs(out_dir, exist_ok=True)

    # Seed training-time stochasticity only; the data split stays fixed
    # at load_and_preprocess_data's own random_state=42 regardless of seed.
    import numpy as _np
    import torch as _torch
    _np.random.seed(seed)
    _torch.manual_seed(seed)

    # Load datasets
    client_train_data, client_val_data, client_test_data, dataset_names = load_and_preprocess_data(
        data_dir=data_dir
    )
    if client_train_data is None:
        print("ERROR: Could not load datasets. Aborting.")
        sys.exit(1)

    print(f"DP-SGD: {'ENABLED (sigma=' + str(DP_CONFIG['noise_multiplier']) + ')' if use_dp else 'DISABLED (plain FL, no privacy noise)'}")

    if hyperparams_path is not None:
        with open(hyperparams_path) as f:
            tuned = json.load(f)["best_params"]
        SCENARIOS["DP-FedAdam (ours)"]["strategy_kwargs"] = {
            "eta": tuned["eta"],
            "beta_1": tuned["beta_1"],
            "beta_2": tuned["beta_2"],
            "tau": tuned["tau"],
        }
        print(f"Loaded Optuna-tuned hyperparameters from {hyperparams_path}: {tuned}")

    num_clients = len(client_train_data)
    input_features = client_train_data[0][0].shape[1]
    cluster_assignment = cluster_clients_by_distribution(
        client_train_data, n_clusters=2, random_state=42
    )
    history_storage: Dict[str, fl.server.history.History] = {}
    calibrated_results: Dict[str, Dict] = {}

    def make_client_fn():
        def client_fn(cid: str) -> fl.client.Client:
            return build_client(
                cid=cid,
                client_train_data=client_train_data,
                client_test_data=client_test_data,
                local_epochs=5,
                use_dp=use_dp,
                dp_config=DP_CONFIG if use_dp else None,
            ).to_client()
        return client_fn

    for name, cfg in SCENARIOS.items():
        print(f"\n{'='*60}")
        print(f"  Running: {name}")
        print(f"{'='*60}")

        strategy_kwargs = dict(cfg["strategy_kwargs"])
        if cfg["strategy_cls"] is FedClusterStrategy:
            strategy_kwargs["cluster_assignment"] = cluster_assignment

        strategy = cfg["strategy_cls"](
            **strategy_kwargs,
            fraction_fit=1.0,
            fraction_evaluate=1.0,
            min_fit_clients=num_clients,
            min_evaluate_clients=num_clients,
            min_available_clients=num_clients,
            evaluate_metrics_aggregation_fn=aggregate_metrics_fn,
            on_fit_config_fn=lambda _round, mu=cfg["mu"]: {"mu": mu},
        )

        history = fl.simulation.start_simulation(
            client_fn=make_client_fn(),
            num_clients=num_clients,
            config=fl.server.ServerConfig(num_rounds=num_rounds),
            strategy=strategy,
            client_resources={"num_cpus": 1, "num_gpus": 0.0},
        )
        history_storage[name] = history

        final_ndarrays = parameters_to_ndarrays(strategy.final_weights)
        calibrated_results[name] = calibrated_final_evaluation(
            final_ndarrays=final_ndarrays,
            input_features=input_features,
            client_val_data=client_val_data,
            client_test_data=client_test_data,
            client_names=dataset_names,
        )
        macro = calibrated_results[name]["macro"]
        macro_local = calibrated_results[name]["macro_local_threshold"]
        pooled = calibrated_results[name]["pooled"]
        print(
            f"  Macro, global threshold  : acc={macro['accuracy']:.3f} prec={macro['precision']:.3f} "
            f"rec={macro['recall']:.3f} f1={macro['f1_score']:.3f} auc={macro['auc']:.3f}"
        )
        print(
            f"  Macro, local threshold   : acc={macro_local['accuracy']:.3f} prec={macro_local['precision']:.3f} "
            f"rec={macro_local['recall']:.3f} f1={macro_local['f1_score']:.3f} auc={macro_local['auc']:.3f}"
        )
        print(
            f"  Pooled (secondary)       : acc={pooled['accuracy']:.3f} prec={pooled['precision']:.3f} "
            f"rec={pooled['recall']:.3f} f1={pooled['f1_score']:.3f} auc={pooled['auc']:.3f}"
        )
        print(f"  Finished: {name}")

    # -------------------------------------------------------------------
    # Save round-level metrics to CSV
    # -------------------------------------------------------------------
    rows = []
    for name, hist in history_storage.items():
        for metric in ["accuracy", "precision", "recall", "f1_score"]:
            data = hist.metrics_distributed.get(metric, [])
            for rnd, val in data:
                rows.append({"strategy": name, "round": rnd, "metric": metric, "value": val})

    df = pd.DataFrame(rows)
    csv_path = os.path.join(out_dir, "comparison_metrics.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nMetrics saved to: {csv_path}")

    # -------------------------------------------------------------------
    # Save final, threshold-calibrated metrics (the paper's benchmark table)
    # -------------------------------------------------------------------
    flat_rows = []
    per_client_rows = []
    for name, result in calibrated_results.items():
        flat_row = {"strategy": name}
        flat_row.update({f"macro_{k}": v for k, v in result["macro"].items()})
        flat_row.update({f"macro_local_{k}": v for k, v in result["macro_local_threshold"].items()})
        flat_row.update({f"pooled_{k}": v for k, v in result["pooled"].items()})
        flat_rows.append(flat_row)

        for client_label, views in result["per_client"].items():
            row = {"strategy": name, "client": client_label}
            row.update({f"global_{k}": v for k, v in views["global_threshold"].items()})
            row.update({f"local_{k}": v for k, v in views["local_threshold"].items()})
            per_client_rows.append(row)

    calibrated_df = pd.DataFrame(flat_rows)
    calibrated_csv_path = os.path.join(out_dir, "calibrated_final_metrics.csv")
    calibrated_df.to_csv(calibrated_csv_path, index=False)
    print(f"Calibrated final metrics (macro + pooled) saved to: {calibrated_csv_path}")

    per_client_df = pd.DataFrame(per_client_rows)
    per_client_csv_path = os.path.join(out_dir, "calibrated_per_client_metrics.csv")
    per_client_df.to_csv(per_client_csv_path, index=False)
    print(f"Per-client calibrated metrics saved to: {per_client_csv_path}")

    # -------------------------------------------------------------------
    # Plot 2×2 figure
    # -------------------------------------------------------------------
    metrics_to_plot = ["accuracy", "precision", "recall", "f1_score"]
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharex=True)
    axes = axes.flatten()

    for ax, metric in zip(axes, metrics_to_plot):
        for name, cfg in SCENARIOS.items():
            hist = history_storage[name]
            data = hist.metrics_distributed.get(metric, [])
            if data:
                rounds, values = zip(*data)
                ax.plot(
                    rounds, values,
                    label=name,
                    linestyle=cfg["linestyle"],
                    color=cfg["color"],
                    marker=".",
                    markersize=3,
                    alpha=0.9,
                )
        ax.set_title(metric.replace("_", " ").capitalize(), fontsize=13)
        ax.set_ylabel("Metric Value")
        ax.set_ylim(0.0, 1.0)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend(fontsize=9)

    for ax in axes[2:]:
        ax.set_xlabel("Federated Round", fontsize=11)

    dp_label = f"Under DP (σ={DP_CONFIG['noise_multiplier']})" if use_dp else "No DP (plain FL)"
    fig.suptitle(
        f"DP-FedAdam vs. Baselines – Performance Comparison {dp_label}, {num_rounds} rounds, {num_clients} clients",
        fontsize=15,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plot_path = os.path.join(out_dir, "comparison_plot.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved to: {plot_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DP-FedAdam – Investigation 1: Strategy comparison")
    parser.add_argument("--data_dir", type=str, default="data",
                        help="Directory containing the five CSV dataset files.")
    parser.add_argument("--rounds", type=int, default=100,
                        help="Number of federated communication rounds.")
    parser.add_argument("--out_dir", type=str, default="results",
                        help="Directory to save metrics CSV and plot PNG.")
    parser.add_argument("--hyperparams", type=str, default=None,
                        help="Path to best_hyperparameters.json produced by "
                             "experiments/run_hpo.py; overrides the proposed "
                             "method's eta/beta_1/beta_2/tau if given.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Seed for training-time stochasticity (model init, "
                             "DP noise, client sampling); the data split stays "
                             "fixed at random_state=42 regardless of this value.")
    parser.add_argument("--no_dp", action="store_true",
                        help="Disable DP-SGD (plain FL, no privacy noise) - for "
                             "comparing federated-without-DP against the "
                             "centralized baseline before introducing privacy noise.")
    args = parser.parse_args()

    run(
        data_dir=args.data_dir,
        num_rounds=args.rounds,
        out_dir=args.out_dir,
        hyperparams_path=args.hyperparams,
        seed=args.seed,
        use_dp=not args.no_dp,
    )
