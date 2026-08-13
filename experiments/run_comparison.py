"""
experiments/run_comparison.py
=============================
Investigation 1 - Performance comparison across FL strategies.

Reproduces the paper's Table benchmark: six strategies over 100 federated
rounds, all under the *same* client-side DP-SGD configuration
(sigma=1.1, C=1.0), so that observed differences are attributable solely to
server-side aggregation design ("For each method, the same client-side
DP-SGD configuration is applied" - Section "Competing Methods"):
    - FedAvg              (mu=0,    plain weighted-average server)
    - FedProx (mu=0.01)    (proximal client, plain weighted-average server)
    - FedCluster (k=2)     (plain FedAvg run independently per client cluster)
    - FedAdagrad           (second-moment-only adaptive server optimizer)
    - FedYogi              (sign-based second-moment adaptive server optimizer)
    - FedCVR (ours)        (mu=0, DP-FedAdam server optimizer - Algorithm 1)

The no-DP vs. DP privacy-utility trade-off for the proposed method alone is
covered separately by ``run_dp_sensitivity.py`` (Investigation 3).

Outputs
-------
  results/comparison_metrics.csv   - round-level metrics for all strategies
  results/comparison_plot.png      - 2x2 metric comparison figure

Usage
-----
    # From the repository root:
    python -m experiments.run_comparison --data_dir ./data --rounds 100

    # Or, specifying an output directory:
    python -m experiments.run_comparison --data_dir /path/to/csvs --out_dir ./results
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from typing import Dict, Optional

import flwr as fl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd

warnings.filterwarnings("ignore")

# Allow running as a script from the repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fedcvr.baselines import FedAdagradStrategy, FedClusterStrategy, FedYogiStrategy
from fedcvr.client import build_client
from fedcvr.data_utils import (
    aggregate_metrics_fn,
    cluster_clients_by_distribution,
    load_and_preprocess_data,
)
from fedcvr.strategy import FedCVRStrategy


# ---------------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------------

# Shared DP-SGD configuration applied identically to every method (sigma=1.1,
# C=1.0), matching Table benchmark's caption in the paper.
DP_CONFIG: Dict = {"noise_multiplier": 1.1, "max_grad_norm": 1.0}

SCENARIOS: Dict[str, Dict] = {
    "FedAvg": {
        "mu": 0.0,
        "strategy_cls": FedCVRStrategy,
        "strategy_kwargs": {"eta": 0.0},   # disable Adam on server (plain FedAvg)
        "linestyle": "-",
        "color": "tab:blue",
    },
    "FedProx (μ=0.01)": {
        "mu": 0.01,
        "strategy_cls": FedCVRStrategy,
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
    "FedCVR (ours)": {
        "mu": 0.0,
        "strategy_cls": FedCVRStrategy,
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
) -> None:
    os.makedirs(out_dir, exist_ok=True)

    # Load datasets
    client_train_data, _client_val_data, client_test_data, dataset_names = load_and_preprocess_data(
        data_dir=data_dir
    )
    if client_train_data is None:
        print("ERROR: Could not load datasets. Aborting.")
        sys.exit(1)

    if hyperparams_path is not None:
        with open(hyperparams_path) as f:
            tuned = json.load(f)["best_params"]
        SCENARIOS["FedCVR (ours)"]["strategy_kwargs"] = {
            "eta": tuned["eta"],
            "beta_1": tuned["beta_1"],
            "beta_2": tuned["beta_2"],
            "tau": tuned["tau"],
        }
        print(f"Loaded Optuna-tuned hyperparameters from {hyperparams_path}: {tuned}")

    num_clients = len(client_train_data)
    cluster_assignment = cluster_clients_by_distribution(
        client_train_data, n_clusters=2, random_state=42
    )
    history_storage: Dict[str, fl.server.history.History] = {}

    def make_client_fn():
        def client_fn(cid: str) -> fl.client.Client:
            return build_client(
                cid=cid,
                client_train_data=client_train_data,
                client_test_data=client_test_data,
                local_epochs=5,
                use_dp=True,
                dp_config=DP_CONFIG,
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

    fig.suptitle(
        "FedCVR vs. Baselines – Performance Comparison Under DP (σ=1.1, 100 rounds, 5 clients)",
        fontsize=15,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plot_path = os.path.join(out_dir, "comparison_plot.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Plot saved to: {plot_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FedCVR – Investigation 1: Strategy comparison")
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
    args = parser.parse_args()

    run(
        data_dir=args.data_dir,
        num_rounds=args.rounds,
        out_dir=args.out_dir,
        hyperparams_path=args.hyperparams,
    )
