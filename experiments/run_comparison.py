"""
experiments/run_comparison.py
=============================
Investigation 1 – Performance comparison across FL strategies.

Runs four strategies over 100 federated rounds using the five real-world
cardiovascular datasets:
    • FedAvg    (μ = 0, standard baseline)
    • FedProx   (μ = 0.01, proximal regularisation only)
    • FedCVR    (μ = 0.1, FedProx client + adaptive server aggregation)
    • FedCVR+DP (FedCVR with Opacus Differential Privacy, noise σ = 1.1)

Outputs
-------
  results/comparison_metrics.csv   – round-level metrics for all strategies
  results/comparison_plot.png      – 2×2 metric comparison figure

Usage
-----
    # From the repository root:
    python -m experiments.run_comparison --data_dir ./data --rounds 100

    # Or, specifying an output directory:
    python -m experiments.run_comparison --data_dir /path/to/csvs --out_dir ./results
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from typing import Dict, Optional

import flwr as fl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Allow running as a script from the repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fedcvr.client import build_client
from fedcvr.data_utils import aggregate_metrics_fn, load_and_preprocess_data
from fedcvr.strategy import FedCVRStrategy


# ---------------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------------

SCENARIOS: Dict[str, Dict] = {
    "FedAvg": {
        "mu": 0.0,
        "dp": None,
        "strategy_kwargs": {"eta": 0.0},   # disable Adam on server (plain FedAvg)
        "linestyle": "-",
        "color": "tab:blue",
    },
    "FedProx (μ=0.01)": {
        "mu": 0.01,
        "dp": None,
        "strategy_kwargs": {"eta": 0.0},   # proximal client, plain server
        "linestyle": "--",
        "color": "tab:orange",
    },
    "FedCVR (ours)": {
        "mu": 0.1,
        "dp": None,
        "strategy_kwargs": {"eta": 0.01},  # full: proximal client + Adam server
        "linestyle": "-",
        "color": "tab:green",
    },
    "FedCVR+DP (σ=1.1)": {
        "mu": 0.1,
        "dp": {"noise_multiplier": 1.1, "max_grad_norm": 1.0},
        "strategy_kwargs": {"eta": 0.01},
        "linestyle": ":",
        "color": "tab:red",
    },
}


# ---------------------------------------------------------------------------
# Main simulation runner
# ---------------------------------------------------------------------------

def run(data_dir: str = ".", num_rounds: int = 100, out_dir: str = "results") -> None:
    os.makedirs(out_dir, exist_ok=True)

    # Load datasets
    client_train_data, client_test_data, dataset_names = load_and_preprocess_data(
        data_dir=data_dir
    )
    if client_train_data is None:
        print("ERROR: Could not load datasets. Aborting.")
        sys.exit(1)

    num_clients = len(client_train_data)
    history_storage: Dict[str, fl.server.history.History] = {}

    for name, cfg in SCENARIOS.items():
        print(f"\n{'='*60}")
        print(f"  Running: {name}")
        print(f"{'='*60}")

        use_dp = cfg["dp"] is not None

        def make_client_fn(dp_cfg: Optional[Dict], mu_val: float):
            def client_fn(cid: str) -> fl.client.Client:
                return build_client(
                    cid=cid,
                    client_train_data=client_train_data,
                    client_test_data=client_test_data,
                    local_epochs=5,
                    use_dp=dp_cfg is not None,
                    dp_config=dp_cfg,
                ).to_client()
            return client_fn

        # Override eta=0 to use plain FedAvg averaging on the server
        eta = cfg["strategy_kwargs"].get("eta", 0.01)

        strategy = FedCVRStrategy(
            eta=eta,
            fraction_fit=1.0,
            fraction_evaluate=1.0,
            min_fit_clients=num_clients,
            min_evaluate_clients=num_clients,
            min_available_clients=num_clients,
            evaluate_metrics_aggregation_fn=aggregate_metrics_fn,
            on_fit_config_fn=lambda _round, mu=cfg["mu"]: {"mu": mu},
        )

        history = fl.simulation.start_simulation(
            client_fn=make_client_fn(cfg["dp"], cfg["mu"]),
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
        "FedCVR vs. Baselines – Performance Comparison (100 rounds, 5 clients)",
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
    args = parser.parse_args()

    run(data_dir=args.data_dir, num_rounds=args.rounds, out_dir=args.out_dir)
