"""
experiments/run_dp_sensitivity.py
==================================
Investigation 3 – Differential Privacy sensitivity analysis.

Evaluates the privacy-utility trade-off of FedCVR across four DP regimes:
    • No DP (baseline)
    • Low Privacy   (noise_multiplier = 0.8,  ε ≈ high)
    • Medium Privacy(noise_multiplier = 1.1,  ε ≈ medium)
    • High Privacy  (noise_multiplier = 1.5,  ε ≈ low)

Outputs
-------
  results/dp_sensitivity_metrics.csv  – round-level metrics for all regimes
  results/dp_sensitivity_plot.png     – 2×2 privacy-utility trade-off figure

Usage
-----
    python -m experiments.run_dp_sensitivity --data_dir ./data --rounds 50
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
import pandas as pd

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fedcvr.client import build_client
from fedcvr.data_utils import aggregate_metrics_fn, load_and_preprocess_data
from fedcvr.strategy import FedCVRStrategy


# ---------------------------------------------------------------------------
# DP scenario configuration
# ---------------------------------------------------------------------------

MU = 0.1         # proximal term (fixed for all DP scenarios)
SERVER_ETA = 0.01

DP_SCENARIOS: Dict[str, Optional[Dict]] = {
    "No DP (Baseline)": None,
    "Low Privacy  (σ=0.8)":    {"noise_multiplier": 0.8,  "max_grad_norm": 1.0},
    "Medium Privacy (σ=1.1)":  {"noise_multiplier": 1.1,  "max_grad_norm": 1.0},
    "High Privacy (σ=1.5)":    {"noise_multiplier": 1.5,  "max_grad_norm": 1.0},
}

LINE_STYLES = {
    "No DP (Baseline)":       ("-",  "tab:blue"),
    "Low Privacy  (σ=0.8)":   ("--", "tab:orange"),
    "Medium Privacy (σ=1.1)": (":",  "tab:green"),
    "High Privacy (σ=1.5)":   ("-.", "tab:red"),
}


# ---------------------------------------------------------------------------
# Main simulation runner
# ---------------------------------------------------------------------------

def run(data_dir: str = ".", num_rounds: int = 50, out_dir: str = "results") -> None:
    os.makedirs(out_dir, exist_ok=True)

    client_train_data, client_test_data, _ = load_and_preprocess_data(data_dir=data_dir)
    if client_train_data is None:
        print("ERROR: Could not load datasets. Aborting.")
        sys.exit(1)

    num_clients = len(client_train_data)
    history_storage: Dict[str, fl.server.history.History] = {}

    for name, dp_cfg in DP_SCENARIOS.items():
        print(f"\n{'='*60}")
        print(f"  Running: {name}")
        print(f"{'='*60}")

        def make_client_fn(dp_config):
            def client_fn(cid: str) -> fl.client.Client:
                return build_client(
                    cid=cid,
                    client_train_data=client_train_data,
                    client_test_data=client_test_data,
                    local_epochs=5,
                    use_dp=dp_config is not None,
                    dp_config=dp_config,
                ).to_client()
            return client_fn

        strategy = FedCVRStrategy(
            eta=SERVER_ETA,
            fraction_fit=1.0,
            fraction_evaluate=1.0,
            min_fit_clients=num_clients,
            min_evaluate_clients=num_clients,
            min_available_clients=num_clients,
            evaluate_metrics_aggregation_fn=aggregate_metrics_fn,
            on_fit_config_fn=lambda _: {"mu": MU},
        )

        history = fl.simulation.start_simulation(
            client_fn=make_client_fn(dp_cfg),
            num_clients=num_clients,
            config=fl.server.ServerConfig(num_rounds=num_rounds),
            strategy=strategy,
            client_resources={"num_cpus": 1, "num_gpus": 0.0},
        )
        history_storage[name] = history
        print(f"  Finished: {name}")

    # -------------------------------------------------------------------
    # Save to CSV
    # -------------------------------------------------------------------
    rows = []
    for name, hist in history_storage.items():
        for metric in ["accuracy", "precision", "recall", "f1_score"]:
            for rnd, val in hist.metrics_distributed.get(metric, []):
                rows.append({"scenario": name, "round": rnd, "metric": metric, "value": val})

    df = pd.DataFrame(rows)
    csv_path = os.path.join(out_dir, "dp_sensitivity_metrics.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nMetrics saved to: {csv_path}")

    # -------------------------------------------------------------------
    # 2×2 privacy-utility plot
    # -------------------------------------------------------------------
    metrics_to_plot = ["accuracy", "precision", "recall", "f1_score"]
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharex=True)
    axes = axes.flatten()

    for ax, metric in zip(axes, metrics_to_plot):
        for name, hist in history_storage.items():
            data = hist.metrics_distributed.get(metric, [])
            if data:
                rounds, values = zip(*data)
                ls, col = LINE_STYLES[name]
                ax.plot(
                    rounds, values,
                    label=name,
                    linestyle=ls,
                    color=col,
                    marker=".",
                    markersize=4,
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
        "FedCVR – Privacy-Utility Trade-off Analysis (DP Sensitivity)",
        fontsize=15,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plot_path = os.path.join(out_dir, "dp_sensitivity_plot.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Plot saved to: {plot_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="FedCVR – Investigation 3: DP sensitivity analysis"
    )
    parser.add_argument("--data_dir", type=str, default="data",
                        help="Directory containing the five CSV dataset files.")
    parser.add_argument("--rounds", type=int, default=50,
                        help="Number of federated communication rounds.")
    parser.add_argument("--out_dir", type=str, default="results",
                        help="Directory to save metrics CSV and plot PNG.")
    args = parser.parse_args()

    run(data_dir=args.data_dir, num_rounds=args.rounds, out_dir=args.out_dir)
