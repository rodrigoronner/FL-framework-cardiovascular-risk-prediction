"""
experiments/run_dp_sensitivity.py
==================================
Privacy-utility trade-off of DP-FedAdam across four DP regimes: No DP,
Low (sigma=0.8), Medium (sigma=1.1), High (sigma=1.5) - fixed batch
size L=32 for every client. See run_privacy_budget_test.py for the
single-digit-epsilon configuration ultimately adopted.

Usage
-----
    python -m experiments.run_dp_sensitivity --data_dir ./data --rounds 100
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
matplotlib.use("Agg")  # non-interactive backend - see run_comparison.py for why
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from flwr.common import parameters_to_ndarrays

from dpfedadam.client import build_client
from dpfedadam.data_utils import aggregate_metrics_fn, load_and_preprocess_data
from dpfedadam.evaluation import calibrated_final_evaluation
from dpfedadam.strategy import DPFedAdamStrategy


# ---------------------------------------------------------------------------
# DP scenario configuration
# ---------------------------------------------------------------------------

MU = 0.0         # no proximal term (Algorithm 2): the proposed method differs
                 # from FedAvg only via the server-side Adam optimizer and DP-SGD
DEFAULT_SERVER_KWARGS: Dict = {
    "eta": 0.1, "beta_1": 0.9, "beta_2": 0.999, "tau": 1e-3,
}  # server Adam hyperparameters as reported in the paper; overridden by
   # --hyperparams if an Optuna run_hpo.py output is supplied.

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

def run(
    data_dir: str = ".",
    num_rounds: int = 100,
    out_dir: str = "results",
    hyperparams_path: Optional[str] = None,
    seed: int = 42,
) -> None:
    os.makedirs(out_dir, exist_ok=True)

    import numpy as _np
    import torch as _torch
    _np.random.seed(seed)
    _torch.manual_seed(seed)

    client_train_data, client_val_data, client_test_data, dataset_names = load_and_preprocess_data(data_dir=data_dir)
    if client_train_data is None:
        print("ERROR: Could not load datasets. Aborting.")
        sys.exit(1)

    server_kwargs = dict(DEFAULT_SERVER_KWARGS)
    if hyperparams_path is not None:
        with open(hyperparams_path) as f:
            tuned = json.load(f)["best_params"]
        server_kwargs.update(tuned)
        print(f"Loaded Optuna-tuned hyperparameters from {hyperparams_path}: {tuned}")

    num_clients = len(client_train_data)
    input_features = client_train_data[0][0].shape[1]
    history_storage: Dict[str, fl.server.history.History] = {}
    calibrated_results: Dict[str, Dict] = {}

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

        strategy = DPFedAdamStrategy(
            **server_kwargs,
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

        final_ndarrays = parameters_to_ndarrays(strategy.final_weights)
        calibrated_results[name] = calibrated_final_evaluation(
            final_ndarrays=final_ndarrays,
            input_features=input_features,
            client_val_data=client_val_data,
            client_test_data=client_test_data,
            client_names=dataset_names,
        )
        macro = calibrated_results[name]["macro"]
        print(
            f"  Macro test metrics: acc={macro['accuracy']:.3f} prec={macro['precision']:.3f} "
            f"rec={macro['recall']:.3f} f1={macro['f1_score']:.3f} auc={macro['auc']:.3f}"
        )
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
    # Save final, threshold-calibrated metrics (the paper's privacy-utility table)
    # -------------------------------------------------------------------
    flat_rows = []
    per_client_rows = []
    for name, result in calibrated_results.items():
        flat_row = {"scenario": name}
        flat_row.update({f"macro_{k}": v for k, v in result["macro"].items()})
        flat_row.update({f"macro_local_{k}": v for k, v in result["macro_local_threshold"].items()})
        flat_row.update({f"pooled_{k}": v for k, v in result["pooled"].items()})
        flat_rows.append(flat_row)
        for client_label, views in result["per_client"].items():
            row = {"scenario": name, "client": client_label}
            row.update({f"global_{k}": v for k, v in views["global_threshold"].items()})
            row.update({f"local_{k}": v for k, v in views["local_threshold"].items()})
            per_client_rows.append(row)

    calibrated_csv_path = os.path.join(out_dir, "calibrated_dp_sensitivity_metrics.csv")
    pd.DataFrame(flat_rows).to_csv(calibrated_csv_path, index=False)
    print(f"Calibrated final metrics (macro + pooled) saved to: {calibrated_csv_path}")

    per_client_csv_path = os.path.join(out_dir, "calibrated_dp_sensitivity_per_client.csv")
    pd.DataFrame(per_client_rows).to_csv(per_client_csv_path, index=False)
    print(f"Per-client calibrated metrics saved to: {per_client_csv_path}")

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
        "DP-FedAdam – Privacy-Utility Trade-off Analysis (DP Sensitivity)",
        fontsize=15,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plot_path = os.path.join(out_dir, "dp_sensitivity_plot.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved to: {plot_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="DP-FedAdam – Investigation 3: DP sensitivity analysis"
    )
    parser.add_argument("--data_dir", type=str, default="data",
                        help="Directory containing the four CSV dataset files.")
    parser.add_argument("--rounds", type=int, default=100,
                        help="Number of federated communication rounds.")
    parser.add_argument("--out_dir", type=str, default="results",
                        help="Directory to save metrics CSV and plot PNG.")
    parser.add_argument("--hyperparams", type=str, default=None,
                        help="Path to best_hyperparameters.json produced by "
                             "experiments/run_hpo.py; overrides the server-side "
                             "eta/beta_1/beta_2/tau if given.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Seed for training-time stochasticity; the data "
                             "split stays fixed at random_state=42 regardless.")
    args = parser.parse_args()

    run(
        data_dir=args.data_dir,
        num_rounds=args.rounds,
        out_dir=args.out_dir,
        hyperparams_path=args.hyperparams,
        seed=args.seed,
    )
