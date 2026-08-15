"""
experiments/run_privacy_budget_test.py
=======================================
Fixed-batch (L=32 for every client) reference run at the adopted
single-digit-epsilon DP configuration: sigma=2.5, local_epochs=1, at the
full 100-round protocol. Serves as the fixed-batch baseline for the
fuzzy-vs-fixed comparison in run_dp_sensitivity_fuzzy.py. See README
"Privacy Budget Accounting" and Results §5 for how this configuration
was chosen.

Usage
-----
    python -m experiments.run_privacy_budget_test --data_dir ./data --rounds 100 \
        --hyperparams results/best_hyperparameters.json --seed 42
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

from fedcvr.client import build_client
from fedcvr.data_utils import aggregate_metrics_fn, get_pre_smote_train_sizes, load_and_preprocess_data
from fedcvr.evaluation import calibrated_final_evaluation
from fedcvr.rdp_accountant import RDPAccountant
from fedcvr.strategy import FedCVRStrategy

MU = 0.0
DEFAULT_SERVER_KWARGS: Dict = {
    "eta": 0.1, "beta_1": 0.9, "beta_2": 0.999, "tau": 1e-3,
}
BATCH_SIZE = 32
DELTA = 1e-5

# Adopted single-digit-epsilon configuration (see module docstring for how
# this was chosen over the "high sigma, epochs=5" alternative).
SCENARIOS: Dict[str, Dict] = {
    "Adopted DP config (σ=2.5, epochs=1, fixed batch)": {"noise_multiplier": 2.5, "local_epochs": 1},
}

LINE_STYLES = {
    "Adopted DP config (σ=2.5, epochs=1, fixed batch)": ("-.", "tab:brown"),
}


def run(
    data_dir: str = "data",
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

    client_train_data, client_val_data, client_test_data, dataset_names = load_and_preprocess_data(
        data_dir=data_dir
    )
    if client_train_data is None:
        print("ERROR: Could not load datasets. Aborting.")
        sys.exit(1)

    client_labels = [f"H{i+1}" for i in range(len(dataset_names))]
    n_train_per_client = get_pre_smote_train_sizes(data_dir=data_dir)
    print(f"Pre-SMOTE training-fold sizes: {dict(zip(client_labels, n_train_per_client))}")

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
    epsilon_reports: Dict[str, Dict[str, float]] = {}

    for name, cfg in SCENARIOS.items():
        sigma = cfg["noise_multiplier"]
        local_epochs = cfg["local_epochs"]
        print(f"\n{'='*60}\n  Running: {name}\n{'='*60}")

        # Confirm the actual per-client epsilon for this config before
        # spending compute on training - the whole point of this script.
        accountant = RDPAccountant(
            noise_multiplier=sigma, max_grad_norm=1.0, batch_size=BATCH_SIZE, delta=DELTA
        )
        eps_per_client = {
            label: accountant.compute_epsilon(n_train, num_rounds, local_epochs)
            for label, n_train in zip(client_labels, n_train_per_client)
        }
        epsilon_reports[name] = eps_per_client
        for label, eps in eps_per_client.items():
            print(f"    {label}: epsilon={eps:.2f}")

        dp_config = {"noise_multiplier": sigma, "max_grad_norm": 1.0}

        def make_client_fn(dp_cfg, epochs):
            def client_fn(cid: str) -> fl.client.Client:
                return build_client(
                    cid=cid,
                    client_train_data=client_train_data,
                    client_test_data=client_test_data,
                    batch_size=BATCH_SIZE,
                    local_epochs=epochs,
                    use_dp=True,
                    dp_config=dp_cfg,
                ).to_client()
            return client_fn

        strategy = FedCVRStrategy(
            **server_kwargs,
            fraction_fit=1.0,
            fraction_evaluate=1.0,
            min_fit_clients=num_clients,
            min_evaluate_clients=num_clients,
            min_available_clients=num_clients,
            evaluate_metrics_aggregation_fn=aggregate_metrics_fn,
            on_fit_config_fn=lambda _round: {"mu": MU},
        )

        history = fl.simulation.start_simulation(
            client_fn=make_client_fn(dp_config, local_epochs),
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
    # Save calibrated utility metrics
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

    calibrated_csv_path = os.path.join(out_dir, "calibrated_privacy_budget_metrics.csv")
    pd.DataFrame(flat_rows).to_csv(calibrated_csv_path, index=False)
    print(f"\nCalibrated final metrics saved to: {calibrated_csv_path}")

    per_client_csv_path = os.path.join(out_dir, "calibrated_privacy_budget_per_client.csv")
    pd.DataFrame(per_client_rows).to_csv(per_client_csv_path, index=False)
    print(f"Per-client calibrated metrics saved to: {per_client_csv_path}")

    eps_rows = []
    for name, eps_per_client in epsilon_reports.items():
        for label, eps in eps_per_client.items():
            eps_rows.append({"scenario": name, "client": label, "epsilon": eps})
    eps_csv_path = os.path.join(out_dir, "privacy_budget_epsilon.csv")
    pd.DataFrame(eps_rows).to_csv(eps_csv_path, index=False)
    print(f"Epsilon report saved to: {eps_csv_path}")

    # -------------------------------------------------------------------
    # Round-level metrics + plot
    # -------------------------------------------------------------------
    rows = []
    for name, hist in history_storage.items():
        for metric in ["accuracy", "precision", "recall", "f1_score"]:
            for rnd, val in hist.metrics_distributed.get(metric, []):
                rows.append({"scenario": name, "round": rnd, "metric": metric, "value": val})
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "privacy_budget_round_metrics.csv"), index=False)

    metrics_to_plot = ["accuracy", "precision", "recall", "f1_score"]
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharex=True)
    axes = axes.flatten()
    for ax, metric in zip(axes, metrics_to_plot):
        for name in SCENARIOS:
            hist = history_storage[name]
            data = hist.metrics_distributed.get(metric, [])
            if data:
                rounds, values = zip(*data)
                ls, col = LINE_STYLES[name]
                ax.plot(rounds, values, label=name, linestyle=ls, color=col, marker=".", markersize=4, alpha=0.9)
        ax.set_title(metric.replace("_", " ").capitalize(), fontsize=13)
        ax.set_ylabel("Metric Value")
        ax.set_ylim(0.0, 1.0)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend(fontsize=9)
    for ax in axes[2:]:
        ax.set_xlabel("Federated Round", fontsize=11)
    fig.suptitle(
        f"FedCVR - Single-Digit-Epsilon Privacy Budget Test, {num_rounds} rounds, {num_clients} clients",
        fontsize=15,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plot_path = os.path.join(out_dir, "privacy_budget_plot.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved to: {plot_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="FedCVR - test whether single-digit epsilon is reachable at the full 100-round protocol"
    )
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument("--out_dir", type=str, default="results")
    parser.add_argument("--hyperparams", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run(
        data_dir=args.data_dir,
        num_rounds=args.rounds,
        out_dir=args.out_dir,
        hyperparams_path=args.hyperparams,
        seed=args.seed,
    )
