"""
experiments/run_dp_sensitivity_fuzzy.py
========================================
Fuzzy per-client batch-size rebalancing (dpfedadam.fuzzy_fairness.
FuzzyFairnessController) at the adopted DP configuration (sigma=2.5,
local_epochs=1 - see run_privacy_budget_test.py). Computes each client's
baseline epsilon at a fixed L=32, asks the fuzzy controller for a
per-client batch-size multiplier from (dataset size, current epsilon),
and trains with those sizes instead of a uniform 32. Reports the epsilon
spread and calibrated utility (macro/pooled/per-client AUC/F1), directly
comparable to run_privacy_budget_test.py's fixed-batch numbers. See
README "Fuzzy Fairness Controller" and Results §4-5.

Usage
-----
    python -m experiments.run_dp_sensitivity_fuzzy --data_dir ./data --rounds 100 \
        --hyperparams results/best_hyperparameters.json --seed 42
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from typing import Dict, List, Optional

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
from dpfedadam.data_utils import (
    FILENAMES,
    aggregate_metrics_fn,
    get_pre_smote_train_sizes,
    load_and_preprocess_data,
)
from dpfedadam.evaluation import calibrated_final_evaluation
from dpfedadam.fuzzy_fairness import FairnessRebalanceReport, FuzzyFairnessController
from dpfedadam.rdp_accountant import RDPAccountant
from dpfedadam.strategy import DPFedAdamStrategy

MU = 0.0  # no proximal term - matches run_dp_sensitivity.py's Algorithm 2 convention
DEFAULT_SERVER_KWARGS: Dict = {
    "eta": 0.1, "beta_1": 0.9, "beta_2": 0.999, "tau": 1e-3,
}
BASE_BATCH_SIZE = 32
MIN_BATCH_SIZE = 8
MAX_BATCH_SIZE = 128
DELTA = 1e-5

# Adopted single-digit-epsilon configuration (see module docstring).
DP_SCENARIOS: Dict[str, Dict] = {
    "Adopted DP config (σ=2.5, epochs=1, fuzzy batch)": {"noise_multiplier": 2.5, "max_grad_norm": 1.0},
}

LINE_STYLES = {
    "Adopted DP config (σ=2.5, epochs=1, fuzzy batch)": ("-.", "tab:blue"),
}


def _fuzzy_rebalance(
    n_train_per_client: List[int],
    noise_multiplier: float,
    n_rounds: int,
    local_epochs: int,
    client_labels: List[str],
) -> FairnessRebalanceReport:
    accountant = RDPAccountant(
        noise_multiplier=noise_multiplier, max_grad_norm=1.0,
        batch_size=BASE_BATCH_SIZE, delta=DELTA,
    )
    controller = FuzzyFairnessController()
    return controller.rebalance(
        accountant=accountant,
        n_train_per_client=n_train_per_client,
        n_rounds=n_rounds,
        local_epochs=local_epochs,
        base_batch_size=BASE_BATCH_SIZE,
        min_batch_size=MIN_BATCH_SIZE,
        max_batch_size=MAX_BATCH_SIZE,
        client_labels=client_labels,
    )


def run(
    data_dir: str = "data",
    num_rounds: int = 100,
    out_dir: str = "results",
    hyperparams_path: Optional[str] = None,
    seed: int = 42,
    local_epochs: int = 1,
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
    fuzzy_reports: Dict[str, FairnessRebalanceReport] = {}

    for name, dp_cfg in DP_SCENARIOS.items():
        print(f"\n{'='*60}\n  Running: {name}  (fuzzy-rebalanced batch sizes)\n{'='*60}")

        report = _fuzzy_rebalance(
            n_train_per_client, dp_cfg["noise_multiplier"], num_rounds,
            local_epochs, client_labels,
        )
        fuzzy_reports[name] = report
        print(report)
        per_client_batch = [c.adjusted_batch_size for c in report.clients]

        def make_client_fn(dp_config: Dict, batch_sizes: List[int]):
            def client_fn(cid: str) -> fl.client.Client:
                idx = int(cid)
                return build_client(
                    cid=cid,
                    client_train_data=client_train_data,
                    client_test_data=client_test_data,
                    batch_size=batch_sizes[idx],
                    local_epochs=local_epochs,
                    use_dp=True,
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
            on_fit_config_fn=lambda _round: {"mu": MU},
        )

        history = fl.simulation.start_simulation(
            client_fn=make_client_fn(dp_cfg, per_client_batch),
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
    # Save calibrated utility metrics (same schema as run_dp_sensitivity.py,
    # for direct fixed-vs-fuzzy comparison)
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

    calibrated_csv_path = os.path.join(out_dir, "calibrated_dp_sensitivity_fuzzy_metrics.csv")
    pd.DataFrame(flat_rows).to_csv(calibrated_csv_path, index=False)
    print(f"\nCalibrated final metrics (macro + pooled) saved to: {calibrated_csv_path}")

    per_client_csv_path = os.path.join(out_dir, "calibrated_dp_sensitivity_fuzzy_per_client.csv")
    pd.DataFrame(per_client_rows).to_csv(per_client_csv_path, index=False)
    print(f"Per-client calibrated metrics saved to: {per_client_csv_path}")

    # -------------------------------------------------------------------
    # Save the fairness side: per-client epsilon before/after rebalancing,
    # plus the spread ratio, for every DP regime.
    # -------------------------------------------------------------------
    fuzzy_rows = []
    for name, report in fuzzy_reports.items():
        for c in report.clients:
            fuzzy_rows.append({
                "scenario": name, "client": c.label, "n_train": c.n_train,
                "baseline_batch_size": c.baseline_batch_size,
                "baseline_epsilon": c.baseline_epsilon,
                "recommended_multiplier": c.recommended_multiplier,
                "adjusted_batch_size": c.adjusted_batch_size,
                "adjusted_epsilon": c.adjusted_epsilon,
            })
        fuzzy_rows.append({
            "scenario": name, "client": "SPREAD (max/min)",
            "baseline_epsilon": report.baseline_spread,
            "adjusted_epsilon": report.adjusted_spread,
        })
    fuzzy_csv_path = os.path.join(out_dir, "fuzzy_epsilon_rebalancing.csv")
    pd.DataFrame(fuzzy_rows).to_csv(fuzzy_csv_path, index=False)
    print(f"Fuzzy epsilon rebalancing report saved to: {fuzzy_csv_path}")

    # -------------------------------------------------------------------
    # Round-level metrics + 2x2 plot (same layout as run_dp_sensitivity.py)
    # -------------------------------------------------------------------
    rows = []
    for name, hist in history_storage.items():
        for metric in ["accuracy", "precision", "recall", "f1_score"]:
            for rnd, val in hist.metrics_distributed.get(metric, []):
                rows.append({"scenario": name, "round": rnd, "metric": metric, "value": val})
    metrics_csv_path = os.path.join(out_dir, "dp_sensitivity_fuzzy_metrics.csv")
    pd.DataFrame(rows).to_csv(metrics_csv_path, index=False)
    print(f"Round-level metrics saved to: {metrics_csv_path}")

    metrics_to_plot = ["accuracy", "precision", "recall", "f1_score"]
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharex=True)
    axes = axes.flatten()
    for ax, metric in zip(axes, metrics_to_plot):
        for name in DP_SCENARIOS:
            hist = history_storage[name]
            data = hist.metrics_distributed.get(metric, [])
            if data:
                rounds, values = zip(*data)
                ls, col = LINE_STYLES[name]
                ax.plot(
                    rounds, values, label=name, linestyle=ls, color=col,
                    marker=".", markersize=4, alpha=0.9,
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
        f"DP-FedAdam + Fuzzy Fairness Rebalancing - DP Sensitivity, {num_rounds} rounds, {num_clients} clients",
        fontsize=15,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plot_path = os.path.join(out_dir, "dp_sensitivity_fuzzy_plot.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved to: {plot_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="DP-FedAdam - DP sensitivity with fuzzy per-client batch-size rebalancing"
    )
    parser.add_argument("--data_dir", type=str, default="data",
                        help="Directory containing the four CSV dataset files.")
    parser.add_argument("--rounds", type=int, default=100,
                        help="Number of federated communication rounds.")
    parser.add_argument("--out_dir", type=str, default="results",
                        help="Directory to save metrics CSVs and plot PNG.")
    parser.add_argument("--hyperparams", type=str, default=None,
                        help="Path to best_hyperparameters.json produced by "
                             "experiments/run_hpo.py; overrides the server-side "
                             "eta/beta_1/beta_2/tau if given.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Seed for training-time stochasticity; the data "
                             "split stays fixed at random_state=42 regardless.")
    parser.add_argument("--local_epochs", type=int, default=1,
                        help="Local SGD epochs per round (default 1, matching "
                             "the adopted single-digit-epsilon DP config).")
    args = parser.parse_args()

    run(
        data_dir=args.data_dir,
        num_rounds=args.rounds,
        out_dir=args.out_dir,
        hyperparams_path=args.hyperparams,
        seed=args.seed,
        local_epochs=args.local_epochs,
    )
