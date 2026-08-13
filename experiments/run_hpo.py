"""
experiments/run_hpo.py
=======================
Hyperparameter optimization for the proposed DP-FedAdam server optimizer,
using Optuna (TPE sampler + median pruning).

Tunes the four server-side hyperparameters (eta, beta_1, beta_2, tau) at the
primary DP operating point (sigma=1.1, C=1.0) against the *validation* fold
of every client (never the test fold used for the paper's benchmark and
privacy-utility tables), using a short 30-round federated run per trial to
keep the search tractable. The best configuration found is written to
``results/best_hyperparameters.json`` and can be fed into
``run_comparison.py`` / ``run_dp_sensitivity.py``.

Usage
-----
    python -m experiments.run_hpo --data_dir ./data --n_trials 40 \
        --rounds_per_trial 30 --out_dir results
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from typing import Dict

import flwr as fl
import optuna

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fedcvr.client import build_client
from fedcvr.data_utils import aggregate_metrics_fn, load_and_preprocess_data
from fedcvr.strategy import FedCVRStrategy

DP_CONFIG: Dict = {"noise_multiplier": 1.1, "max_grad_norm": 1.0}


def _objective_factory(
    client_train_data, client_val_data, num_rounds: int
):
    num_clients = len(client_train_data)

    def objective(trial: optuna.Trial) -> float:
        eta = trial.suggest_float("eta", 0.01, 0.5, log=True)
        beta_1 = trial.suggest_float("beta_1", 0.7, 0.99)
        beta_2 = trial.suggest_float("beta_2", 0.9, 0.9999, log=True)
        tau = trial.suggest_float("tau", 1e-5, 1e-1, log=True)

        def client_fn(cid: str) -> fl.client.Client:
            return build_client(
                cid=cid,
                client_train_data=client_train_data,
                client_test_data=client_val_data,  # evaluate on val, not test
                local_epochs=5,
                use_dp=True,
                dp_config=DP_CONFIG,
            ).to_client()

        strategy = FedCVRStrategy(
            eta=eta,
            beta_1=beta_1,
            beta_2=beta_2,
            tau=tau,
            fraction_fit=1.0,
            fraction_evaluate=1.0,
            min_fit_clients=num_clients,
            min_evaluate_clients=num_clients,
            min_available_clients=num_clients,
            evaluate_metrics_aggregation_fn=aggregate_metrics_fn,
            on_fit_config_fn=lambda _round: {"mu": 0.0},
        )

        history = fl.simulation.start_simulation(
            client_fn=client_fn,
            num_clients=num_clients,
            config=fl.server.ServerConfig(num_rounds=num_rounds),
            strategy=strategy,
            client_resources={"num_cpus": 1, "num_gpus": 0.0},
        )

        f1_curve = history.metrics_distributed.get("f1_score", [])
        if not f1_curve:
            return 0.0

        # Report intermediate F1 for pruning, at every round the strategy
        # produced a value, and return the final round's F1 as the
        # trial's objective (val-set F1, not accuracy, given the paper's
        # emphasis on recall/F1 under class imbalance).
        for rnd, val in f1_curve:
            trial.report(val, step=rnd)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return f1_curve[-1][1]

    return objective


def run(
    data_dir: str = "data",
    n_trials: int = 40,
    rounds_per_trial: int = 30,
    out_dir: str = "results",
    seed: int = 42,
) -> None:
    os.makedirs(out_dir, exist_ok=True)

    client_train_data, client_val_data, _client_test_data, _ = load_and_preprocess_data(
        data_dir=data_dir
    )
    if client_train_data is None:
        print("ERROR: Could not load datasets. Aborting.")
        sys.exit(1)

    sampler = optuna.samplers.TPESampler(seed=seed)
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=10)
    study = optuna.create_study(
        direction="maximize", sampler=sampler, pruner=pruner,
        study_name="dp_fedadam_hpo",
    )
    objective = _objective_factory(client_train_data, client_val_data, rounds_per_trial)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print("\nBest trial:")
    print(f"  Value (val F1): {study.best_value:.4f}")
    print(f"  Params: {study.best_params}")

    out_path = os.path.join(out_dir, "best_hyperparameters.json")
    with open(out_path, "w") as f:
        json.dump(
            {
                "best_value_val_f1": study.best_value,
                "best_params": study.best_params,
                "n_trials": n_trials,
                "rounds_per_trial": rounds_per_trial,
                "dp_config": DP_CONFIG,
            },
            f,
            indent=2,
        )
    print(f"Saved best hyperparameters to: {out_path}")

    trials_df = study.trials_dataframe()
    trials_csv = os.path.join(out_dir, "hpo_trials.csv")
    trials_df.to_csv(trials_csv, index=False)
    print(f"Saved trial history to: {trials_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="FedCVR - Optuna hyperparameter search for DP-FedAdam server optimizer"
    )
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--n_trials", type=int, default=40)
    parser.add_argument(
        "--rounds_per_trial", type=int, default=30,
        help="Shortened federated horizon per trial (full benchmark uses 100).",
    )
    parser.add_argument("--out_dir", type=str, default="results")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run(
        data_dir=args.data_dir,
        n_trials=args.n_trials,
        rounds_per_trial=args.rounds_per_trial,
        out_dir=args.out_dir,
        seed=args.seed,
    )
