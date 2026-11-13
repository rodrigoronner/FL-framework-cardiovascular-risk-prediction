"""
experiments/revision_runner.py
===============================
Single-process federated simulator used for every experiment of the
revised paper. Each client runs DP-SGD through Opacus ``make_private``
(Poisson sampling, RDP accountant) and the server applies FedAvg, FedProx,
FedAdagrad, FedYogi or FedAdam (Reddi et al., 2021) in one Python loop,
so that thousands of (partition seed x training seed x arm) runs are
affordable and every run leaves an inspectable record:

* one CSV row per run in ``results_revision/<suite>.csv`` (tracked in git);
* one ``.npz`` per run in ``results_revision/preds/`` holding the test and
  training scores, used for mean ROC curves, subgroup recall and the
  membership-inference audit.

Two seeds are varied independently: ``partition_seed`` draws the
train/validation/test split, ``train_seed`` drives initialization, Poisson
sampling and DP noise. All arms share the same (partition, train) seed
pairs, so comparisons between arms are paired.

For DP runs the reported epsilon is cross-checked against the Opacus
accountant attached to the client that actually trained
(``eps_executed_*`` columns): the analytical audit and the executed
mechanism must agree.

Usage
-----
    python -m experiments.revision_runner --suite main --workers 8
    python -m experiments.revision_runner --list
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import warnings
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import numpy as np

warnings.filterwarnings("ignore")
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)

DELTA = 1e-5
CLIP = 1.0
BASE_L, MIN_L, MAX_L = 32, 8, 128
CLIENT_LR = 0.01
OUT_DIR = os.path.join(ROOT, "results_revision")

# Server hyperparameters. DP-FedAdam uses the configuration tuned for the
# first submission and held fixed throughout the revision (no re-tuning on
# the revised federation); baselines use the defaults of Reddi et al.
SERVER_HP = {
    "fedadam_dp": {"eta": 0.1681351247682783, "beta_1": 0.7816860862983565,
                   "beta_2": 0.9614101463242622, "tau": 0.0012792558846521696},
    "fedadam_nodp": {"eta": 0.0798814136031772, "beta_1": 0.7331874394990071,
                     "beta_2": 0.9096928466079917, "tau": 0.00178370625205815},
    "fedadam_default": {"eta": 0.1, "beta_1": 0.9, "beta_2": 0.999, "tau": 1e-3},
    "fedadagrad": {"eta": 0.1, "tau": 1e-3},
    "fedyogi": {"eta": 0.1, "beta_1": 0.9, "beta_2": 0.999, "tau": 1e-3},
}
FEDPROX_MU = 0.01


# ---------------------------------------------------------------------------
# Batch-size allocation
# ---------------------------------------------------------------------------

def allocate(allocation: str, n_train: List[int], sigma: float, rounds: int,
             epochs: int, target_eps: Optional[float] = None, frac: Optional[float] = None) -> Dict:
    """Return per-client lot sizes and accounted epsilons for one arm."""
    from dpfedadam.epsilon_matched import EpsilonMatchedController
    from dpfedadam.fuzzy_fairness import FuzzyFairnessController
    from dpfedadam.proportional_fairness import ProportionalFairnessController
    from dpfedadam.rdp_accountant import RDPAccountant

    acc = RDPAccountant(noise_multiplier=sigma, max_grad_norm=CLIP, batch_size=BASE_L, delta=DELTA)
    kw = dict(accountant=acc, n_train_per_client=n_train, n_rounds=rounds, local_epochs=epochs,
              base_batch_size=BASE_L, min_batch_size=MIN_L, max_batch_size=MAX_L)

    def eps_of(lots):
        out = []
        for n, lot in zip(n_train, lots):
            acc.batch_size = lot
            out.append(acc.compute_epsilon(n, rounds, epochs))
        return out

    base_eps = eps_of([BASE_L] * len(n_train))
    if allocation == "fixed":
        lots = [BASE_L] * len(n_train)
    elif allocation == "fuzzy":
        report = FuzzyFairnessController.for_federation(n_train, base_eps).rebalance(**kw)
        lots = [c.adjusted_batch_size for c in report.clients]
    elif allocation == "proportional":
        report = ProportionalFairnessController().rebalance(**kw)
        lots = [c.adjusted_batch_size for c in report.clients]
    elif allocation == "search":
        if target_eps is None:  # match the fuzzy controller's worst case
            target_eps = max(allocate("fuzzy", n_train, sigma, rounds, epochs)["eps"])
        report = EpsilonMatchedController(target_eps).rebalance(**kw)
        lots = [c.adjusted_batch_size for c in report.clients]
    elif allocation == "frac":
        # Headroom study: clients above the cap shrink to the cap; clients
        # below it move a fraction `frac` of the way from L_base to the
        # largest lot the cap allows (0 = no harm, 1 = exact equalization).
        cap = [c.adjusted_batch_size for c in EpsilonMatchedController(target_eps).rebalance(**kw).clients]
        lots = [c if c < BASE_L else int(round(BASE_L + frac * (c - BASE_L))) for c in cap]
    elif allocation in ("noharm", "half"):
        # Cap-based rules at the same worst-case target as "search".
        #   noharm: only sites above the cap shrink their lots; no site's
        #           budget rises (a Pareto improvement over fixed lots).
        #   half:   sites below the cap spend half of their headroom, i.e.
        #           move halfway from L_base to the largest lot the cap allows.
        if target_eps is None:
            target_eps = max(allocate("fuzzy", n_train, sigma, rounds, epochs)["eps"])
        cap = [c.adjusted_batch_size for c in EpsilonMatchedController(target_eps).rebalance(**kw).clients]
        if allocation == "noharm":
            lots = [min(BASE_L, c) for c in cap]
        else:
            lots = [c if c < BASE_L else int(round(BASE_L + 0.5 * (c - BASE_L))) for c in cap]
    else:
        raise ValueError(allocation)
    return {"lots": lots, "eps": eps_of(lots), "base_eps": base_eps}


# ---------------------------------------------------------------------------
# Federated training
# ---------------------------------------------------------------------------

class _Client:
    def __init__(self, X, y, lot: int, dp: bool, sigma: float, n_features: int):
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        from dpfedadam.model import Net

        self.n = len(y)
        self.model = Net(input_features=n_features)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=CLIENT_LR)
        self.loader = DataLoader(
            TensorDataset(torch.tensor(X, dtype=torch.float32),
                          torch.tensor(y, dtype=torch.float32).view(-1, 1)),
            batch_size=lot, shuffle=True,
        )
        self.criterion = torch.nn.BCELoss()
        self.engine = None
        self.sample_rate = None
        if dp:
            from opacus import PrivacyEngine

            self.engine = PrivacyEngine(accountant="rdp")
            self.model, self.optimizer, self.loader = self.engine.make_private(
                module=self.model, optimizer=self.optimizer, data_loader=self.loader,
                noise_multiplier=sigma, max_grad_norm=CLIP,
            )
            self.sample_rate = float(self.loader.sample_rate)

    def _net(self):
        return getattr(self.model, "_module", self.model)

    def fit(self, weights: List[np.ndarray], epochs: int, mu: float) -> List[np.ndarray]:
        import torch

        net = self._net()
        net.load_state_dict(OrderedDict(
            (k, torch.tensor(w)) for k, w in zip(net.state_dict().keys(), weights)))
        global_params = [p.detach().clone() for p in net.parameters()]
        self.model.train()
        for _ in range(epochs):
            for xb, yb in self.loader:
                self.optimizer.zero_grad()
                loss = self.criterion(self.model(xb), yb) if len(xb) else self.model(xb).sum()
                loss.backward()
                self.optimizer.step()
                if mu > 0.0:
                    # FedProx proximal pull. It depends only on the public
                    # global model, so it is applied after the privatized
                    # step and costs no privacy.
                    with torch.no_grad():
                        for p, g in zip(net.parameters(), global_params):
                            p.add_(-CLIENT_LR * mu * (p - g))
        return [v.detach().cpu().numpy().copy() for v in net.state_dict().values()]


def _server_step(agg: str, hp: Dict, state: Dict, w, delta, t: int):
    if agg in ("fedavg", "fedprox"):
        return [a + d for a, d in zip(w, delta)]
    if agg == "fedadagrad":
        state["v"] = [v + d ** 2 for v, d in zip(state["v"], delta)]
        return [a + hp["eta"] * d / (np.sqrt(v) + hp["tau"]) for a, d, v in zip(w, delta, state["v"])]
    if agg == "fedyogi":
        state["m"] = [hp["beta_1"] * m + (1 - hp["beta_1"]) * d for m, d in zip(state["m"], delta)]
        state["v"] = [v - (1 - hp["beta_2"]) * np.sign(v - d ** 2) * d ** 2 for v, d in zip(state["v"], delta)]
        return [a + hp["eta"] * m / (np.sqrt(np.abs(v)) + hp["tau"])
                for a, m, v in zip(w, state["m"], state["v"])]
    if agg == "fedadam":
        state["m"] = [hp["beta_1"] * m + (1 - hp["beta_1"]) * d for m, d in zip(state["m"], delta)]
        state["v"] = [hp["beta_2"] * v + (1 - hp["beta_2"]) * d ** 2 for v, d in zip(state["v"], delta)]
        m_hat = [m / (1 - hp["beta_1"] ** t) for m in state["m"]]
        v_hat = [v / (1 - hp["beta_2"] ** t) for v in state["v"]]
        return [a + hp["eta"] * mh / (np.sqrt(vh) + hp["tau"]) for a, mh, vh in zip(w, m_hat, v_hat)]
    raise ValueError(agg)


def federate(train_sets, lots, agg, hp, dp, sigma, rounds, epochs, seed,
             eval_fn=None) -> Tuple[List[np.ndarray], Dict]:
    import torch

    from dpfedadam.model import Net

    torch.manual_seed(seed)
    np.random.seed(seed)
    n_features = train_sets[0][0].shape[1]
    clients = [_Client(X, y, lot, dp, sigma, n_features) for (X, y), lot in zip(train_sets, lots)]
    sizes = np.array([c.n for c in clients], dtype=float)
    p = sizes / sizes.sum()
    w = [v.numpy().copy() for v in Net(input_features=n_features).state_dict().values()]
    mu = FEDPROX_MU if agg == "fedprox" else 0.0
    state: Dict = {}
    history = []
    for t in range(1, rounds + 1):
        updates = [c.fit(w, epochs, mu) for c in clients]
        avg = [sum(pk * u[j] for pk, u in zip(p, updates)) for j in range(len(w))]
        if t == 1:
            # Convention of the first submission's server: w_0 is bootstrapped from
            # round 1's weighted average, moment vectors start at zero.
            w = avg
            state = {"m": [np.zeros_like(a) for a in avg], "v": [np.zeros_like(a) for a in avg]}
        else:
            w = _server_step(agg, hp, state, w, [a - b for a, b in zip(avg, w)], t)
        if eval_fn is not None:
            history.append(eval_fn(w))
    info = {"history": history}
    if dp:
        info["eps_executed"] = [float(c.engine.get_epsilon(DELTA)) for c in clients]
        info["sample_rate"] = [c.sample_rate for c in clients]
    return w, info


# ---------------------------------------------------------------------------
# One run
# ---------------------------------------------------------------------------

def _shard(sets, k: int, seed: int):
    """Split every client's fold into ``k`` disjoint stratified shards."""
    if k == 1:
        return sets
    rng = np.random.RandomState(seed)
    out = []
    for X, y in sets:
        idx = np.concatenate([rng.permutation(np.where(y == c)[0]) for c in (0, 1)])
        for s in range(k):
            sel = idx[s::k]
            out.append((X[sel], y[sel]))
    return out


def _shard_ratios(sets, ratios, seed: int):
    """Split site s's fold into len(ratios[s]) disjoint stratified shards
    whose sizes follow the fractions ratios[s] (graded federations)."""
    rng = np.random.RandomState(seed)
    out = []
    for (X, y), fr in zip(sets, ratios):
        cuts = np.cumsum([0.0] + list(fr))
        parts = [[] for _ in fr]
        for c in (0, 1):
            idx = rng.permutation(np.where(y == c)[0])
            bounds = np.round(cuts * len(idx)).astype(int)
            for j in range(len(fr)):
                parts[j].append(idx[bounds[j]:bounds[j + 1]])
        for p in parts:
            sel = np.sort(np.concatenate(p))
            out.append((X[sel], y[sel]))
    return out


def run_id(cfg: Dict) -> str:
    return hashlib.md5(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:12]


def run_one(cfg: Dict) -> Dict:
    import torch

    torch.set_num_threads(1)
    from sklearn.metrics import roc_auc_score

    from dpfedadam.data_utils import load_and_preprocess_data
    from dpfedadam.evaluation import (calibrated_final_evaluation, load_model_from_ndarrays,
                                      predict_proba)

    t0 = time.time()
    c = {"allocation": "fixed", "aggregator": "fedadam", "dp": True, "sigma": 2.5,
         "epochs": 1, "rounds": 100, "shards": 1, "target_eps": None, "scope": "federated",
         "hp": None, "tag": "", "shard_ratios": None, "headroom_frac": None}
    c.update(cfg)
    train, val, test, _ = load_and_preprocess_data(
        os.path.join(ROOT, "data"), random_state=c["partition_seed"], verbose=False)
    site_test, site_val = test, val
    k = c["shards"]
    ratios = c.get("shard_ratios")
    if ratios:
        train, val, test = (_shard_ratios(s, ratios, c["partition_seed"]) for s in (train, val, test))
        labels = [f"H{s + 1}.{j + 1}" for s, r in enumerate(ratios) for j in range(len(r))]
    else:
        train, val, test = (_shard(s, k, c["partition_seed"]) for s in (train, val, test))
        labels = [f"H{i // k + 1}" + (f".{i % k + 1}" if k > 1 else "") for i in range(len(train))]

    if c["scope"] == "pooled":  # centralized reference: one model on pooled training data
        fit_sets = [(np.concatenate([X for X, _ in train]), np.concatenate([y for _, y in train]))]
    else:
        fit_sets = train
    n_train = [len(y) for _, y in fit_sets]
    n_features = train[0][0].shape[1]

    row = dict(c)
    row["run_id"] = run_id(cfg)
    if c["dp"]:
        alloc = allocate(c["allocation"], n_train, c["sigma"], c["rounds"], c["epochs"], c["target_eps"],
                         c.get("headroom_frac"))
        lots = alloc["lots"]
    else:
        alloc, lots = None, [BASE_L] * len(fit_sets)
    hp_key = c["hp"] or {"fedadam": "fedadam_dp" if c["dp"] else "fedadam_nodp"}.get(c["aggregator"], c["aggregator"])
    hp = SERVER_HP.get(hp_key, {})

    def evaluate(weights):
        return calibrated_final_evaluation(weights, n_features, val, test, labels)

    scores = {}
    if c["scope"] == "local":  # one independent model per site, no federation
        per_client, weights_by_client = {}, []
        for i, lab in enumerate(labels):
            w, _ = federate([fit_sets[i]], [lots[i]], "fedavg", {}, c["dp"], c["sigma"],
                            c["rounds"], c["epochs"], c["train_seed"])
            res = calibrated_final_evaluation(w, n_features, [val[i]], [test[i]], [lab])
            per_client[lab] = res["per_client"][lab]
            weights_by_client.append(w)
        result = {"per_client": per_client,
                  "macro": {m: float(np.nanmean([v["global_threshold"][m] for v in per_client.values()]))
                            for m in ("accuracy", "precision", "recall", "f1_score", "auc")},
                  "pooled": {}}
        info = {"history": []}
        for i, lab in enumerate(labels):
            mdl = load_model_from_ndarrays(weights_by_client[i], n_features)
            scores[lab] = (predict_proba(mdl, test[i][0]), predict_proba(mdl, train[i][0]))
    else:
        hist_fn = None
        if c.get("history"):
            def hist_fn(weights):
                # Building the evaluation model draws from the global RNG;
                # restore the state so that recording the per-round curve
                # leaves the training trajectory bit-for-bit unchanged.
                rng_state = torch.get_rng_state()
                mdl = load_model_from_ndarrays(weights, n_features)
                auc = float(np.mean([roc_auc_score(y, predict_proba(mdl, X)) for X, y in site_test]))
                torch.set_rng_state(rng_state)
                return auc
        w, info = federate(fit_sets, lots, c["aggregator"], hp, c["dp"], c["sigma"],
                           c["rounds"], c["epochs"], c["train_seed"], hist_fn)
        result = evaluate(w)
        mdl = load_model_from_ndarrays(w, n_features)
        for i, lab in enumerate(labels):
            scores[lab] = (predict_proba(mdl, test[i][0]), predict_proba(mdl, train[i][0]))
        # Validation AUC per client and per site (used only to develop the
        # headroom policies; the test folds are not consulted for that).
        for i, lab in enumerate(labels):
            Xv, yv = val[i]
            row[f"val_auc_{lab}"] = (float(roc_auc_score(yv, predict_proba(mdl, Xv)))
                                     if len(np.unique(yv)) > 1 else float("nan"))
        row["val_macro_auc_sites"] = float(np.nanmean([
            np.nanmean([row[f"val_auc_{lab}"] for lab in labels if lab.split(".")[0] == f"H{s + 1}"])
            for s in range(len(site_test))]))

    row["macro_auc"] = result["macro"]["auc"]
    for m in ("accuracy", "precision", "recall", "f1_score"):
        row[f"macro_{m}"] = result["macro"][m]
    for m, v in result.get("pooled", {}).items():
        row[f"pooled_{m}"] = v
    for lab, views in result["per_client"].items():
        for m in ("auc", "f1_score", "recall", "precision", "accuracy"):
            row[f"{m}_{lab}"] = views["global_threshold"][m]
        row[f"f1_local_{lab}"] = views["local_threshold"]["f1_score"]
    if k > 1 or ratios:  # site-level AUC is the mean over that site's shards
        row["macro_auc_sites"] = float(np.mean([
            np.mean([row[f"auc_{lab}"] for lab in labels if lab.startswith(f"H{s + 1}.")])
            for s in range(len(site_test))]))
    row["n_train"] = json.dumps(n_train)
    row["lots"] = json.dumps(lots)
    if alloc is not None:
        row["eps"] = json.dumps([round(e, 4) for e in alloc["eps"]])
        row["eps_base"] = json.dumps([round(e, 4) for e in alloc["base_eps"]])
        row["eps_worst"] = max(alloc["eps"])
        row["eps_spread"] = max(alloc["eps"]) / min(alloc["eps"])
        if "eps_executed" in info:
            row["eps_executed"] = json.dumps([round(e, 4) for e in info["eps_executed"]])
            row["eps_max_abs_diff"] = float(np.max(np.abs(np.array(info["eps_executed"]) - np.array(alloc["eps"]))))
            row["sample_rate"] = json.dumps([round(q, 6) for q in info["sample_rate"]])
    row["seconds"] = round(time.time() - t0, 1)

    pred_dir = os.path.join(OUT_DIR, "preds", c["suite"])
    os.makedirs(pred_dir, exist_ok=True)
    arrays = {}
    for i, lab in enumerate(labels):
        arrays[f"test_p_{lab}"], arrays[f"train_p_{lab}"] = scores[lab]
        arrays[f"test_y_{lab}"], arrays[f"train_y_{lab}"] = test[i][1], train[i][1]
        arrays[f"test_agesex_{lab}"] = test[i][0][:, :2]
    if info.get("history"):
        arrays["history_macro_auc"] = np.array(info["history"])
    np.savez_compressed(os.path.join(pred_dir, row["run_id"] + ".npz"), **arrays)
    return row


# ---------------------------------------------------------------------------
# Experiment suites
# ---------------------------------------------------------------------------

def _grid(n_part: int, n_seed: int):
    return [(p, s) for p in range(n_part) for s in range(n_seed)]


def suites() -> Dict[str, List[Dict]]:
    s: Dict[str, List[Dict]] = {}
    pairs = _grid(20, 3)
    # (1) Allocation arms at the adopted DP configuration, paired design.
    s["main"] = [dict(suite="main", partition_seed=p, train_seed=t, allocation=a, history=True)
                 for p, t in pairs for a in ("fixed", "fuzzy", "search", "proportional")]
    # (1b) Same arms with the literature-default server hyperparameters
    #      (Reddi et al.), i.e. no hyperparameter was tuned on patient data.
    s["main_default"] = [dict(suite="main_default", partition_seed=p, train_seed=t, allocation=a,
                              hp="fedadam_default", history=True)
                         for p, t in pairs for a in ("fixed", "fuzzy", "search", "proportional")]
    # (2) Server optimizers under the same DP mechanism (fixed L = 32).
    s["dp_optimizers"] = [dict(suite="dp_optimizers", partition_seed=p, train_seed=t, aggregator=a, hp=h)
                          for p, t in pairs
                          for a, h in (("fedavg", None), ("fedprox", None), ("fedadagrad", None),
                                       ("fedyogi", None), ("fedadam", "fedadam_default"))]
    # (3) No-DP references: federated optimizers, pooled-centralized, local-only.
    nodp = dict(dp=False, epochs=5)
    s["nodp"] = ([dict(suite="nodp", partition_seed=p, train_seed=t, aggregator=a, **nodp)
                  for p, t in pairs for a in ("fedavg", "fedprox", "fedadagrad", "fedyogi", "fedadam")]
                 + [dict(suite="nodp", partition_seed=p, train_seed=t, aggregator="fedavg", scope=sc, **nodp)
                    for p, t in pairs for sc in ("pooled", "local")])
    s["nodp_default"] = [dict(suite="nodp_default", partition_seed=p, train_seed=t, aggregator="fedadam",
                              hp="fedadam_default", **nodp) for p, t in pairs]
    # (4) Stronger privacy targets: sigma calibrated so that the fixed
    #     allocation's worst-case epsilon equals the target.
    s["eps_targets"] = [dict(suite="eps_targets", partition_seed=p, train_seed=t, allocation=a,
                             sigma=sg, tag=f"target_eps_{te}", hp="fedadam_default")
                        for p, t in _grid(10, 3) for te, sg in calibrated_sigmas().items()
                        for a in ("fixed", "fuzzy", "search")]
    # (5) Scalability: every site split into 2 or 4 disjoint clients.
    s["scalability"] = [dict(suite="scalability", partition_seed=p, train_seed=t, allocation=a, shards=k, hp="fedadam_default")
                        for p, t in _grid(10, 3) for k in (2, 4) for a in ("fixed", "fuzzy", "search")]
    # (6) Graded federation: unequal shards so that client sizes span the
    #     range evenly and the medium terms of the controller are exercised.
    s["graded"] = [dict(suite="graded", partition_seed=p, train_seed=t, allocation=a,
                        shard_ratios=GRADED_RATIOS, hp="fedadam_default")
                   for p, t in _grid(10, 3) for a in ("fixed", "fuzzy", "search", "proportional")]
    # (7) Headroom: does spending the larger sites' budget headroom buy
    #     utility? Same partitions and seeds as main_default / graded.
    s["headroom"] = [dict(suite="headroom", partition_seed=p, train_seed=t, allocation=a, hp="fedadam_default")
                     for p, t in pairs for a in ("noharm", "half")]
    s["graded_headroom"] = [dict(suite="graded_headroom", partition_seed=p, train_seed=t, allocation=a,
                                 shard_ratios=GRADED_RATIOS, hp="fedadam_default")
                            for p, t in _grid(10, 3) for a in ("noharm", "half")]
    # (8) No-harm at the size-proportional rule's worst case (second cap),
    #     to pair with the proportional arm already run.
    s["noharm_targets"] = (
        [dict(suite="noharm_targets", partition_seed=p, train_seed=t, allocation="noharm",
              target_eps=4.7181, tag="cap_4.72", hp="fedadam_default") for p, t in pairs]
        + [dict(suite="noharm_targets", partition_seed=p, train_seed=t, allocation="noharm",
                target_eps=7.0817, tag="cap_7.08", shard_ratios=GRADED_RATIOS, hp="fedadam_default")
           for p, t in _grid(10, 3)])
    # (9) Headroom study: when does spending the larger clients' headroom
    #     pay? See experiments/headroom_study.py and its protocol file.
    from experiments.headroom_study import study_configs
    s["headroom_study"] = study_configs(arms=("0", "0.5", "1"))
    s["headroom_study_fuzzy"] = study_configs(arms=("fuzzy",))
    return s


GRADED_RATIOS = [[1 / 3, 2 / 3], [0.2, 0.8], [0.5, 0.5], [1 / 3, 2 / 3]]


def calibrated_sigmas(targets=(1.0, 3.0, 8.0), n_min: int = 211, rounds: int = 100) -> Dict[str, float]:
    """Noise multiplier at which the smallest site (fixed L = 32) spends
    exactly the target epsilon; found by bisection on the accountant."""
    from dpfedadam.rdp_accountant import RDPAccountant

    out = {}
    for target in targets:
        lo, hi = 0.3, 60.0
        for _ in range(50):
            mid = (lo + hi) / 2
            eps = RDPAccountant(noise_multiplier=mid, batch_size=BASE_L, delta=DELTA).compute_epsilon(n_min, rounds, 1)
            lo, hi = (mid, hi) if eps > target else (lo, mid)
        out[str(target)] = round(hi, 4)
    return out


def main() -> None:
    import multiprocessing as mp

    import pandas as pd

    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", type=str, default="main")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()
    all_suites = suites()
    if args.list:
        for name, cfgs in all_suites.items():
            print(f"{name:15s} {len(cfgs)} runs")
        return
    cfgs = all_suites[args.suite][: args.limit]
    os.makedirs(OUT_DIR, exist_ok=True)
    csv_path = os.path.join(OUT_DIR, f"{args.suite}.csv")
    done = set(pd.read_csv(csv_path)["run_id"]) if os.path.exists(csv_path) else set()
    todo = [c for c in cfgs if run_id(c) not in done]
    print(f"[{args.suite}] {len(todo)} runs to do ({len(done)} already done)", flush=True)
    rows = [] if not done else pd.read_csv(csv_path).to_dict("records")
    with mp.get_context("spawn").Pool(args.workers) as pool:
        for j, row in enumerate(pool.imap_unordered(run_one, todo), 1):
            rows.append(row)
            pd.DataFrame(rows).to_csv(csv_path, index=False)
            if j % 10 == 0 or j == len(todo):
                print(f"  {j}/{len(todo)} done; last macro AUC {row['macro_auc']:.3f} ({row['seconds']}s)", flush=True)


if __name__ == "__main__":
    main()
