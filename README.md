# DP-FedAdam — Interpretable and Fairness-Aware Differentially Private Federated Learning for Cardiovascular Risk Prediction

> **Paper:** *Interpretable and Fairness-Aware Differentially Private Federated Learning for Cardiovascular Risk Prediction: A Fuzzy Approach to Practical Privacy Budgets*

> **Note on anonymization:** This repository has been anonymized for double-blind peer review. Author names, institutional affiliations, and any other identifying details have been removed. The repository will be made fully public, with authorship attribution, upon acceptance of the manuscript.

---

## Overview

DP-FedAdam is a federated learning framework for cardiovascular risk prediction that treats the server-side Adam optimizer and the client-side DP-SGD privacy mechanism as a single, jointly designed noise-filtering system. Unlike conventional DP-FL pipelines that compose these components as independent black boxes, the framework makes their interaction explicit in closed form, establishing three properties demanded by the target clinical setting: interpretability, explainability, and ethical governability.

**Core theoretical insight.** The server's first-moment accumulator acts as a temporal low-pass filter over the zero-mean Gaussian DP noise injected by DP-SGD. Because the true gradient signal is persistent near a minimum while the noise is zero-mean, momentum accumulates the signal constructively and progressively cancels the noise. The steady-state variance of the momentum vector is bounded by Var(g_noise) · (1 − β₁) / (1 + β₁), which at β₁ = 0.9 yields an effective 19× variance reduction compared with a stateless aggregator. The argument is stated in a single closed-form equation and can be communicated to a clinician or data protection officer in one paragraph, without a surrogate model.

**Fairness audit.** A per-client Rényi Differential Privacy (RDP) accountant reports individual privacy expenditure for each participating site. Under the primary configuration (σ = 1.1, C = 1.0, L = 32, E = 5, T = 100), per-client ε varies with dataset-size imbalance — smaller sites pay a higher privacy cost via subsampling amplification for the same nominal (σ, C). See the Results section (pending re-run, see note below) for current per-client figures. Making the spread explicit converts a cryptographically honest but ethically one-sided single-number ε into an auditable fairness table, and a Mamdani fuzzy controller (`fedcvr/fuzzy_fairness.py`) can actively narrow it by recommending a per-client batch-size adjustment.

**Ethical deployment constraint.** A recall-versus-ε governance curve, grounded in the 2021 ESC and 2019 ACC/AHA cardiovascular prevention guidelines, shows that recall is the clinical metric most sensitive to privacy tightening: across the full privacy range, it falls 23.3 percentage points (78.7% → 55.4%), while accuracy declines only 3.6 points. The curve serves as an explainable governance tool that ethics committees and data protection officers can use to negotiate a clinically safe lower bound on the deployable privacy budget.

```
                 ┌─────────────────────────────────────────────────────┐
                 │               Federated Server                       │
                 │   Adaptive Optimizer  (β₁=0.9, β₂=0.999)           │
                 │   m_t = β₁·m_{t-1} + (1−β₁)·ḡ_t   ← low-pass     │
                 │   w_{t+1} = w_t − η · m̂_t / (√v̂_t + τ)           │
                 │   RDP Accountant  →  per-client ε audit             │
                 └──────┬────────────┬────────────┬────────────┬───────┘
                        │  w_t       │            │            │
           ┌────────────▼──┐   ┌─────▼──────┐   ┌─▼──────────┐   ...
           │  H1 Z-Alizadeh│   │  H2 IEEE-  │   │ H3 UCI     │
           │  Sani, n=303  │   │  CHD 1,190 │   │ Cleveland  │
           │               │   │            │   │ 920        │
           │  Local SGD    │   │  Local SGD │   │ Local SGD  │
           │  DP-SGD       │   │  DP-SGD    │   │ DP-SGD     │
           │  clip C=1.0   │   │  clip C=1.0│   │ clip C=1.0 │
           │  + N(0,σ²C²I) │   │  + noise   │   │ + noise    │
           └───────────────┘   └────────────┘   └────────────┘
                   ↑ DP-protected pseudo-gradient g̃_k sent to server ↑
```

---

## Repository Structure

```
FL-framework-cardiovascular-risk-prediction/
├── fedcvr/   # Core Python package
│   ├── __init__.py
│   ├── model.py                   # DNN: Input(22)→64→ReLU→32→ReLU→1→Sigmoid
│   ├── client.py                  # DP-SGD client via Opacus
│   ├── strategy.py                # DP-FedAdam server (Adam aggregation + RDP)
│   ├── baselines.py               # FedAdagrad, FedYogi, FedCluster baselines
│   ├── rdp_accountant.py          # Per-client Rényi-DP accounting
│   ├── fuzzy_fairness.py          # Mamdani fuzzy controller: rebalances per-client batch size
│   ├── evaluation.py              # Threshold-calibrated eval: pooled / macro / per-client views
│   └── data_utils.py              # Harmonization, preprocessing, SMOTE, clustering
│
├── experiments/
│   ├── run_hpo.py                     # Optuna (TPE+pruning) server-hyperparameter search
│   ├── run_comparison.py              # 6-method benchmark (--no_dp for the no-privacy stage)
│   ├── run_dp_sensitivity.py          # Sigma sensitivity sweep, fixed batch size
│   ├── run_dp_sensitivity_fuzzy.py    # Adopted DP config, fuzzy per-client batch size
│   └── run_privacy_budget_test.py     # Adopted DP config, fixed batch size (fuzzy baseline)
│
├── data/
│   ├── README.md                  # Download instructions for all 4 datasets
│   ├── convert_cleveland.py       # UCI .data → .csv converter (H3, 4 sites combined)
│   ├── convert_cardio_sulianova.py  # sulianova → .csv converter (H4)
│   └── convert_z_alizadeh_sani.py   # Z-Alizadeh Sani → .csv converter (H1)
│
├── results/                       # Auto-created output directory
│   ├── centralized_baseline.py    # Non-federated, non-DP classical-ML baseline (RF/XGBoost/LightGBM/CatBoost)
│   ├── bootstrap_auc_ci.py        # Bootstrap 95% CI for the centralized-baseline AUC
│   ├── per_dataset_baseline.py    # Non-federated, DP-SGD-DNN solo-per-client baseline
│   └── resume_pipeline.sh         # Resume-aware N=5-seed benchmark runner
├── requirements.txt
└── .gitignore
```
---

## Method

### Model Architecture

A lightweight deep neural network (DNN) for binary cardiovascular risk classification:

```
Input(22) → Linear(64) → ReLU → Linear(32) → ReLU → Linear(1) → Sigmoid
```

The input layer accepts 22 harmonized features: the 6-feature schema shared by all four clients, plus two blocks of client-specific extra features (always zero for the clients that don't natively have them) — see "Datasets" below. Training uses the Binary Cross-Entropy (BCE) loss. The local optimizer at each client is vanilla SGD (no client-side momentum, no learning rate schedule), with a fixed learning rate η_c = 0.01 and five local epochs per round. This deliberate choice means that all denoising is attributed to the server, isolating the low-pass filter effect for interpretability.

### Client Update — DP-SGD

Each selected client trains locally using DP-SGD enforced by the Opacus library. For every mini-batch B:

1. **Per-sample gradient clipping** bounds the sensitivity:

```
ḡ_i = g_i / max(1, ‖g_i‖₂ / C)
```

2. **Gaussian noise injection** enforces (ε, δ)-DP:

```
g̃_B = Σ_{i∈B} ḡ_i + N(0, σ²C²I)
```

The resulting pseudo-gradient Δw_k = w − w_k is returned to the server. No client-side momentum is used, ensuring that the server's Adam accumulator is the sole source of noise attenuation.

Default DP parameters: C = 1.0, L = 32 (base batch size, fuzzy-rebalanced per client under DP - see "Fuzzy Fairness Controller"). Sensitivity sweep: σ ∈ {0.8, 1.1, 1.5}. Adopted configuration: σ = 2.5, local_epochs = 1 (see Results §5).

### Server Aggregation — DP-FedAdam

The server maintains first- and second-moment vectors across rounds and applies bias-corrected Adam-style updates. The critical insight is that m_t functions as a temporal low-pass filter on the zero-mean DP noise:

```
ḡ_t  = (1/|S_t|) · Σ_{k∈S_t} Δw_k          # aggregate pseudo-gradients
m_t  = β₁·m_{t-1} + (1−β₁)·ḡ_t             # low-pass filter on DP noise
v_t  = β₂·v_{t-1} + (1−β₂)·ḡ_t²            # element-wise
m̂_t  = m_t / (1 − β₁ᵗ)                      # bias correction
v̂_t  = v_t / (1 − β₂ᵗ)
w_{t+1} = w_t − η · m̂_t / (√v̂_t + τ)       # adaptive update
```

Default server parameters: η = 0.1, β₁ = 0.9, β₂ = 0.999, τ = 1e-3.

**Variance reduction bound.** Unrolling the recursion under stationarity gives:

```
Var(m_t) ≤ Var(g_noise) · (1 − β₁) / (1 + β₁)
```

At β₁ = 0.9, this is a 19× reduction relative to a stateless aggregator. The bound precedes the measurement and predicts its direction, making the performance advantage mechanistically transparent.

### Privacy Budget Accounting

The framework integrates a per-client RDP accountant. Because each mini-batch subsamples a fraction q = L / n_{train,k} of client k's data, and dataset sizes differ, per-client ε varies under the same nominal (σ, C). The globally reported ε is always the worst-case (maximum) across sites.

```python
# Example: query per-client privacy expenditure after training
accountant = RDPAccountant(noise_multiplier=1.1, max_grad_norm=1.0, batch_size=32, delta=1e-5)
per_client_eps = accountant.audit_all_clients(
    n_train_per_client=[<see data_utils output for current pre-SMOTE sizes>],
    n_rounds=100, local_epochs=5,
    client_labels=["H1", "H2", "H3", "H4"],
)
```

### Fuzzy Fairness Controller

Because the DP-SGD subsampling rate q = L / n_train differs by client (same batch size L, different dataset sizes), smaller clients accrue a higher effective epsilon than larger ones under an identical nominal (σ, C) — a fairness problem across institutions with different amounts of data. `fedcvr/fuzzy_fairness.py` implements a Mamdani fuzzy controller that recommends a per-client batch-size multiplier from two linguistic inputs — dataset size (small/medium/large) and current epsilon (low/moderate/high) — via a 3×3 rule base, with a single output (decrease/keep/increase batch size). A small, high-epsilon client is pushed toward a smaller batch (lowering its epsilon); a large, low-epsilon client toward a larger one (spending its privacy headroom on faster convergence). `FuzzyFairnessController.rebalance()` computes the baseline and fuzzy-adjusted epsilon per client via `RDPAccountant`; `experiments/run_dp_sensitivity_fuzzy.py` applies the recommended batch sizes to real training. See Results §4 for the measured effect on both the epsilon spread and utility.

### Evaluation Protocol

Flower's per-round `client.evaluate()` uses a fixed 0.5 threshold, fine for tracking the round-by-round convergence curve but not for final reported metrics — DP-SGD's per-sample gradient clipping can leave a 0.5-cutoff model precision-heavy and recall-starved even when AUC is healthy. `fedcvr/evaluation.py` instead runs a centralized pass after training: it selects the F1-maximizing threshold on the validation fold only, then reports metrics on the held-out test fold at that threshold. Four views are computed — **pooled** (micro-average over the concatenated test set, skewed toward whichever client contributes the most test volume), **macro** (unweighted per-client mean at the same global threshold — every institution counts equally, but a threshold tuned for the federation-wide prevalence can still miscalibrate an individual low-prevalence site), **macro, local threshold** (each client calibrates its own operating point on its own validation fold — the realistic deployment scenario), and **per-client** (both views, for the site-by-site breakdown table). Macro is the primary metric reported in Results, for the reasons above.

### Datasets

Four publicly available cardiac datasets simulate a genuinely heterogeneous, non-IID federation. No two clients share identical data-generating processes, and — as of 2026-08-14 — no two share the same country or institution either.

| Client | Dataset | Samples | Region / Institution |
|--------|---------|---------|----------------|
| H1 | Z-Alizadeh Sani | 303 | Shaheed Rajaei Cardiovascular Center, Tehran, Iran |
| H2 | IEEE Comprehensive HD (heart_statlog_cleveland_hungary_final) | 1,190 | Aggregated (IEEE Dataport) |
| H3 | UCI Heart Disease (Cleveland+Hungarian+Switzerland+VA, combined) | 920 | Multicenter (UCI Archive) |
| H4 | Cardiovascular Disease dataset (sulianova), subsampled | 1,025 | Ambulatory checkup cohort |

**Total:** 3,438 records. H1 was originally the Framingham Heart Study; its
10-year-CHD outcome was shown (`results/centralized_baseline.py`, four
classical ML algorithms, no DP/federation) to cap at AUC≈0.65 regardless of
using 6 or its full 15 native features — a real ceiling for that specific
prospective-risk task, not a bug. H1 was substituted for Z-Alizadeh Sani
(see `data/README.md` for the full replacement history and rationale).
A fifth client, FIC Pakistan (368 samples, `Mortality` target), was tried
and then **removed (2026-08-14)**: its target measured in-hospital death
among already-admitted patients rather than disease presence, so the
classic risk factors correlated with target in the *opposite* direction
from every other client — a federated model trained across the remaining
clients scored AUC < 0.5 on it. See `data/README.md` for the full record.

All four clients share **six common cardiovascular risk features**: age, systolic blood pressure, diastolic blood pressure, serum cholesterol, fasting blood glucose, and sex. Two clients contribute further client-specific columns on top of that shared base (always zero for the other clients) — H4's ordinal cholesterol/glucose categories one-hot encoded (6 columns), and H1's top-10 native diagnostic features by predictive importance (10 columns) — bringing the full harmonized schema to 22 features. See `data/README.md` for the authoritative column-by-column breakdown.

**Preprocessing pipeline (per client, in this order — split happens early so every later statistic is fit on the training fold only):**
1. Target binarisation
2. Stratified 70/10/20 train/validation/test split (`random_state=42`)
3. IQR-based outlier capping (1.5-IQR rule; bounds from the training fold only)
4. Median imputation for missing values (medians from the training fold only)
5. SMOTE oversampling within the training fold (class imbalance correction)
6. MinMaxScaler normalization to [0, 1] (fit on the pre-SMOTE training fold, applied to all folds) — chosen over standardization to match the fixed DP-SGD clipping norm `C` shared by every client and to leave the one-hot columns undistorted

---

## Installation

```bash
# Clone the repository
# (Full URL provided in the manuscript after acceptance; anonymized for peer review)
git clone <repository-url>
cd FL-framework-cardiovascular-risk-prediction

# Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
# .venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt
```

> **Note:** Opacus requires PyTorch ≥ 2.0. All experiments were run on a CPU (Intel Xeon, 4 vCores, 12–16 GiB RAM). GPU is optional.

---

## Quickstart

### 1. Download the datasets

Follow the instructions in `data/README.md` and place the four CSV files in the `data/` directory (run the three `data/convert_*.py` scripts for H1/H3/H4 first).

### 2. (Optional) Tune server hyperparameters with Optuna

```bash
python -m experiments.run_hpo --data_dir data --n_trials 40 --rounds_per_trial 30 --out_dir results
```

Writes `results/best_hyperparameters.json`. For the DP-adopted configuration specifically (Results §6 shows this matters - hyperparameters tuned without DP aren't necessarily right once real per-step noise is in the loop):

```bash
python -m experiments.run_hpo --data_dir data --n_trials 40 --rounds_per_trial 30 \
    --noise_multiplier 2.5 --local_epochs 1 --out_dir results --out_name best_hyperparameters_dp.json
```

### 3. Run the 6-method benchmark (100 rounds, all under DP)

```bash
python -m experiments.run_comparison \
    --data_dir  data \
    --rounds    100 \
    --out_dir   results \
    --hyperparams results/best_hyperparameters.json
```

Runs DP-FedAdam, FedAvg, FedProx, FedCluster, FedAdagrad, and FedYogi under the same configuration, so that observed differences are attributable solely to server-side aggregation design. Pass `--no_dp` for the no-privacy stage, or nothing for DP-SGD (`fedcvr.client`'s default DP config). Saves round-level metrics plus threshold-calibrated pooled/macro/per-client final metrics (`results/calibrated_final_metrics.csv`, `results/calibrated_per_client_metrics.csv`). For N=5-seed statistics with Welch's t-test, use `results/resume_pipeline.sh` (resume-aware: skips seeds already completed).

### 4. Run the privacy-utility trade-off analysis

```bash
python -m experiments.run_dp_sensitivity --data_dir data --rounds 100 --out_dir results \
    --hyperparams results/best_hyperparameters.json
```

Evaluates the proposed method alone across the no-DP baseline and three DP regimes (σ ∈ {0.8, 1.1, 1.5}, fixed batch size). Then, at the adopted configuration (σ=2.5, local_epochs=1 - see Results §5):

```bash
python -m experiments.run_privacy_budget_test --data_dir data --rounds 100 --out_dir results \
    --hyperparams results/best_hyperparameters.json         # fixed batch size
python -m experiments.run_dp_sensitivity_fuzzy --data_dir data --rounds 100 --out_dir results \
    --hyperparams results/best_hyperparameters.json         # fuzzy per-client batch size
```

### 5. (Optional) Sanity-check baselines

```bash
python results/centralized_baseline.py   # non-federated, non-DP, classical ML per dataset
python results/per_dataset_baseline.py   # non-federated, DP-SGD DNN per dataset
```

---

## Results

*(100 rounds. §1, §3, §4, §5 are single-seed (seed=42), server hyperparameters from `results/best_hyperparameters.json` (tuned without DP). §2 is N=10-seed; §6 is N=10-seed using `results/best_hyperparameters_dp.json` (tuned specifically for the adopted DP config, sigma=2.5/local_epochs=1) - see §6 for why a separate hyperparameter set matters. Welch's t-test throughout. Numbers regenerated 2026-08-14/15 against the corrected 4-client pipeline.)*

### 1. Centralized baseline (non-federated, non-DP ceiling)

Average of RandomForest/XGBoost/LightGBM/CatBoost, per client (`results/centralized_baseline.py`):

| Client | AUC | F1 |
|---|---|---|
| H1: Z-Alizadeh Sani | 0.859 | 0.891 |
| H2: IEEE-CHD | 0.811 | 0.758 |
| H3: UCI Cleveland | 0.736 | 0.736 |
| H4: Cardiovascular Disease (sulianova) | 0.828 | 0.752 |

### 2. Federated, no DP — cost of federation alone

FedCVR vs. 5 baselines (FedAvg, FedProx, FedCluster, FedAdagrad, FedYogi), macro-averaged across clients, N=10 seeds (`results/statistical_tests_summary.csv`), Welch's t-test on macro AUC vs. FedCVR:

| Strategy | Macro AUC (mean ± std) | Macro F1 (mean ± std) | p vs. FedCVR |
|---|---|---|---|
| FedCVR (ours) | 0.803 ± 0.009 | 0.777 ± 0.015 | — |
| FedAdagrad | 0.810 ± 0.005 | 0.784 ± 0.006 | 0.072 |
| FedAvg | 0.801 ± 0.003 | 0.773 ± 0.004 | 0.472 |
| FedProx | 0.801 ± 0.003 | 0.771 ± 0.004 | 0.466 |
| FedYogi | 0.800 ± 0.012 | 0.775 ± 0.009 | 0.470 |
| FedCluster | 0.756 ± 0.020 | 0.749 ± 0.015 | **<0.001** |

At N=10 seeds, FedCVR is still **not statistically distinguishable** from FedAvg, FedProx, or FedYogi (all p>0.4) — competitive with, not clearly superior to, standard FedOpt baselines on this benchmark. FedAdagrad's edge tightened toward significance with more seeds (p=0.072, was 0.215 at N=5) but hasn't crossed the 0.05 threshold. FedCVR **is** significantly better than FedCluster, which shows degenerate behavior on some seeds (near-constant-positive predictions, recall≈1.0 with accuracy≈0.6).

FedCVR vs. the centralized ceiling, per client (single seed): the federation cost is small and sometimes *negative* (FedCVR AUC beats centralized by +0.019 on H1 and +0.062 on H3; trails by −0.014 to −0.030 on H2/H4).

A single-seed convergence check (100 vs. 500 rounds) confirmed the model plateaus by round ~20–30 with no further gain (calibrated macro AUC 0.818 at 100 rounds vs. 0.814 at 500) — 100 rounds is sufficient for all reported results.

### 3. DP sensitivity, fixed batch size (L=32 for every client)

| Regime | Macro AUC | Macro F1 |
|---|---|---|
| No DP | 0.799 | 0.776 |
| σ=0.8 | 0.807 | 0.785 |
| σ=1.1 | 0.803 | 0.774 |
| σ=1.5 | 0.798 | 0.778 |

**Epsilon caveat, resolved 2026-08-14:** an earlier draft's Table 3 claimed ε≈4.8–13.4 for σ=1.1; recomputing against the actual 100-round protocol gives ε≈13.6–91 across clients (fixed batch=32, local_epochs=5). Sweeping the RDP composition over round count shows the original 4.8–13.4 range corresponds almost exactly to ~14 rounds of composition, not 100 — an internal inconsistency in the original draft (the privacy audit was seemingly computed before the 100-round protocol was finalized), not a bug in the current accounting code. See §5 for the resolution.

### 4. Fuzzy fairness rebalancing (`experiments/run_dp_sensitivity_fuzzy.py`)

Per-client batch size recommended by the Mamdani fuzzy controller (`fedcvr/fuzzy_fairness.py`) from (dataset size, current epsilon), vs. fixed L=32, at 100 rounds:

| Regime | ε spread, fixed | ε spread, fuzzy | Macro AUC, fixed | Macro AUC, fuzzy |
|---|---|---|---|---|
| σ=0.8 | 2.48x | 2.43x | 0.807 | 0.816 |
| σ=1.1 | 2.57x | 2.46x | 0.803 | 0.806 |
| σ=1.5 | 2.53x | 2.40x | 0.798 | 0.814 |

Fuzzy rebalancing narrows the per-client privacy-fairness gap in every regime; this single-seed run also showed flat-to-better AUC under fuzzy rebalancing. **The properly-validated utility result is in §6** (N=10 seeds, DP-specific hyperparameters): fuzzy's AUC advantage is statistically significant there. The fairness effect (narrower ε spread) is deterministic RDP-accounting math and stands regardless of seed or hyperparameters.

### 5. Single-digit-epsilon feasibility (`experiments/run_privacy_budget_test.py`)

Resolving the §3 epsilon caveat: is a clinically-defensible (single-digit) ε reachable at the real 100-round protocol? Two configurations tested, both keeping all 100 rounds (shortening the protocol was rejected — the model needs ~20–30 rounds to converge, see §2):

| Configuration | ε (worst-case client, H1) | Macro AUC | Macro F1 |
|---|---|---|---|
| σ=5.0, local_epochs=5 (noise only) | 9.56 | 0.756 | 0.773 |
| **σ=2.5, local_epochs=1 (recommended)** | **8.82** | **0.772** | 0.760 |

**Adopted configuration: σ=2.5, local_epochs=1.** It reaches a tighter epsilon than the noise-only alternative while costing only 0.027 AUC relative to the no-DP ceiling (0.799→0.772) — spending fewer local SGD steps per round buys more privacy per unit of utility lost than raising noise alone.

### 6. N=10-seed validation: fixed vs. fuzzy batch size, with hyperparameters re-tuned for DP

A first N=5-seed pass reused `best_hyperparameters.json` (tuned *without* DP) for the adopted DP configuration and found no significant fuzzy utility benefit (p=0.084, point estimate trending slightly unfavorable). Since the DP-FedAdam server optimizer is directly interacting with per-step Gaussian noise under this configuration, hyperparameters tuned in a noise-free setting aren't necessarily right for it. Re-ran Optuna specifically under sigma=2.5/local_epochs=1 (`experiments/run_hpo.py --noise_multiplier 2.5 --local_epochs 1`, 40 trials) → `results/best_hyperparameters_dp.json` (eta=0.168, beta_1=0.782, beta_2=0.961, tau=0.00128 - notably different from the no-DP-tuned values), then re-ran both variants at N=10 seeds:

| Variant | Macro AUC (mean ± std) | Macro F1 (mean ± std) |
|---|---|---|
| Fixed (L=32 for every client) | 0.764 ± 0.011 | 0.751 ± 0.014 |
| Fuzzy (per-client L) | 0.776 ± 0.012 | 0.765 ± 0.018 |
| Welch p-value | **0.044** | 0.100 |

With DP-specific hyperparameters and N=10 seeds, fuzzy rebalancing's AUC advantage **is** statistically significant (p=0.044). F1 trends the same direction but doesn't cross 0.05 (p=0.100). This reverses the earlier N=5 reading — hyperparameters tuned for the actual DP condition change the outcome, not just precision from more seeds.

Two things worth disclosing rather than glossing over: (1) both variants' absolute AUC dropped relative to the earlier N=5 run that reused no-DP-tuned hyperparameters (fixed: 0.788→0.764) - the new hyperparameters aren't uniformly better, they happen to synergize specifically with per-client batch sizing; (2) Optuna's search uses a 30-round proxy per trial (`--rounds_per_trial 30`) as a tractability compromise, which does not always perfectly predict full 100-round behavior.

---

## Hyperparameter Reference

| Parameter | Description | Value |
|-----------|-------------|-------|
| `num_clients` | Participating hospitals | 4 |
| `communication_rounds` | Benchmark (plateaus by ~round 20-30, no gain past 100 - see Results §2) | 100 |
| `random_state` | Reproducibility seed | 42 |
| `local_epochs` | Local epochs per round (5 for the no-DP/fixed-batch benchmark; **1** in the adopted single-digit-epsilon DP configuration - see Results §5) | 5 (1 under adopted DP config) |
| `local_optimizer` | Client-side optimizer | SGD (no momentum) |
| `client_lr` | Client learning rate η_c | 0.01 |
| `batch_size` | Mini-batch size L (Opacus); fuzzy-rebalanced per client under DP (Results §4) | 32 (base) |
| `clip_norm` C | Per-sample gradient clip norm | 1.0 |
| `sigma` σ | DP noise multiplier (sensitivity levels tested: 0.8/1.1/1.5; **2.5 adopted** for the single-digit-epsilon configuration) | 2.5 (adopted) |
| `server_lr` η | Server-side learning rate (Optuna-tuned, `results/best_hyperparameters.json`) | 0.080 |
| `beta1` β₁ | First moment decay (Optuna-tuned) | 0.733 |
| `beta2` β₂ | Second moment decay (Optuna-tuned) | 0.910 |
| `tau` τ | Numerical stability constant (Optuna-tuned) | 0.00178 |
| `delta` δ | DP delta | 1e-5 |

---

## License

This project is released under the **MIT License**. See `LICENSE` for details.

The datasets used in this work are subject to their own respective licenses. Please refer to `data/README.md` for original sources and terms of use.
