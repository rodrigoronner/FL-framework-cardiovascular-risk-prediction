# DP-FedAdam — Interpretable Differentially Private Federated Learning for Cardiovascular Risk Prediction

> **Paper:** *Interpretable Differentially Private Federated Learning for Cardiovascular Risk Prediction: Mechanistic Transparency and Fairness Auditing*

> **Note on anonymization:** This repository has been anonymized for double-blind peer review. Author names, institutional affiliations, and any other identifying details have been removed. The repository will be made fully public, with authorship attribution, upon acceptance of the manuscript.

---

## Overview

DP-FedAdam is a federated learning framework for cardiovascular risk prediction that treats the server-side Adam optimizer and the client-side DP-SGD privacy mechanism as a single, jointly designed noise-filtering system. Unlike conventional DP-FL pipelines that compose these components as independent black boxes, the framework makes their interaction explicit in closed form, establishing three properties demanded by the target clinical setting: interpretability, explainability, and ethical governability.

**Core theoretical insight.** The server's first-moment accumulator acts as a temporal low-pass filter over the zero-mean Gaussian DP noise injected by DP-SGD. Because the true gradient signal is persistent near a minimum while the noise is zero-mean, momentum accumulates the signal constructively and progressively cancels the noise. The steady-state variance of the momentum vector is bounded by Var(g_noise) · (1 − β₁) / (1 + β₁), which at β₁ = 0.9 yields an effective 19× variance reduction compared with a stateless aggregator. The argument is stated in a single closed-form equation and can be communicated to a clinician or data protection officer in one paragraph, without a surrogate model.

**Fairness audit.** A per-client Rényi Differential Privacy (RDP) accountant reports individual privacy expenditure for each participating site. Under the primary configuration (σ = 1.1, C = 1.0, L = 32, E = 5, T = 100), per-client ε ranges from ≈ 4.8 (H1 Framingham, n = 2,967) to ≈ 13.4 (H3 Cleveland, n = 644), a 2.8× spread driven entirely by dataset-size imbalance. Making the spread explicit converts a cryptographically honest but ethically one-sided single-number ε into an auditable fairness table.

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
           │  H1 Framingham│   │  H2 IEEE-  │   │ H3 UCI     │
           │  4,238 records│   │  CHD 1,190 │   │ Cleveland  │
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
│   ├── model.py                   # DNN: Input(6)→64→ReLU→32→ReLU→1→Sigmoid
│   ├── client.py                  # DP-SGD client via Opacus
│   ├── strategy.py                # DP-FedAdam server (Adam aggregation + RDP)
│   ├── baselines.py               # FedAdagrad, FedYogi, FedCluster baselines
│   ├── rdp_accountant.py          # Per-client Rényi-DP accounting
│   └── data_utils.py              # Harmonization, preprocessing, SMOTE, clustering
│
├── experiments/
│   ├── run_comparison.py          # 6-method benchmark, all under DP, 100 rounds
│   └── run_dp_sensitivity.py      # Privacy-utility trade-off, 5,000 rounds
│
├── data/
│   ├── README.md                  # Download instructions for all 5 datasets
│   └── convert_cleveland.py       # UCI .data → .csv converter
│
├── results/                       # Auto-created output directory
├── requirements.txt
└── .gitignore
```
---

## Method

### Model Architecture

A lightweight deep neural network (DNN) for binary cardiovascular risk classification:

```
Input(6) → Linear(64) → ReLU → Linear(32) → ReLU → Linear(1) → Sigmoid
```

The input layer accepts six harmonized clinical features. Training uses the Binary Cross-Entropy (BCE) loss. The local optimizer at each client is vanilla SGD (no client-side momentum, no learning rate schedule), with a fixed learning rate η_c = 0.01 and five local epochs per round. This deliberate choice means that all denoising is attributed to the server, isolating the low-pass filter effect for interpretability.

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

Default DP parameters: C = 1.0, σ ∈ {0.8, 1.1, 1.5}, L = 32 (batch size).

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
    n_train_per_client=[2967, 833, 644, 700, 718],
    n_rounds=100, local_epochs=5,
    client_labels=["H1", "H2", "H3", "H4", "H5"],
)
# Returns: {'H1': 4.8, 'H2': 11.2, 'H3': 13.4, 'H4': 12.1, 'H5': 11.9}
```

### Datasets

Five publicly available cardiac EHR datasets simulate a genuinely heterogeneous, non-IID federation. No two clients share identical data-generating processes.

| Client | Dataset | Samples | Features (raw) | Region |
|--------|---------|---------|----------------|--------|
| H1 | Framingham Heart Study | 4,238 | 16 | Longitudinal, USA |
| H2 | IEEE Comprehensive HD | 1,190 | 28 | Aggregated (IEEE Dataport) |
| H3 | UCI Heart Disease (Cleveland) | 920 | 14 | Multicenter (UCI Archive) |
| H4 | FIC Pakistan (Faisalabad) | 1,000 | 19 | Clinical, Pakistan |
| H5 | Kaggle Heart Prediction | 1,025 | 14 | Cleaned clinical |

**Total:** 8,373 records. Local dataset sizes span a 4.6-fold range (920–4,238), creating the imbalance that drives the per-client ε spread documented in the fairness audit.

All datasets are harmonized to **six common cardiovascular risk features**: age, systolic blood pressure, diastolic blood pressure, serum cholesterol, fasting blood glucose, and sex. Feature selection was based on cross-dataset availability, regardless of naming convention; no features were synthesized or imputed across sites.

**Preprocessing pipeline (per client):**
1. IQR-based outlier capping (1.5-IQR rule, continuous features)
2. Median imputation for missing values
3. Stratified 70/10/20 train/validation/test split
4. SMOTE oversampling within the training fold (class imbalance correction)
5. StandardScaler normalization (fit on the pre-SMOTE training fold, applied to all folds)

**Splits:** 70% train / 10% validation / 20% test, stratified, `random_state=42`.

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

Follow the instructions in `data/README.md` and place the five CSV files in the `data/` directory.

### 2. Run the 6-method benchmark (100 rounds, all under DP)

```bash
python -m experiments.run_comparison \
    --data_dir  data \
    --rounds    100 \
    --out_dir   results
```

Runs DP-FedAdam, FedAvg, FedProx, FedCluster, FedAdagrad, and FedYogi, all under the same DP-SGD configuration (σ = 1.1, C = 1.0, ε ≈ 13.4, δ = 1e-5), so that observed differences are attributable solely to server-side aggregation design. Saves:
- `results/comparison_metrics.csv`
- `results/comparison_plot.png`

### 3. Run the privacy-utility trade-off analysis (5,000 rounds)

```bash
python -m experiments.run_dp_sensitivity \
    --data_dir  data \
    --rounds    5000 \
    --out_dir   results
```

Evaluates the proposed method alone across the no-DP baseline and three DP regimes (σ ∈ {0.8, 1.1, 1.5}, defined in `DP_SCENARIOS` inside the script). Saves:
- `results/dp_sensitivity_metrics.csv`
- `results/dp_sensitivity_plot.png` — the recall-versus-ε governance curve

---

## Results

> The tables below reproduce the paper's published results (N = 5 independent seeds, averaged externally). `run_comparison.py` and `run_dp_sensitivity.py` as shipped here run a single seed per invocation (there is no `--n_runs`/seed CLI flag); reproducing the mean ± std figures requires editing `random_state` in the source and rerunning multiple times.

### Benchmark — All Methods Under DP (ε ≈ 13.4, σ = 1.1, N = 5 runs)

| Method | Type | Accuracy | Recall | F1 | AUC | p-value vs proposed |
|--------|------|----------|--------|----|-----|---------------------|
| FedAvg | Stateless | 0.85 ± 0.02 | 0.65 ± 0.04 | 0.67 ± 0.04 | 0.88 ± 0.02 | < 0.001 |
| FedProx | Regularisation | 0.87 ± 0.01 | 0.68 ± 0.02 | 0.70 ± 0.03 | 0.90 ± 0.01 | < 0.001 |
| FedCluster | Clustering | 0.88 ± 0.02 | 0.70 ± 0.03 | 0.71 ± 0.03 | 0.91 ± 0.02 | < 0.001 |
| FedAdagrad | Adaptive | 0.89 ± 0.02 | 0.72 ± 0.03 | 0.73 ± 0.03 | 0.92 ± 0.01 | < 0.001 |
| FedYogi | Adaptive | 0.91 ± 0.01 | 0.76 ± 0.02 | 0.75 ± 0.02 | 0.94 ± 0.01 | 0.014 |
| **DP-FedAdam (proposed)** | **Adaptive** | **0.92 ± 0.01** | **0.78 ± 0.02** | **0.77 ± 0.02** | **0.96 ± 0.01** | — |

Statistical comparisons use Welch's two-tailed t-test across N = 5 independent runs. With N = 5 the statistical power is limited; results should be interpreted as indicative trends.

### Privacy-Utility Trade-off

| Scenario | σ | ε (approx.) | Accuracy (%) | Precision (%) | Recall (%) | F1 (%) |
|----------|---|-------------|-------------|---------------|-----------|--------|
| No DP (Baseline) | — | ∞ | 94.6 | 89.3 | 78.7 | 83.3 |
| Low privacy | 0.8 | 80.5 | 93.5 | 88.1 | 75.1 | 81.0 |
| Moderate privacy | 1.1 | 13.4 | 92.8 | 87.5 | 68.3 | 76.6 |
| High privacy | 1.5 | 6.6 | 91.0 | 86.2 | 55.4 | 67.5 |

**Key finding:** Across the full privacy range, recall declines 23.3 percentage points (78.7% → 55.4%) while accuracy falls only 3.6 points (94.6% → 91.0%). The steep recall drop below ε ≈ 13.4 defines the practical ethical deployment boundary for this application, as grounded in the 2021 ESC and 2019 ACC/AHA cardiovascular prevention guidelines.

### Per-Client Fairness Audit (σ = 1.1, δ = 1e-5)

| Client | Dataset | n_train | q (sampling rate) | ε |
|--------|---------|---------|-------------------|---|
| H1 | Framingham | 2,967 | 0.011 | ≈ 4.8 |
| H2 | IEEE-CHD | 833 | 0.038 | ≈ 11.2 |
| H3 | UCI Cleveland | 644 | 0.050 | ≈ 13.4 |
| H4 | FIC Pakistan | 700 | 0.046 | ≈ 12.1 |
| H5 | Kaggle HD | 718 | 0.045 | ≈ 11.9 |

The 2.8× spread between H1 and H3 is driven entirely by dataset-size imbalance. Smaller sites pay a disproportionately higher privacy cost under the same nominal (σ, C). The globally reported ε is always the worst-case (H3 = 13.4).

---

## Hyperparameter Reference

| Parameter | Description | Value |
|-----------|-------------|-------|
| `num_clients` | Participating hospitals | 5 |
| `communication_rounds` | Benchmark / longitudinal | 100 / 5,000 |
| `random_state` | Reproducibility seed | 42 |
| `local_epochs` | Local epochs per round | 5 |
| `local_optimizer` | Client-side optimizer | SGD (no momentum) |
| `client_lr` | Client learning rate η_c | 0.01 |
| `batch_size` | Mini-batch size L (Opacus) | 32 |
| `clip_norm` C | Per-sample gradient clip norm | 1.0 |
| `sigma` σ | DP noise multiplier (levels tested) | {0.8, 1.1, 1.5} |
| `server_lr` η | Server-side learning rate | 0.1 |
| `beta1` β₁ | First moment decay | 0.9 |
| `beta2` β₂ | Second moment decay | 0.999 |
| `tau` τ | Numerical stability constant | 1e-3 |
| `delta` δ | DP delta | 1e-5 |

---

## License

This project is released under the **MIT License**. See `LICENSE` for details.

The datasets used in this work are subject to their own respective licenses. Please refer to `data/README.md` for original sources and terms of use.
