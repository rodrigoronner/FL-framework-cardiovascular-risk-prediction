# FedCVR – Federated Cardiovascular Risk Prediction

> **Paper:** *Overcoming Clinical Data Heterogeneity: A Secure Federated Framework for Cardiovascular Risk Prediction*
> **Journal:** iSys – Revista Brasileira de Sistemas de Informação (Brazilian Journal of Information Systems), 2025
> **Authors:** Rodrigo Tertulino · Ricardo Almeida · Laércio Alencar
> **Institution:** IFRN – Federal Institute of Education, Science and Technology of Rio Grande do Norte

---

## Overview

FedCVR is a federated learning framework designed for cardiovascular risk prediction across institutionally siloed, heterogeneous clinical datasets. It addresses two key challenges simultaneously:

**Non-IID data heterogeneity** — real hospital datasets differ in feature distributions, class imbalance, and local sample sizes. FedCVR uses a FedProx-style proximal term on each client to penalise excessive drift from the global model.

**Privacy preservation** — raw patient gradients can leak sensitive information. FedCVR applies client-side Differential Privacy via Opacus (per-sample gradient clipping + calibrated Gaussian noise) before any update leaves a client.

**Adaptive aggregation** — rather than plain weighted averaging, the server applies an Adam-style moment estimator to the aggregated pseudo-gradient, accelerating convergence under non-IID conditions.

```
                ┌──────────────────────────────────────────────────┐
                │                   FL Server                       │
                │  w_{t+1} = w_t + η · m̂_t / (√v̂_t + ε)         │
                │  (adaptive moment aggregation)                    │
                └────────┬──────────────┬──────────────────────────┘
                         │              │  broadcast w_t
           ┌─────────────┘              └──────────────┐
           ▼                                            ▼
  ┌──────────────────┐                       ┌──────────────────┐
  │  Client 0        │        · · ·          │  Client 4        │
  │  Framingham      │                       │  Hungarian       │
  │                  │                       │                  │
  │  1. FedProx      │                       │  1. FedProx      │
  │     local SGD    │                       │     local SGD    │
  │  2. DP: clip +   │                       │  2. DP: clip +   │
  │     Gaussian     │                       │     Gaussian     │
  │     noise        │                       │     noise        │
  └──────────────────┘                       └──────────────────┘
```

---

## Repository Structure

```
fedcvr/
├── fedcvr/                   # Core Python package
│   ├── __init__.py
│   ├── model.py              # 3-layer DNN (logit output)
│   ├── client.py             # FedCVRClient – FedProx + Opacus DP
│   ├── strategy.py           # FedCVRStrategy – Adam server aggregation
│   └── data_utils.py         # Data loading, harmonisation, preprocessing
│
├── experiments/              # Reproducible experiment scripts
│   ├── run_comparison.py     # Investigation 1: FedCVR vs baselines
│   └── run_dp_sensitivity.py # Investigation 3: DP sensitivity analysis
│
├── data/                     # Place dataset CSV files here (see data/README.md)
│   ├── README.md             # Download instructions for all 5 datasets
│   └── convert_cleveland.py  # Utility: convert UCI .data → .csv
│
├── results/                  # Output directory (auto-created by experiments)
├── requirements.txt
└── .gitignore
```

---

## Method

### Model Architecture

A three-layer deep neural network (DNN):

```
Input(10) → Linear(16) → ReLU → Linear(8) → ReLU → Linear(1)
```

Binary cross-entropy with logits (`BCEWithLogitsLoss`) is used with a class-imbalance correction weight computed per client from local label frequencies.

### Client Update (FedProx + DP)

Each client minimises:

```
L_local(w) = L_CE(w) + (μ/2) · ‖w − w_global‖²
```

where `μ` is the proximal coefficient (default `0.1`). After computing gradients via SGD (momentum 0.9, lr 0.01, 5 local epochs), Opacus clips per-sample gradients to norm `C = 1.0` and adds calibrated Gaussian noise `N(0, σ²C²I)` before the update is transmitted.

### Server Aggregation (Adaptive Moment Estimation)

The server maintains first- and second-moment vectors and applies bias-corrected Adam-style updates:

```
Δ_t  = FedAvg(client_updates) − w_t
m_t  = β₁·m_{t-1} + (1−β₁)·Δ_t          (β₁ = 0.9)
v_t  = β₂·v_{t-1} + (1−β₂)·Δ_t²         (β₂ = 0.999)
m̂_t  = m_t / (1 − β₁ᵗ)
v̂_t  = v_t / (1 − β₂ᵗ)
w_{t+1} = w_t + η·m̂_t / (√v̂_t + ε)      (η = 0.01, ε = 1e-8)
```

### Datasets

Five publicly available cardiovascular datasets simulate an institutionally siloed, non-IID federation:

| Client | Dataset | Samples | Primary heterogeneity |
|--------|---------|---------|----------------------|
| 0 | Framingham Heart Study | ~4,238 | Longitudinal, community |
| 1 | UCI Cleveland | 303 | Clinical, catheterisation |
| 2 | FIC Pakistan | ~1,000 | Regional, demographic shift |
| 3 | Heart Disease Prediction | ~270 | Mixed, imbalanced |
| 4 | Hungarian | ~294 | European clinical |

All datasets are harmonised to 10 common features. See `data/README.md` for download instructions.

---

## Installation

```bash
# Clone the repository
git clone https://github.com/rodrigo-tertulino/fedcvr.git
cd fedcvr

# Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
# .venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt
```

> **Note:** Opacus requires PyTorch ≥ 2.0. GPU is optional – all experiments were run on CPU.

---

## Quickstart

### 1. Download the datasets

Follow the instructions in `data/README.md` and place the five CSV files in the `data/` directory.

### 2. Run Investigation 1 – Strategy comparison

```bash
python -m experiments.run_comparison \
    --data_dir data \
    --rounds   100 \
    --out_dir  results
```

This runs four scenarios (FedAvg, FedProx, FedCVR, FedCVR+DP) for 100 rounds and saves:
- `results/comparison_metrics.csv`
- `results/comparison_plot.png`

### 3. Run Investigation 3 – DP sensitivity analysis

```bash
python -m experiments.run_dp_sensitivity \
    --data_dir data \
    --rounds   50 \
    --out_dir  results
```

Evaluates four privacy levels (no DP, σ=0.8, σ=1.1, σ=1.5) and saves:
- `results/dp_sensitivity_metrics.csv`
- `results/dp_sensitivity_plot.png`

---

## Key Results (from the paper)

| Strategy | Accuracy | Precision | Recall | F1-Score |
|----------|----------|-----------|--------|----------|
| FedAvg (baseline) | — | — | — | — |
| FedProx | — | — | — | — |
| **FedCVR (ours)** | — | — | — | — |
| FedCVR + DP (σ=1.1) | — | — | — | — |

*See Table 1 and Table 2 in the paper for the full numerical results.*

### Privacy-Utility Trade-off (Investigation 3)

| Scenario | Noise (σ) | Approx. ε | Accuracy | Recall |
|----------|-----------|-----------|----------|--------|
| No DP | — | ∞ | — | — |
| Low Privacy | 0.8 | high | — | — |
| Medium Privacy | 1.1 | medium | — | — |
| High Privacy | 1.5 | low | — | — |

*See Table 3 in the paper.*

---

## Citation

If you use this code or the methodology in your work, please cite:

```bibtex
@article{tertulino2025fedcvr,
  title   = {Overcoming Clinical Data Heterogeneity: A Secure Federated
             Framework for Cardiovascular Risk Prediction},
  author  = {Tertulino, Rodrigo and Almeida, Ricardo and Alencar, La{\'e}rcio},
  journal = {iSys -- Revista Brasileira de Sistemas de Informa{\c{c}}{\~a}o},
  year    = {2025}
}
```

---

## License

This project is released under the **MIT License**. See `LICENSE` for details.

The datasets used in this work are subject to their own respective licenses.
Please refer to `data/README.md` for the original sources and terms of use.

---

## Contact

Rodrigo Tertulino — `rodrigo.tertulino@ifrn.edu.br`
IFRN – Departamento de Informática, Natal, Brazil
