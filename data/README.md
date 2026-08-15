# Data Directory

This directory should contain the four cardiovascular CSV files used in the DP-FedAdam experiments.
The datasets are **not included** in the repository because they are publicly available under their
own licenses. Follow the download instructions below, then place each file in `data/` with the
exact filename shown.

**History note (2026-08-14):** a fifth client, FIC Pakistan (`fic_pakistan.csv`,
`Mortality` target), was removed. Its target measured in-hospital death among
already-admitted heart patients, not presence/diagnosis of heart disease like
the other four clients - the classic risk factors (age, sex, systolic BP,
fasting glucose) correlated *positively* with target in every other client
(+0.11 to +0.44) but *negatively* in FIC Pakistan (-0.09 to -0.28). A federated model
trained across the other clients scored AUC < 0.5 (worse than random)
specifically on this client - a genuine label-semantics conflict, not a
preprocessing bug, so the dataset was dropped rather than patched. The
former H5 (sulianova) is now labelled H4. See `fedcvr/data_utils.py`'s
module docstring and git history for the full record.

---

## Dataset H1 – Z-Alizadeh Sani (Shaheed Rajaei Cardiovascular Center, Tehran)
**File expected:** `zalizadeh_sani.csv`

| Field | Details |
|-------|---------|
| Source | Kaggle mirror of the UCI Z-Alizadeh Sani dataset |
| URL | https://www.kaggle.com/datasets/tanyachi99/zalizadeh-sani-dataset-2csv |
| Also at | https://archive.ics.uci.edu/dataset/411/extention+of+z+alizadeh+sani+dataset (UCI, DOI 10.24432/C5461K) |
| Samples | 303 |
| Key target column | `Cath` (`Cad` = coronary artery disease present, `Normal` = absent) |
| Key columns used | `Age`, `Sex`, `BP` (systolic BP), `FBS`, plus `LDL`/`HDL`/`TG` (combined via the Friedewald formula into total cholesterol), and the 10 extra native features listed below |

Convert with:

```bash
python data/convert_z_alizadeh_sani.py --input "Z-Alizadeh sani dataset (2).csv" --output data/zalizadeh_sani.csv
```

**History note:** H1 was originally the Framingham Heart Study, then briefly a
NHANES-derived client, before landing on Z-Alizadeh Sani (2026-08-13). Both
earlier candidates are documented in git history / project memory, not in
this directory, per the "only files we're using" cleanup — see `git log`
on `data/convert_z_alizadeh_sani.py`'s predecessors if the history is
needed.

**H1-only extra features.** Unlike H2–H4, H1 contributes 10 additional
native columns beyond the shared 6-feature schema (see "Harmonized
Schema" below) — a diagnostic (not just risk-factor) feature set,
including `Typical Chest Pain`, echo/ECG findings, and lab values. These
were selected as the top-10 features by CatBoost importance on the full
40-column native schema, which raised centralized AUC from 0.59–0.63
(6-feature schema alone) to 0.85–0.87. They are zero for every other
client, which does not collect them.

---

## Dataset H2 – IEEE Comprehensive Heart Disease
**File expected:** `ieee_chd.csv`

| Field | Details |
|-------|---------|
| Source | IEEE DataPort – Comprehensive Heart Disease Dataset |
| URL | https://ieee-dataport.org/open-access/heart-disease-dataset-comprehensive |
| Samples | 1,190 |
| Key target column | `HeartDisease` |
| Key columns used | `Age`, `Sex`, `SBP` (systolic BP), `DBP` (diastolic BP), `Cholesterol`, `FastingBS` |

Download and save as `data/ieee_chd.csv`.

---

## Dataset H3 – UCI Cleveland Heart Disease
**File expected:** `cleveland.csv`

| Field | Details |
|-------|---------|
| Source | UCI Machine Learning Repository |
| URL | https://archive.ics.uci.edu/dataset/45/heart+disease |
| File to download | `processed.cleveland.data` |
| Samples | 920 (after preprocessing) |
| Key target column | `num` (0 = no disease, 1–4 = disease, binarised to 0/1) |
| Key columns used | `age`, `sex`, `trestbps` (systolic BP), `chol` (cholesterol), `fbs` (fasting blood glucose) |

The raw file uses `.data` format with `?` for missing values. Convert it with:

```bash
python data/convert_cleveland.py --input processed.cleveland.data --output data/cleveland.csv
```

Note: diastolic BP (`diaBP`) is not present in the Cleveland dataset and will be imputed with the column median.

---

## Dataset H4 – Cardiovascular Disease dataset (sulianova)
**File expected:** `kaggle_heart.csv`

| Field | Details |
|-------|---------|
| Source | Kaggle – Cardiovascular Disease dataset (sulianova) |
| URL | https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset |
| Raw samples | 70,000 (semicolon-delimited `cardio_train.csv`); subsampled to 1,025 |
| Key target column | `cardio` |
| Key columns used | `age` (days, converted to years), `gender`, `ap_hi`/`ap_lo` (systolic/diastolic BP), `cholesterol`/`gluc` (3-level ordinal, one-hot encoded — see below) |

Substituted for the paper's original H5 source
(`rashadrmammadov/heart-disease-prediction`), which became inaccessible.
Convert with:

```bash
python data/convert_cardio_sulianova.py --input cardio_train.csv --output data/kaggle_heart.csv --n_samples 1025
```

Note: H4 has both systolic (`ap_hi`) and diastolic (`ap_lo`) BP natively —
unlike H2 and H3, whose diastolic BP is absent and median-imputed.

---

## Harmonized Schema (22 features)

All four datasets share a base **6-feature schema**; two clients each add extra
columns that are zero for every other client (see `fedcvr/data_utils.py`
for the authoritative definitions). Features not present in a given
dataset are filled with `NaN` and median-imputed **from that client's own
training fold only** (no cross-fold or cross-client leakage).

**Base 6 (all clients):**

| Feature | Description | Unit |
|---------|-------------|------|
| `age` | Patient age | years |
| `sex` | Biological sex | 1 = Male, 0 = Female |
| `trestbps` | Systolic blood pressure | mm Hg |
| `diaBP` | Diastolic blood pressure | mm Hg |
| `chol` | Serum cholesterol | mg/dl |
| `fbs` | Fasting blood glucose (or fasting blood sugar > 120 mg/dl) | continuous mg/dl or binary |

**+6, H4-only** (`chol_ord_1..3`, `fbs_ord_1..3`): one-hot encoding of H4's
native 3-level ordinal cholesterol/glucose risk categories, since they are
not on the same continuous mg/dl scale as the other three clients' `chol`/`fbs`.

**+10, H1-only** (`h1_typical_chest_pain`, `h1_ef_tte`, `h1_region_rwma`,
`h1_htn`, `h1_tg`, `h1_k`, `h1_tinversion`, `h1_esr`, `h1_neut`, `h1_hdl`):
Z-Alizadeh Sani's top-10 native features by predictive importance — see
the H1 section above.

**Target:** `target` — binary (0 = no cardiovascular disease, 1 = disease).

The base 6 were selected based on cross-dataset availability and alignment
with risk factors referenced in the 2021 ESC and 2019 ACC/AHA
cardiovascular prevention guidelines. The two extra blocks are later,
client-specific additions (2026-08-13) that measurably improved that
client's standalone predictive performance without altering the other
clients' feature vectors (always zero for them).

---

## Preprocessing Pipeline

The following steps are applied **per client** inside `fedcvr/data_utils.py`,
in this order (the split happens early so that every later statistic — IQR
bounds, medians, the scaler — is fit on the training fold only and merely
applied to val/test, with no leakage):

1. **Column harmonization** — dataset-specific column names are mapped to the schema above.
2. **Target binarisation** — any positive class value is mapped to 1 (deterministic, so doing this before the split introduces no leakage).
3. **Stratified 70/10/20 split** — training (70%), validation (10%), test (20%), `random_state=42`.
4. **IQR-based outlier capping** — bounds (Q1 − 1.5×IQR, Q3 + 1.5×IQR) computed from the training fold only, applied to all three folds.
5. **Median imputation** — medians computed from the (capped) training fold only, applied to all three folds.
6. **SMOTE oversampling** — applied to the **training fold only** to address class imbalance.
7. **MinMaxScaler normalisation to [0, 1]** — fitted on the (pre-SMOTE) training fold; applied to all folds. Chosen over standardization so every feature shares the same bounded range as the fixed per-sample DP-SGD clipping norm `C`, and so the one-hot columns (already in [0, 1]) are left undistorted.
