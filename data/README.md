# Data Directory

This directory should contain the five cardiovascular CSV files used in the FedCVR experiments.
The datasets are **not included** in the repository because they are publicly available under their
own licenses. Follow the download instructions below, then place each file in `data/` with the
exact filename shown.

---

## Dataset H1 – Framingham Heart Study
**File expected:** `framingham.csv`

| Field | Details |
|-------|---------|
| Source | Kaggle – Heart Disease Prediction Using Logistic Regression |
| URL | https://www.kaggle.com/datasets/dileep070/heart-disease-prediction-using-logistic-regression |
| Samples | 4,238 |
| Key target column | `TenYearCHD` |
| Key columns used | `male` (sex), `age`, `sysBP` (systolic BP), `diaBP` (diastolic BP), `totChol` (cholesterol), `glucose` (fasting blood glucose) |

Download the CSV and save it as `data/framingham.csv`.

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

## Dataset H4 – FIC Pakistan (Faisalabad)
**File expected:** `fic_pakistan.csv`

| Field | Details |
|-------|---------|
| Source | Kaggle – Heart Attack Prediction |
| URL | https://www.kaggle.com/datasets/imnikhilanand/heart-attack-prediction |
| Samples | 1,000 |
| Key target column | `Mortality` |
| Key columns used | `Age`, `Gender` (sex), `Systolic BP`, `Diastolic BP`, `Cholestrol`, `FBS` |

Download and save as `data/fic_pakistan.csv`.

---

## Dataset H5 – Kaggle Heart Prediction
**File expected:** `kaggle_heart.csv`

| Field | Details |
|-------|---------|
| Source | Kaggle – Heart Failure Prediction Dataset |
| URL | https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction |
| Samples | 1,025 |
| Key target column | `HeartDisease` |
| Key columns used | `Age`, `Sex`, `RestingBP` (systolic BP), `Cholesterol`, `FastingBS` |

Download and save as `data/kaggle_heart.csv`.

Note: diastolic BP (`diaBP`) is not present in this dataset and will be imputed with the column median.

---

## Harmonized 6-Feature Schema

All five datasets are mapped to the following **6-feature schema** during preprocessing.
Features not present in a given dataset are filled with `NaN` and imputed with the column median.

| Feature | Description | Unit |
|---------|-------------|------|
| `age` | Patient age | years |
| `sex` | Biological sex | 1 = Male, 0 = Female |
| `trestbps` | Systolic blood pressure | mm Hg |
| `diaBP` | Diastolic blood pressure | mm Hg |
| `chol` | Serum cholesterol | mg/dl |
| `fbs` | Fasting blood glucose (or fasting blood sugar > 120 mg/dl) | continuous mg/dl or binary |

**Target:** `target` — binary (0 = no cardiovascular disease, 1 = disease).

This feature set was selected based on cross-dataset availability and alignment with the six risk factors
referenced in the 2021 ESC and 2019 ACC/AHA cardiovascular prevention guidelines.

---

## Preprocessing Pipeline

The following steps are applied **per client** inside `fedcvr/data_utils.py`:

1. **Column harmonization** — dataset-specific column names are mapped to the 6-feature schema above.
2. **IQR-based outlier capping** — values below Q1 − 1.5×IQR or above Q3 + 1.5×IQR are clipped.
3. **Median imputation** — remaining `NaN` values are filled with the column median.
4. **Target binarisation** — any positive class value is mapped to 1.
5. **Stratified 70/10/20 split** — training (70%), validation (10%), test (20%), `random_state=42`.
6. **SMOTE oversampling** — applied to the **training fold only** to address class imbalance.
7. **StandardScaler normalisation** — fitted on the (pre-SMOTE) training fold; applied to all folds.
