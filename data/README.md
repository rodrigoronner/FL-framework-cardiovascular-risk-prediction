# Data Directory

This directory should contain the five cardiovascular CSV files used in the FedCVR experiments.
The datasets are **not included** in the repository because they are publicly available under their
own licenses. Follow the download instructions below, then place each file here as-is.

---

## Dataset 1 – Framingham Heart Study
**File expected:** `framingham.csv`

| Field | Details |
|-------|---------|
| Source | Kaggle – Framingham Heart Study dataset |
| URL | https://www.kaggle.com/datasets/aasheesh200/framage |
| Samples | ~4,238 |
| Key target column | `TenYearCHD` |

Download the CSV and rename / save it as `data/framingham.csv`.

---

## Dataset 2 – UCI Cleveland Heart Disease
**File expected:** `cleveland.csv`

| Field | Details |
|-------|---------|
| Source | UCI Machine Learning Repository |
| URL | https://archive.ics.uci.edu/dataset/45/heart+disease |
| File to download | `processed.cleveland.data` |
| Samples | 303 |
| Key target column | `num` (0 = no disease, 1–4 = disease) |

The raw file uses `.data` format with `?` for missing values. Convert it with:

```bash
python data/convert_cleveland.py --input processed.cleveland.data --output data/cleveland.csv
```

---

## Dataset 3 – FIC Pakistan
**File expected:** `fic_pakistan.csv`

| Field | Details |
|-------|---------|
| Source | Kaggle – Heart Attack Prediction Dataset (Pakistan) |
| URL | https://www.kaggle.com/datasets/nabeelsajid9/heart-attack-prediction |
| Samples | ~1,000 |
| Key target column | `Mortality` |

Download and save as `data/fic_pakistan.csv`.

---

## Dataset 4 – Heart Disease Prediction
**File expected:** `heart_disease_prediction.csv`

| Field | Details |
|-------|---------|
| Source | Kaggle – Heart Disease Prediction |
| URL | https://www.kaggle.com/datasets/rishidamarla/heart-disease-prediction |
| Samples | ~270 |
| Key target column | `Heart Disease` |

Download and save as `data/heart_disease_prediction.csv`.

---

## Dataset 5 – Hungarian Heart Disease
**File expected:** `Hungarian-98-10.csv`

| Field | Details |
|-------|---------|
| Source | UCI Machine Learning Repository |
| URL | https://archive.ics.uci.edu/dataset/45/heart+disease |
| File to download | `processed.hungarian.data` |
| Samples | ~294 |
| Key target column | `num` |

Convert with:

```bash
python data/convert_cleveland.py --input processed.hungarian.data --output data/Hungarian-98-10.csv
```

---

## Harmonised Feature Schema

All five datasets are mapped to the following 10-feature schema during preprocessing:

| Feature | Description |
|---------|-------------|
| `age` | Patient age in years |
| `sex` | Biological sex (1 = Male, 0 = Female) |
| `cp` | Chest pain type (0–3) |
| `trestbps` | Resting blood pressure (mm Hg) |
| `chol` | Serum cholesterol (mg/dl) |
| `fbs` | Fasting blood sugar > 120 mg/dl (1 = True) |
| `restecg` | Resting ECG results (0–2) |
| `thalach` | Maximum heart rate achieved |
| `exang` | Exercise-induced angina (1 = Yes) |
| `oldpeak` | ST depression induced by exercise |

Target: `target` – binary (0 = no disease, 1 = disease).

Missing features in a given dataset are imputed with the column median.
