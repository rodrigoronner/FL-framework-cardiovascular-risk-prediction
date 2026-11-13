# Data

Four public cardiac cohorts act as the four institutions. The processed CSV files are included; the converters rebuild them from the original public files.

| Site | File | Source | Records | Positive | Converter |
|---|---|---|---|---|---|
| H1 | `zalizadeh_sani.csv` | Z-Alizadeh Sani, Shaheed Rajaei Cardiovascular Center, Tehran — UCI ML Repository (doi:10.24432/C5Q31T) | 303 | 71.3% | `convert_z_alizadeh_sani.py --input "Z-Alizadeh sani dataset.xlsx"` |
| H2 | `mendeley_cvd.csv` | Cardiovascular Disease Dataset, multispecialty hospital, India — Doppala & Bhattacharyya, Mendeley Data V1 (doi:10.17632/dzz48mvjht.1, CC BY 4.0) | 1,000 | 58.0% | `convert_mendeley_cvd.py --input Cardiovascular_Disease_Dataset.csv` |
| H3 | `cleveland.csv` | UCI Heart Disease, Cleveland + Budapest + Zurich/Basel + Long Beach (doi:10.24432/C52P4X) | 920 (918 after removing 2 exact duplicates) | 55.3% | `convert_cleveland.py --input_dir <folder with processed.*.data>` |
| H4 | `kaggle_heart.csv` | Cardiovascular Disease dataset (Kaggle, sulianova), ambulatory examinations; stratified subsample of 1,025 of 70,000 records, fixed seed | 1,025 | 49.5% | `convert_cardio_sulianova.py --input cardio_train.csv --n_samples 1025` |

Targets: angiographic coronary artery disease (H1), binary diagnosis (H2, H4), `num > 0` (H3).

## Integrity

`python -m experiments.check_data_integrity` must print `RESULT: OK`. It checks that no record is shared between clients (H2 and H3, the only pair with a common full schema, are matched on age, sex, resting blood pressure, cholesterol, maximum heart rate and ST depression) and reports duplicates within clients. H4 has 63 records that coincide with another record on all of its coarse fields; they are kept as distinct survey participants, an assumption stated in the paper.

**History (revision R1).** The first submission used the IEEE DataPort "Heart Disease Dataset (Comprehensive)" as H2. That file merges the UCI sites and therefore contained H3: all 827 matchable H3 records appeared in it, and 127 of the 184 H3 test records were in the H2 training fold. It was replaced by the Mendeley cohort and removed from the repository. Earlier candidate clients (Framingham, NHANES, FIC Pakistan) were dropped before the first submission and are in the git history.

## Harmonized inputs (22 columns)

| Input | H1 | H2 | H3 | H4 |
|---|---|---|---|---|
| age, sex, systolic blood pressure | ✓ | ✓ | ✓ | ✓ |
| serum cholesterol (mg/dL) | ✓ | ✓ | ✓ | – |
| fasting glucose > 120 mg/dL (binary) | ✓ (thresholded) | ✓ | ✓ | – |
| diastolic blood pressure | – | – | – | ✓ |
| cholesterol and glucose categories, one-hot (6) | – | – | – | ✓ |
| H1 native variables (10) | ✓ | – | – | – |

An input a site does not record is a constant zero at that site, not imputed. Preprocessing is data-independent: values are clipped and scaled with fixed public physiological ranges and missing values replaced by fixed clinical reference values (`dpfedadam/data_utils.py`). The ten H1 variables are ranked by CatBoost importance on H1's **training fold** of each partition; this ranking is not covered by the privacy accountant and is declared in the paper.

Splits: stratified 70/10/20 train/validation/test per site; partition seeds 0–19.
