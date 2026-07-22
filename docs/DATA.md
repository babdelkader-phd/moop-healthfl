# Data Requirements and Cohort Definition

Everything the pipeline reads from MIMIC-IV, and every transformation applied,
so that the cohort is fully reproducible from a credentialed copy of
**MIMIC-IV v2.2** (https://physionet.org/content/mimiciv/2.2/).

## Required files

Pass `--mimic-path /path/to/mimic-iv` where that directory contains:

| File | Columns read |
|---|---|
| `icu/icustays.csv.gz` | `stay_id, subject_id, hadm_id, first_careunit, intime, outtime` |
| `icu/chartevents.csv.gz` | `stay_id, itemid, charttime, valuenum` (read in 1M-row chunks) |
| `hosp/patients.csv.gz` | `subject_id, gender, anchor_age, anchor_year, dod` |
| `hosp/admissions.csv.gz` | `hadm_id, subject_id, race, insurance, hospital_expire_flag` |
| `hosp/labevents.csv.gz` | `hadm_id, itemid, charttime, valuenum` (read in 1M-row chunks) |

## Clinical features (13) and exact item IDs

**Vital signs** — from `chartevents`, restricted to the first 24 h after ICU
admission (`intime` to `intime + 24h`), aggregated as the per-stay mean:

| Feature | itemid(s) | Note |
|---|---|---|
| `heart_rate` | 220045 | |
| `systolic_bp` | 220050, 220179 | arterial + non-invasive merged |
| `temperature` | 223761, 223762 | 223761 is °F, converted to °C: (F−32)·5⁄9 |
| `resp_rate` | 220210 | |
| `spo2` | 220277 | |

(Item IDs 220051/220052 are loaded with the vital set but only the mapped
features above enter the final table.)

**Laboratory values** — from `labevents`, joined via `hadm_id`, same 24 h
window, per-stay mean:

| Feature | itemid |
|---|---|
| `creatinine` | 50912 |
| `lactate` | 50813 |
| `wbc` | 51301 |
| `hemoglobin` | 50811 |

**Demographics** (from `icustays` + `patients` + `admissions`):

| Feature | Definition |
|---|---|
| `age` | `anchor_age + (year(intime) − anchor_year)`, clipped to [18, 91] |
| `gender` | 1 = Female, 0 = Male |
| `ethnicity` | from `race`: 0 = White, 1 = Black/African American, 2 = Hispanic/Latino, 3 = Asian, 4 = Other/Unknown |
| `insurance` | 0 = Medicare, 1 = Medicaid, 2 = Other |

**Label**: `mortality` = `hospital_expire_flag` (in-hospital death), missing → 0.

## Cohort filtering and imputation

1. Keep all adult ICU stays (age clipped to 18–91).
2. Drop stays where more than 50% of the nine clinical features
   (vitals + labs) are missing.
3. Median-impute remaining missing clinical values; any residual NaN in the
   full feature set is median-imputed before scaling.
4. Standardize all 13 features to zero mean / unit variance (scaler fit on the
   full cohort).

Expected result with MIMIC-IV v2.2: **94,095 ICU stays, 11.8% mortality**.
The natural class prevalence is deliberately preserved (no resampling); class
imbalance is handled at the loss level during training.

## Federated client partition (`hospital_id`)

Clients are defined by `first_careunit`:

| Client | Care units | Expected n | Mortality |
|---|---|---|---|
| Hospital 0 | MICU | 20,616 | 16.4% |
| Hospital 1 | MICU/SICU | 15,393 | 15.4% |
| Hospital 2 | CVICU | 14,745 | 4.3% |
| Hospital 3 | SICU + CCU | 23,676 | 12.5% |
| Hospital 4 | TSICU + Neuro SICU + other/unmapped | 19,665 | 9.2% |

This is a **simulated** federation (single-source, care-unit-partitioned) —
a controlled proxy for institutional heterogeneity, not a multi-hospital
deployment.

## Test split

Global test set: 20% stratified split of the pooled cohort,
`random_state=42`, fixed across all methods and seeds (18,819 samples).

## Caching

The first run writes the final feature table to
`<output-dir>/mimic_iv_features.parquet` (columns: `stay_id, hospital_id,
mortality, age, gender, ethnicity, insurance` + the nine clinical features).
Subsequent runs and all `experiments_extra/` harnesses start from this cache.
**The cache is derived from MIMIC-IV and must not be shared or committed.**
