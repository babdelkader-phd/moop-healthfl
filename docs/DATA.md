# Data Requirements and Cohort Definition

Everything the pipeline reads from MIMIC-IV, and every transformation applied,
so that the cohort is fully reproducible from a credentialed copy of
**MIMIC-IV v3.1** (https://physionet.org/content/mimiciv/3.1/).

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
3. Missing clinical values stay missing in the cached feature table and are
   set at load time to a fixed public reference value per feature
   (`dp_core_v5.CLINICAL_REFERENCE`). Hemoglobin (item 50811) is missing in
   81% of stays and lactate in 46%.
4. One-hot encode the categorical demographics (5 ethnicity + 3 insurance
   indicators), giving a 19-dimensional input (10 continuous + gender +
   8 indicators).
5. Transform the **continuous** features only: (x − reference) / scale with
   the same public constants, clipped to [−5, 5]. Nothing is fitted to the
   data, so no statistic crosses the train/validation/test boundary and
   preprocessing has no privacy cost. Clipping also neutralises gross
   charting errors in the raw data.

Expected result with MIMIC-IV v3.1: **94,095 ICU stays from 65,178 patients,
11.8% mortality**.
The natural class prevalence is deliberately preserved (no resampling); class
imbalance is handled at the loss level during training.

## Patient-level split

All partitioning is at the **patient** level: the 65,178 unique `subject_id`s
are split 70/10/20 into train/validation/test (`RandomState(42)`), so every
stay of a patient falls into exactly one partition — 65,613 / 9,524 / 18,958
stays respectively, fixed across all methods and seeds. The validation split
is used only for threshold selection and NSGA-II fitness; all reported
results come from the held-out test split.

## Federated client partition (`hospital_id`)

Clients are defined by `first_careunit`, and each **training patient is
assigned to exactly one client** (the modal care unit of their stays; 6,800
multi-unit patients, 9,426 stays = 14.4% of the training partition
reassigned; zero cross-client patient overlap). Expected training partition:

| Client | Care units | Training stays | Patients | Mortality |
|---|---|---|---|---|
| Hospital 0 | MICU | 15,023 | 8,952 | 15.6% |
| Hospital 1 | MICU/SICU | 10,542 | 7,084 | 15.5% |
| Hospital 2 | CVICU | 9,998 | 7,966 | 4.8% |
| Hospital 3 | SICU + CCU | 16,461 | 11,291 | 12.8% |
| Hospital 4 | TSICU + Neuro SICU + other/unmapped | 13,589 | 10,331 | 8.9% |

This is a **simulated** federation (single-source, care-unit-partitioned) —
a controlled proxy for institutional heterogeneity, not a multi-hospital
deployment.

## Privacy unit

The certified (ε, δ) guarantees are at **patient-level adjacency**: adding or
removing one patient, with all of their ICU stays, from a client's training
set. One patient can contribute multiple stays (median 1; 75% single-stay;
maximum 41); the mechanism clips each patient's summed contribution, so the
guarantee does not depend on the number of stays. The table above lists the
full training partition; within each client 50% of patients form the design
cohort (budget search only) and 50% the deployment cohort (reported models).
See [PRIVACY_LEDGER.md](PRIVACY_LEDGER.md).

## Caching

The first run writes the final feature table to
`<output-dir>/mimic_iv_features_v2.parquet` (columns: `stay_id, subject_id,
hospital_id, mortality, age, gender, ethnicity, insurance` + the nine
clinical features; `subject_id` is retained for the patient-level split).
Subsequent runs and all `experiments_extra/` harnesses start from this cache.
**The cache is derived from MIMIC-IV and must not be shared or committed.**
