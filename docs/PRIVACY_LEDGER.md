# Privacy ledger

Every use of patient data in the pipeline, who it involves, and how it is covered.
"Certified" means covered by the per-client (ε_k, δ = 10⁻⁵) guarantee of the reported
model, at **patient-level** adjacency (add or remove one patient with all their ICU stays).

| # | Data use | Patients involved | Coverage | Where in code |
|---|----------|-------------------|----------|---------------|
| 1 | Cohort filter: drop stays with > 50% clinical features missing | Each stay on its own | Per-row rule; no pooled statistic | `load_mimic_data` |
| 2 | Missing-value imputation | None | Public reference value per feature | `dp_core_v5.ReferencePreprocessor` |
| 3 | Standardization and clipping to [−5, 5] | None | Public reference value and scale per feature | `dp_core_v5.CLINICAL_REFERENCE` |
| 4 | Class weight of the loss (w_pos = 9) | None | Public mortality prior of 10% | `Config.MORTALITY_PRIOR` |
| 5 | Training of each reported model: all rounds, all client updates | Deployment cohort (50% of training patients) | **Certified**: (ε_k, δ)-DP, patient level, one run | `HealthcareFLTrainer.local_training_with_dp` |
| 6 | Budget-search candidate runs (NSGA-II, MO-Privacy, LHS) | Design cohort (50% of training patients) | **Not certified.** Runs compose: per client, ε ≈ 555 for the LHS search, 136–1,105 for NSGA-II and 100–794 for the two-objective search, whose candidates concentrate at different budgets (second master rerun, 50% design cohort; recomputed and stored with every search result); about 1,500 on average for the three. Treated as development data | `search_composition` |
| 7 | Client sample counts: aggregation weights, sampling rate q_k | Deployment cohort | Released exactly; treated as public consortium metadata | `federated_averaging`, `dp_schedule` |
| 8 | Search fitness, decision threshold, recalibration | Validation patients | No guarantee claimed. Disjoint from all training patients, so releasing these does not weaken item 5 | `evaluate_global_model`, `run_fcfl` |
| 9 | Final evaluation | Test patients | No guarantee claimed. Disjoint from all training patients | `finalize_result` |
| 10 | Client-level monitoring (local 20% split) | Deployment cohort, held-out part | Used for per-client AUROC and as non-members in the membership-inference test; not used for training | `evaluate_per_client` |
| 11 | DP hyperparameter sensitivity grid (clipping norm, lot size, learning rate) | Design cohort | **Not certified**; same status as item 6. Scored on validation | `experiments_extra/run_dp_mechanism.py tuning` |
| 12 | Mechanism ablation and other comparison runs on the deployment cohort (other budgets, clipping-only arms, confirmation of a tuned configuration) | Deployment cohort | Each certified run holds for that run **alone**. Releasing several models trained on the same patients composes. Arms with `NOISE_SCALE != 1` have no guarantee. A deployment would train and release one model | `experiments_extra/run_dp_mechanism.py` |

## Mechanism

Per optimizer step at client k, with m_k local training patients and expected lot size B = 64:

1. Each patient is included independently with probability q_k = B / m_k.
2. A sampled patient contributes all of their stays; the stays' gradients are summed.
3. Each per-patient sum is clipped to ℓ2 norm C = 30 (selected on the design cohort; ledger item 11).
4. The clipped sums are added, N(0, σ_k² C²) noise is added per coordinate, and the result is divided by B.

Removing a patient changes the sum in step 4 by at most C whatever their number of
stays, so the Poisson-subsampled Gaussian accountant applies directly at patient level.
σ_k is the smallest value for which the RDP accountant certifies (ε_k, δ) over all
S_k = T · E · ⌈m_k / B⌉ steps.

## Why the search needs its own cohort

A certificate for one run is not a certificate for a workflow of many runs on the same
data. Each search candidate is a DP training run, and privacy loss adds up across runs.
Computed with the same accountant (LHS candidates, 15 rounds each, nominal budgets 1–10):

| Workflow | Composed ε per client (RDP upper bound) |
|----------|------------------------------------------|
| One candidate run | 1.3 – 4.5 (nominal budgets 2.5 – 9.1) |
| One search, 1,275 candidates | ≈ 550 |
| One search + final model | ≈ 550 |
| Three searches + final model (the submitted workflow) | ≈ 1,640 |

With the design/deployment split, deployment patients are touched by exactly one run
and receive exactly (ε_k, δ).

## Not claimed

- No formal guarantee for design-cohort, validation or test patients.
- Non-private baselines carry no guarantee.
- A threshold or calibration map fitted on validation data is not itself differentially private.
- The honest-but-curious server sees every client update of the reported run; the guarantee is stated against that observer. Secure aggregation is not used or assumed.

## Ablation switches

Each switch's second value reproduces the original submission and exists only to
measure the effect of the change. They are not privacy-safe.

| Switch (CLI) | Default | Control |
|--------------|---------|---------|
| `--privacy-unit` | `patient` | `record` (one ICU stay; patient protected only via group privacy) |
| `--preprocessing` | `reference` | `cohort` (pooled medians and fitted scaler) |
| `--class-weight` | `prior` | `pooled` (from training labels) |
| `--design-fraction` | `0.50` | `0` (search on the training cohort) |
