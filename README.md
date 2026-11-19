# MOOP-HealthFL

**Certified differential privacy in clinical federated learning: rigorous evaluation and multi-objective privacy-budget selection**

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21499213.svg)](https://doi.org/10.5281/zenodo.21499213)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

This repository contains the full implementation and evaluation harnesses for the MOOP-HealthFL paper. The study rigorously evaluates **certified** differential privacy (per-example DP-SGD with a Rényi-DP accountant calibrating total-training (ε, δ) guarantees) in a simulated five-client clinical federation built from MIMIC-IV, and formulates heterogeneous per-client privacy-budget allocation as a constrained three-objective Pareto search (privacy, utility, fairness) solved with NSGA-II under clinical constraints, with a matched-budget Latin-hypercube random-search control and a five-level fairness evaluation (client, group, individual, system, outcome).

- **Archived release (citable):** https://doi.org/10.5281/zenodo.21499213
- **Development repository:** https://github.com/babdelkader-phd/moop-healthfl

> Citation details will be completed once the associated paper is published.

---

## What is in here

| Path | Contents |
|---|---|
| `moop_healthfl.py` | The complete pipeline: MIMIC-IV feature extraction, patient-level split and patient→client assignment, certified DP-FL training (pure-NumPy MLP with per-example gradients), all ten methods, NSGA-II + LHS searches, five-level fairness evaluation, figures |
| `dp_core_v5.py` | Certified-DP core: RDP accountant for the Poisson-subsampled Gaussian mechanism, noise calibration by binary search, per-example gradients, clipping/noising, one-hot encoding, uniform validation threshold rule, operating-point metrics |
| `experiments_extra/` | Complementary harnesses: patient-clustered bootstrap CIs + paired test, membership inference attack, stay-vs-patient split ablation, reduced-round search fidelity, participation-rate ablation |
| `analysis/` | Post-hoc analyses from saved predictions (no training): main table over splits × seeds, calibration and recalibration, alert burden with patient-clustered bootstrap, global vs per-unit thresholds. `python analysis/run_posthoc.py --root <dir holding split*/> --out results_revision` |
| `tests/` | Test suite: accountant vs. Google `dp-accounting` (RDP match, PLD upper-bound check), finite-difference check of per-example gradients, clipping sensitivity, patient-disjointness, seed determinism |
| `docs/PRIVACY_LEDGER.md` | Every use of patient data and how it is covered: mechanism, design/deployment cohorts, composed cost of a search, ablation switches |
| `docs/DATA.md` | Exact data requirements: MIMIC-IV files, item IDs, cohort definition, client partition |
| `requirements.txt` | Runtime dependencies (`requirements-dev.txt` adds the test dependencies) |

No patient data is included or ever committed (see `.gitignore`); the pipeline builds its feature cache locally from your own credentialed MIMIC-IV copy.

## Installation

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

No deep-learning framework is required: certified per-example DP-SGD is implemented in a pure-NumPy backend (19–64–32–1 MLP, 3,393 parameters), which is what produced the paper's results.

To run the tests (no MIMIC-IV needed):

```bash
pip install -r requirements-dev.txt
python -m pytest tests
```

## Getting the data

The experiments use **MIMIC-IV v3.1** (PhysioNet). Access requires:

1. A [PhysioNet](https://physionet.org/) account with CITI "Data or Specimens Only Research" training,
2. Signing the MIMIC-IV data use agreement,
3. Downloading MIMIC-IV so that you have the standard layout:

```
/path/to/mimic-iv-3.1/
├── icu/    icustays.csv.gz, chartevents.csv.gz
└── hosp/   patients.csv.gz, admissions.csv.gz, labevents.csv.gz
```

Only those five files are read. See [docs/DATA.md](docs/DATA.md) for the exact item IDs, cohort filters, and the care-unit → federated-client mapping.

## Running the main experiment

```bash
# Full paper protocol: 3 seeds (42/123/456), 10 methods, NSGA-II pop=50, gen=50,
# 15 eval rounds per candidate, 50 full-training rounds, certified budgets ε ∈ [1, 10].
python moop_healthfl.py --mimic-path /path/to/mimic-iv-3.1 --output-dir output
```

- **First run** extracts features from the raw CSVs (~15–30 min) and caches them to `output/mimic_iv_features_v2.parquet` (the cache retains `subject_id` for the patient-level split); later runs start from the cache in seconds.
- **Later runs** need no `--mimic-path`: the cache in `--output-dir` (or `--cache-file`) is used directly.
- **Checkpointing**: per-seed, per-method results are checkpointed; interrupted runs resume where they stopped.
- **Saved predictions**: every trained model's validation and test probabilities, its threshold, and each client's ε, σ, sampling rate and step count are written to `output/predictions/seed<seed>_<method>.npz`; `output/predictions/eval_split.npz` holds the aligned labels, patient/stay identifiers, care unit, ethnicity, gender and insurance. Calibration, threshold, subgroup and bootstrap analyses can be run from these files without retraining. They are MIMIC-derived and must not be redistributed.
- **Parallelism**: baselines run one process per (method, seed); searches evaluate candidates across the pool. Each process uses a single BLAS thread. `--n-jobs` sets the pool size.
- **Options**: `--methods fedavg fixed_eps_1.0 …` runs a subset; `--seed` a single training seed; `--split-seed` changes the patient-level split (default 42).
- **Runtime**: the NSGA-II search is the dominant cost — approximately **9.5–10 hours per seed** with 15 parallel workers (use `--n-jobs` to control); the complete experiment (10 methods × 3 seeds, including the LHS control) took ≈ 90 hours on a 16-core CPU machine. Individual baselines take minutes.
- **Noise calibration**: the RDP accountant calibrates each client's noise multiplier once per (ε, sampling rate, step count) so that the *full 50-round protocol* satisfies the client's total (ε, δ = 10⁻⁵) budget at patient-level adjacency; calibrations are cached.
- **Smoke test** (no MIMIC-IV needed — synthetic data fallback, reduced NSGA-II):

```bash
python moop_healthfl.py --fast
```

  Note: `--fast` (pop=20, gen=30, 10 eval rounds, seed 42 only) is for pipeline verification only; paper numbers come from the full protocol.

Outputs land in `output/`: `results/aggregated_results.json` (all metrics, mean ± std over seeds), result figures (PDF), logs, and checkpoints.

## Running the complementary analyses

These reuse the cached features and the main run's checkpoint (they evaluate the reported seed-42 best-compromise allocation), so **run the main experiment first**:

```bash
python3 experiments_extra/run_bootstrap_ci.py          # ~1 h    — patient-clustered 95% CIs, paired MOOP-vs-Fixed(1.0) test, per-group / worst-group metrics
python3 experiments_extra/run_mia_attack.py            # ~1 h    — membership inference (Yeom loss-threshold + confidence), patient-disjoint members/non-members
python3 experiments_extra/run_split_ablation.py        # ~1–1.5 h — stay-level vs patient-level split (leakage quantification)
python3 experiments_extra/run_round_fidelity.py        # ~2–3 h  — Spearman fidelity of the 15-round search proxy vs 50-round training
python3 experiments_extra/run_efficiency_ablation.py   # ~1.5 h  — client participation rates 20–100%
```

Results are written to `experiments_extra/results/*.json`. If you used a non-default `--output-dir` for the main run, pass `--cache-file /your/output/mimic_iv_features_v2.parquet`. See [experiments_extra/README.md](experiments_extra/README.md) for details, including the privacy-unit note (record-level adjacency; patient-level guarantees would require per-patient clipping).

## Expected results (full protocol, mean ± std over 3 seeds, patient-level test split, validation-selected thresholds)

| Method | AUROC | AUPRC | Sensitivity | Specificity | PPV | Avg. ε (total) |
|---|---|---|---|---|---|---|
| FedAvg (no DP) | 0.773 ± 0.000 | 0.384 | 0.666 | 0.729 | 0.251 | ∞ |
| Fixed-ε(1.0) | 0.719 ± 0.001 | 0.285 | 0.685 | 0.632 | 0.203 | 1.00 |
| Fixed-ε(2.0) | 0.731 ± 0.005 | 0.302 | 0.682 | 0.651 | 0.211 | 2.00 |
| Fixed-ε(10.0) | 0.739 ± 0.001 | 0.316 | 0.684 | 0.671 | 0.221 | 10.00 |
| LHS-Search | 0.732 ± 0.002 | 0.303 | 0.688 | 0.649 | 0.211 | ≈ 2.5 |
| **MOOP-HealthFL** | **0.731 ± 0.003** | 0.303 | 0.684 | 0.653 | 0.212 | **≈ 2.5** |

Key qualitative findings (see the paper): certified privacy costs 0.034–0.054 AUROC; DP models select decision thresholds ≈ 0.005–0.01 vs ≈ 0.51 non-private, and at matched sensitivity lose 6–10 points of specificity and 4–5 points of PPV; MOOP-HealthFL finds frontier-consistent allocations automatically but does not beat the fixed-budget frontier; fairness differences across methods lie within seed variability. Complementary analyses: membership-inference attack AUC ≈ 0.50 for all methods; stay-level splitting inflates AUROC by ≈ 0.012 and manufactures apparent attainment of the 0.70 sensitivity target.

Exact values vary slightly with hardware/BLAS; the qualitative ordering and all paper conclusions are seed-stable.

## Reproducibility notes

- Seeds 42, 123, 456 control model initialisation, dropout, Poisson sampling, DP noise, and NSGA-II; a run is fully determined by its seed, and parallel and sequential execution give identical results. The patient-level data split is controlled separately by `--split-seed` (default 42) and is the same for all three seeds.
- All partitioning is at the **patient** level (70/10/20 by `subject_id`, `RandomState(42)`), and each training patient is assigned to exactly one client (modal care unit) — zero cross-client patient overlap.
- The federation is **simulated**: MIMIC-IV is partitioned by ICU care unit into five clients (single-source proxy for institutional heterogeneity) — see the paper's threat-to-validity discussion.
- All DP methods share the identical certified DP-SGD mechanism at **patient-level** adjacency: patients are Poisson-sampled, each sampled patient's summed gradient is clipped (norm 1.0), δ = 10⁻⁵, and the RDP accountant covers the **total** training run. `--privacy-unit record` gives the stay-level variant for comparison.
- Preprocessing uses fixed public clinical reference values only; no statistic is fitted to patient data. The class weight comes from a public 10% mortality prior.
- Budget searches train on a disjoint **design cohort** (50% of training patients); every reported model trains once on the **deployment cohort**, so its ε is a guarantee for the whole workflow. See [docs/PRIVACY_LEDGER.md](docs/PRIVACY_LEDGER.md).
- A uniform validation-selected decision threshold (max specificity s.t. validation sensitivity ≥ 0.70; Youden fallback) is applied to every method, private and non-private.

## License

[MIT](LICENSE). MIMIC-IV itself is governed by the PhysioNet Credentialed Health Data License — data and derived feature caches must never be redistributed through this repository.
