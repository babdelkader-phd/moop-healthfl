# MOOP-HealthFL  [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21499213.svg)](https://doi.org/10.5281/zenodo.21499212)

**Multi-Objective Optimization for Privacy-Preserving Federated Learning on Clinical Data: Balancing Privacy, Utility, Fairness, and Efficiency**

This repository contains the full implementation and evaluation harnesses for the MOOP-HealthFL paper. The framework formulates adaptive per-client differential privacy (DP) budget allocation in healthcare federated learning (FL) as a constrained four-objective Pareto optimization problem — privacy, utility, fairness, efficiency — solved with NSGA-II under clinical safety constraints, with a five-level fairness evaluation (client, group, individual, system, outcome).

> Citation information will be added upon publication. The repository is anonymized for review.

---

## What is in here

| Path | Contents |
|---|---|
| `moop_healthfl.py` | The complete pipeline: MIMIC-IV feature extraction, care-unit federation, DP-FL training (pure-NumPy MLP with optional TensorFlow backend), all seven baselines, NSGA-II search, five-level fairness evaluation, figures |
| `experiments_extra/` | Complementary harnesses: membership inference attack, bootstrap confidence intervals + worst-group analysis, participation-rate ablation |
| `docs/DATA.md` | Exact data requirements: MIMIC-IV files, item IDs, cohort definition, client partition |
| `requirements.txt` | Tested dependency set (Python ≥ 3.9) |

No patient data is included or ever committed (see `.gitignore`); the pipeline builds its feature cache locally from your own credentialed MIMIC-IV copy.

## Installation

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

TensorFlow is optional — without it, the pipeline uses the built-in pure-NumPy MLP (same 3,009-parameter architecture; this is what produced the paper's results).

## Getting the data

The experiments use **MIMIC-IV v2.2** (PhysioNet). Access requires:

1. A [PhysioNet](https://physionet.org/) account with CITI "Data or Specimens Only Research" training,
2. Signing the MIMIC-IV data use agreement,
3. Downloading MIMIC-IV so that you have the standard layout:

```
/path/to/mimic-iv/
├── icu/    icustays.csv.gz, chartevents.csv.gz
└── hosp/   patients.csv.gz, admissions.csv.gz, labevents.csv.gz
```

Only those five files are read. See [docs/DATA.md](docs/DATA.md) for the exact item IDs, cohort filters, and the care-unit → federated-client mapping.

## Running the main experiment

```bash
# Full paper protocol: 3 seeds (42/123/456), NSGA-II pop=50, gen=50,
# 15 eval rounds per candidate, 50 full-training rounds.
python moop_healthfl.py --mimic-path /path/to/mimic-iv --output-dir output
```

- **First run** extracts features from the raw CSVs (~15–30 min) and caches them to `output/mimic_iv_features.parquet`; later runs start from the cache in seconds.
- **Checkpointing**: per-seed, per-method results are checkpointed; interrupted runs resume where they stopped.
- **Runtime**: the NSGA-II search is the dominant cost — approximately **4–19 hours per seed** on an 8-core machine (population members are evaluated in parallel; use `--n-jobs` to control workers). Baselines take ~3 minutes each.
- **Smoke test** (no MIMIC-IV needed — synthetic data fallback, reduced NSGA-II):

```bash
python moop_healthfl.py --fast
```

  Note: `--fast` (pop=20, gen=30, 10 eval rounds, seed 42 only) is for pipeline verification only; paper numbers come from the full protocol.

Outputs land in `output/`: `results/aggregated_results.json` (all metrics, mean ± std over seeds), result figures (PDF), logs, and checkpoints.

## Running the complementary analyses

These reuse the cached features and evaluate the paper's published seed-42 best-compromise allocation (avg ε = 0.887):

```bash
python3 experiments_extra/run_mia_attack.py            # ~1 h  — membership inference (Yeom loss-threshold + confidence)
python3 experiments_extra/run_bootstrap_ci.py          # ~1 h  — 95% bootstrap CIs, per-group and worst-group metrics
python3 experiments_extra/run_efficiency_ablation.py   # ~1.5 h — client participation rates 20–100%
```

Results are written to `experiments_extra/results/*.json`. If you used a non-default `--output-dir` for the main run, pass `--cache-file /your/output/mimic_iv_features.parquet`.

## Expected results (full protocol, mean ± std over 3 seeds)

| Method | AUROC | Sensitivity | Avg. ε | Unified fairness loss |
|---|---|---|---|---|
| FedAvg (no DP) | 0.781 ± 0.002 | 0.719 ± 0.006 | ∞ | 0.065 |
| Fixed-ε(1.0) | 0.761 ± 0.001 | 0.698 ± 0.006 | 1.00 | 0.066 |
| **MOOP-HealthFL** | **0.761 ± 0.001** | 0.695 ± 0.003 | **0.909** | **0.065** |

Complementary analyses: membership-inference attack AUC ≈ 0.50 for all methods (no measurable model-level leakage at this scale); bootstrap AUROC CI half-width ≈ 0.010; participation at 20% costs ≈ 0.010 AUROC for an 80% communication reduction.

Exact values vary slightly with hardware/BLAS; the qualitative ordering and all paper conclusions are seed-stable.

## Reproducibility notes

- Seeds 42, 123, 456 control data splits, model init, DP noise, and NSGA-II.
- The federation is **simulated**: MIMIC-IV is partitioned by ICU care unit into five clients (single-source proxy for institutional heterogeneity) — see the paper's threat-to-validity discussion.
- The global test split (20%, stratified, `random_state=42`) is fixed across all methods and seeds.
- All DP methods share the identical per-round Gaussian mechanism (clip norm 1.0, δ = 10⁻⁵), so reported ε values are directly comparable.

## License

[MIT](LICENSE). MIMIC-IV itself is governed by the PhysioNet Credentialed Health Data License — data and derived feature caches must never be redistributed through this repository.
