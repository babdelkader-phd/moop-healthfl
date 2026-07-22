# Complementary Evaluation Harnesses

Three analyses reported in the paper's "Robustness and Empirical Privacy
Analyses" and "Participation-Rate Ablation" sections. All rebuild the **exact**
federation of the main experiment (same cached features, scaler, care-unit
partition, and seed-42 test split) by importing `moop_healthfl.py`, and
evaluate the paper's published seed-42 best-compromise allocation
`{0: 0.892, 1: 0.918, 2: 0.736, 3: 1.310, 4: 0.581}` (avg ε = 0.887).

| Script | What it measures | Approx. runtime |
|---|---|---|
| `run_mia_attack.py` | Membership inference: Yeom loss-threshold + confidence attacks; attack AUC (0.5 = no leakage) and membership advantage | ~1 h |
| `run_bootstrap_ci.py` | 95% bootstrap CIs (1,000 resamples) for AUROC/sensitivity/specificity/FNR-gaps; per-ethnicity and per-gender tables; worst-group sensitivity and AUROC | ~1 h |
| `run_efficiency_ablation.py` | Client participation rates 20–100%: AUROC vs. communication cost | ~1.5 h |

## Prerequisite

The feature cache must exist (built by the first main-experiment run):

```bash
python moop_healthfl.py --mimic-path /path/to/mimic-iv   # creates output/mimic_iv_features.parquet
```

## Run

```bash
python3 experiments_extra/run_mia_attack.py
python3 experiments_extra/run_bootstrap_ci.py --n-boot 1000
python3 experiments_extra/run_efficiency_ablation.py
```

All scripts accept `--seeds` (default `42 123 456`) and `--cache-file` (if the
cache is not at `output/mimic_iv_features.parquet`). Outputs land in
`experiments_extra/results/*.json`.

## Interpretation notes (from the paper)

- **MIA**: expect attack AUC ≈ 0.50 for *all* methods, including non-private
  FedAvg — the compact 3,009-parameter model on 94k samples has a negligible
  train/validation generalization gap, which is exactly what loss-threshold
  attacks exploit. These results do **not** show that any method reduces
  attack success relative to another; stronger (gradient-level) attacks are
  needed to differentiate empirical protection.
- **Bootstrap**: models are retrained here (MOOP with the fixed seed-42
  allocation across all seeds), so point estimates differ slightly from the
  paper's headline table — expected retraining variation, stated explicitly
  in the paper.
- **Membership ground truth** for the MIA uses the per-client 80/20
  train/validation splits (disjoint by construction), balanced to a 50% base
  rate; it never uses the global test set.
