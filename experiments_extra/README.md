# Post-Review Evaluation Harnesses (v5 certified-DP protocol)

Scripts for the complementary analyses. All rebuild the **exact** federation
of the main run through `moop_healthfl.prepare_federation` (the single
implementation of imputation, split, client assignment and scaling):

- patient-level 70/10/20 split, RandomState(42);
- each patient assigned to exactly one client (modal care unit) — no
  cross-client patient overlap;
- one-hot encoding (19 features), preprocessing from public reference
  values (nothing fitted to the data);
- 50% of each client's training patients held out as the design cohort;
  harnesses train on the deployment cohort;
- certified DP-SGD at patient level (per-patient clipping + RDP accountant),
  ε = **total** training budget in [1, 10];
- uniform validation-selected decision threshold for every method.

| Script | Item addressed | Approx. runtime |
|---|---|---|
| `run_mia_attack.py` | Membership inference (Yeom loss + confidence); patient-disjoint members/non-members | ~45–60 min |
| `run_bootstrap_ci.py` | **Patient-clustered** 95% CIs + **paired** MOOP-vs-Fixed(1.0) AUROC test; per-group tables; worst-group | ~60 min |
| `run_split_ablation.py` | Stay-level vs patient-level split, isolating leakage-manufactured performance | ~60–90 min |
| `run_round_fidelity.py` | Spearman rank fidelity of 15-round search proxy vs 50-round full training | ~2–3 h |
| `run_efficiency_ablation.py` | Participation-rate ablation (now a *post-hoc* efficiency analysis; efficiency left the objective vector in v5) | ~90 min |

## Order of operations

The main v5 experiment **must run first** — the MIA and bootstrap harnesses
evaluate the *published* v5 seed-42 best-compromise allocation, read from
`output/checkpoints/experiment_results.pkl`. If that checkpoint is
absent, `common_setup.load_v5_allocation()` raises (split ablation falls back
to uniform ε=1 with a warning).

```bash

python3 moop_healthfl.py --mimic-path "/path/to/mimic-iv-3.1" --output-dir output
# then:
python3 experiments_extra/run_bootstrap_ci.py --n-boot 1000
python3 experiments_extra/run_mia_attack.py
python3 experiments_extra/run_split_ablation.py
python3 experiments_extra/run_round_fidelity.py
python3 experiments_extra/run_efficiency_ablation.py
```

Outputs land in `experiments_extra/results/*.json`.

## Privacy unit (important)

The certified guarantee is at **patient-level adjacency**: adding or removing
one patient with all of their ICU stays. Each sampled patient's summed
gradient is clipped, so no group-privacy factor is needed. The harnesses
train on the deployment cohort, like the main run. See
[../docs/PRIVACY_LEDGER.md](../docs/PRIVACY_LEDGER.md).

## run_dp_mechanism.py (added in the revision)

`mechanism`: clipping-versus-noise ablation on the deployment cohort (non-private,
sampling only, clipping only, clipping + certified noise, with and without class
weighting, several clipping norms). Arms with the noise switched off carry no privacy
guarantee. `tuning`: sensitivity of certified training to clipping norm, lot size and
learning rate, trained on the design cohort and scored on validation, followed by one
confirmation of the best configuration on the deployment cohort. Runs in parallel
(`--n-jobs`); results go to `results_revision/dp_*.csv`.
`design-fraction`: compares design cohorts of 25% and 50% of training patients by the
cost to the deployment model and by how well the search proxy separates budgets.
