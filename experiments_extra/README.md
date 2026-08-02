# Post-Review Evaluation Harnesses (v5 certified-DP protocol)

Scripts for the complementary analyses. All rebuild the **exact** federation
of the v5 certified-DP run (`output/`) by importing
`moop_healthfl.py` and `dp_core_v5.py`:

- patient-level 70/10/20 split, RandomState(42);
- each patient assigned to exactly one client (modal care unit) — no
  cross-client patient overlap;
- one-hot encoding (19 features), train-only scaler;
- certified DP-SGD (per-example clipping + RDP accountant), ε = **total**
  training budget in [1, 10];
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
cd moop-healthfl-release
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

The certified guarantee is at **record (ICU-stay) adjacency**: adding or
removing one stay. Because a patient can contribute up to 41 stays to one
client (median 1; 75% of patients have a single stay), a *patient-level*
guarantee follows only via group privacy with a factor equal to that count.
State the stay-level unit explicitly in the manuscript; a per-patient
guarantee would require grouping each patient's stays into one clipped
contribution (a possible refinement, not implemented here).
