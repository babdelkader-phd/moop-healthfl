"""
Reduced-round search-fidelity analysis.

Justify evaluating NSGA-II candidates at
T_eval = 15 rounds while the selected allocation is trained for T = 50.
Samples candidate allocations spanning the budget range, trains each at BOTH
15 and 50 rounds (on the validation split, as in the search), and reports
Spearman rank correlations of the search objectives between the two horizons.
High correlation => the 15-round proxy preserves the ranking that a full
50-round evaluation would produce.

Usage:
    cd moop-healthfl-release
    python3 experiments_extra/run_round_fidelity.py [--n-candidates 12]
"""

import argparse
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common_setup import (m, dpc, build_federation, save_json, SEEDS)  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "results", "round_fidelity_results.json")


def objectives(trainer, fairness_eval, alloc, n_rounds, seed):
    trainer.reset(seed)
    res = trainer.train_federated(num_rounds=n_rounds, epsilon_allocations=alloc)
    m_ = res["final_metrics"]
    y_true = trainer.y_test.flatten().astype(int)
    fair = fairness_eval.evaluate_all(
        y_true, m_["predictions"], m_["prediction_probs"], trainer.X_test,
        res["client_aucs"], alloc, {c.client_id: c.n_samples for c in trainer.clients})
    return {
        "utility_loss": 1.0 - m_["auc"],
        "fairness_loss": fair["fairness_unified"],
        "sensitivity": m_["sensitivity"],
        "obj_sum": (float(np.mean(list(alloc.values()))) / m.Config.EPSILON_MAX)
                   + (1.0 - m_["auc"]) + fair["fairness_unified"],
    }


def spearman(a, b):
    from scipy.stats import spearmanr
    r, p = spearmanr(a, b)
    return {"rho": float(r), "p": float(p)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--n-candidates", type=int, default=12)
    ap.add_argument("--cache-file", default=None)
    args = ap.parse_args()

    K = m.Config.N_CLIENTS
    lo, hi = m.Config.EPSILON_MIN, m.Config.EPSILON_MAX
    results = {}
    for seed in args.seeds:
        _, trainer, fairness_eval, _ = build_federation(
            cache_file=args.cache_file, seed=seed)
        rng = np.random.default_rng(seed)
        # LHS candidates across the budget hypercube
        strata = np.tile(np.arange(args.n_candidates, dtype=float), (K, 1)).T
        for j in range(K):
            rng.shuffle(strata[:, j])
        u = (strata + rng.random((args.n_candidates, K))) / args.n_candidates
        cands = lo + u * (hi - lo)

        rows15, rows50 = [], []
        for i, xi in enumerate(cands):
            alloc = {j: float(xi[j]) for j in range(K)}
            print(f"[seed {seed}] candidate {i+1}/{args.n_candidates} @15 ...")
            rows15.append(objectives(trainer, fairness_eval, alloc, 15, seed))
            print(f"[seed {seed}] candidate {i+1}/{args.n_candidates} @50 ...")
            rows50.append(objectives(trainer, fairness_eval, alloc, 50, seed))

        corr = {}
        for key in ["utility_loss", "fairness_loss", "sensitivity", "obj_sum"]:
            corr[key] = spearman([r[key] for r in rows15],
                                 [r[key] for r in rows50])
        results[str(seed)] = {"rows15": rows15, "rows50": rows50, "spearman": corr}
        print(f"[seed {seed}] Spearman rho: " +
              ", ".join(f"{k}={v['rho']:.3f}" for k, v in corr.items()))

    # Pooled correlation across all seeds
    allr = {k: {"r15": [], "r50": []} for k in
            ["utility_loss", "fairness_loss", "sensitivity", "obj_sum"]}
    for s in results.values():
        for k in allr:
            allr[k]["r15"] += [r[k] for r in s["rows15"]]
            allr[k]["r50"] += [r[k] for r in s["rows50"]]
    pooled = {k: spearman(v["r15"], v["r50"]) for k, v in allr.items()}
    print("\n=== Pooled Spearman (15 vs 50 rounds) ===")
    for k, v in pooled.items():
        print(f"  {k:16s} rho={v['rho']:.3f} (p={v['p']:.1e})")

    save_json({"per_seed": results, "pooled_spearman": pooled}, OUT)


if __name__ == "__main__":
    main()
