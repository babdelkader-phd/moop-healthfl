"""
Bootstrap confidence intervals + per-group fairness breakdown.

Addresses reviewer requirements:
  - "confidence intervals for fairness metrics / significance testing"
  - "group-specific sensitivity/specificity table"
  - "fairness by ethnicity and gender separately"
  - "worst-group sensitivity and worst-group AUROC"

For each headline method and seed, the final global model's test-set
predictions are bootstrapped (default 1000 resamples of the global test set)
to yield 95% percentile CIs for AUROC, sensitivity, specificity, and the
outcome-fairness (FNR-gap) component. Per-group metrics (each ethnicity and
gender subgroup) and worst-group sensitivity / worst-group AUROC are computed
on the full test set.

Runtime: ~5 methods x 3 seeds x (~3 min training) + fast bootstrap ~= 1 h.

Usage:
    python3 experiments_extra/run_bootstrap_ci.py [--n-boot 1000]
"""

import argparse
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common_setup import build_federation, method_registry, save_json, SEEDS  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "results", "bootstrap_ci_results.json")


def base_metrics(y, p, thr=0.5):
    from sklearn.metrics import roc_auc_score
    yhat = (p > thr).astype(int)
    tp = np.sum((yhat == 1) & (y == 1)); fn = np.sum((yhat == 0) & (y == 1))
    tn = np.sum((yhat == 0) & (y == 0)); fp = np.sum((yhat == 1) & (y == 0))
    sens = tp / (tp + fn) if (tp + fn) else float("nan")
    spec = tn / (tn + fp) if (tn + fp) else float("nan")
    try:
        auc = roc_auc_score(y, p)
    except ValueError:
        auc = float("nan")
    return {"auroc": float(auc), "sensitivity": float(sens), "specificity": float(spec)}


def fnr_gap(y, p, groups, thr=0.5):
    """Max pairwise FNR difference across groups (outcome-fairness gap)."""
    yhat = (p > thr).astype(int)
    fnrs = []
    for g in np.unique(groups):
        mask = (groups == g) & (y == 1)
        if mask.sum() >= 10:
            fnrs.append(np.mean(yhat[mask] == 0))
    return float(np.max(fnrs) - np.min(fnrs)) if len(fnrs) >= 2 else float("nan")


def bootstrap(y, p, eth, gen, n_boot, rng):
    stats = {"auroc": [], "sensitivity": [], "specificity": [],
             "fnr_gap_ethnicity": [], "fnr_gap_gender": []}
    n = len(y)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        bm = base_metrics(y[idx], p[idx])
        for k in ("auroc", "sensitivity", "specificity"):
            stats[k].append(bm[k])
        stats["fnr_gap_ethnicity"].append(fnr_gap(y[idx], p[idx], eth[idx]))
        stats["fnr_gap_gender"].append(fnr_gap(y[idx], p[idx], gen[idx]))
    out = {}
    for k, v in stats.items():
        v = np.array(v, dtype=float)
        v = v[~np.isnan(v)]
        out[k] = {"mean": float(np.mean(v)),
                  "ci95_low": float(np.percentile(v, 2.5)),
                  "ci95_high": float(np.percentile(v, 97.5))}
    return out


def per_group_table(y, p, groups, min_n=50):
    table = {}
    for g in np.unique(groups):
        mask = groups == g
        if mask.sum() >= min_n and len(np.unique(y[mask])) == 2:
            gm = base_metrics(y[mask], p[mask])
            gm["n"] = int(mask.sum())
            table[str(g)] = gm
    return table


def worst_group(table):
    if not table:
        return {}
    ws = min(table.values(), key=lambda d: d["sensitivity"])
    wa = min(table.values(), key=lambda d: d["auroc"])
    return {"worst_group_sensitivity": ws["sensitivity"],
            "worst_group_auroc": wa["auroc"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--cache-file", default=None)
    args = ap.parse_args()

    results = {}
    for seed in args.seeds:
        kwargs = {"seed": seed}
        if args.cache_file:
            kwargs["cache_file"] = args.cache_file
        _, trainer, _, demo = build_federation(**kwargs)
        y = trainer.y_test.flatten().astype(int)
        eth = demo["ethnicity"].values
        gen = demo["gender"].values
        rng = np.random.default_rng(seed)

        for name, run in method_registry().items():
            print(f"[seed {seed}] training {name} ...")
            run(trainer, seed)
            p = trainer.global_model.predict(trainer.X_test, verbose=0).flatten()

            entry = {
                "point": base_metrics(y, p),
                "bootstrap": bootstrap(y, p, eth, gen, args.n_boot, rng),
                "by_ethnicity": per_group_table(y, p, eth),
                "by_gender": per_group_table(y, p, gen),
            }
            entry["worst_group"] = worst_group(
                {**entry["by_ethnicity"], **entry["by_gender"]})
            results.setdefault(name, {})[str(seed)] = entry

            b = entry["bootstrap"]
            print(f"    AUROC {b['auroc']['mean']:.3f} "
                  f"[{b['auroc']['ci95_low']:.3f}, {b['auroc']['ci95_high']:.3f}]  "
                  f"worst-group sens {entry['worst_group'].get('worst_group_sensitivity', float('nan')):.3f}")

    save_json(results, OUT)


if __name__ == "__main__":
    main()
