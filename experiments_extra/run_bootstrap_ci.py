"""
Bootstrap confidence intervals + per-group fairness breakdown.

  - "confidence intervals for fairness metrics / significance testing"
  - "group-specific sensitivity/specificity table"
  - "fairness by ethnicity and gender separately"
  - "worst-group sensitivity and worst-group AUROC"

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


def bootstrap(y, p, eth, gen, subj, n_boot, rng, thr=0.5):
    """PATIENT-CLUSTERED bootstrap: resample subject_id (keep all stays of a
    drawn subject), following the patient-level split used everywhere else."""
    from common_setup import clustered_bootstrap_indices
    stats = {"auroc": [], "sensitivity": [], "specificity": [],
             "fnr_gap_ethnicity": [], "fnr_gap_gender": []}
    for _ in range(n_boot):
        idx = clustered_bootstrap_indices(subj, rng)
        bm = base_metrics(y[idx], p[idx], thr)
        for k in ("auroc", "sensitivity", "specificity"):
            stats[k].append(bm[k])
        stats["fnr_gap_ethnicity"].append(fnr_gap(y[idx], p[idx], eth[idx], thr))
        stats["fnr_gap_gender"].append(fnr_gap(y[idx], p[idx], gen[idx], thr))
    out = {}
    for k, v in stats.items():
        v = np.array(v, dtype=float)
        v = v[~np.isnan(v)]
        out[k] = {"mean": float(np.mean(v)),
                  "ci95_low": float(np.percentile(v, 2.5)),
                  "ci95_high": float(np.percentile(v, 97.5))}
    return out


def paired_auroc_diff(y, p_a, p_b, subj, n_boot, rng):
    """Paired patient-clustered bootstrap of AUROC(method_a) - AUROC(method_b)
    on the SAME resampled patients. Returns mean diff and 95% CI; if the CI
    excludes 0 the difference is significant at ~0.05."""
    from common_setup import clustered_bootstrap_indices
    from sklearn.metrics import roc_auc_score
    diffs = []
    for _ in range(n_boot):
        idx = clustered_bootstrap_indices(subj, rng)
        yy = y[idx]
        if len(np.unique(yy)) < 2:
            continue
        diffs.append(roc_auc_score(yy, p_a[idx]) - roc_auc_score(yy, p_b[idx]))
    d = np.array(diffs)
    return {"mean_diff": float(d.mean()),
            "ci95_low": float(np.percentile(d, 2.5)),
            "ci95_high": float(np.percentile(d, 97.5)),
            "excludes_zero": bool(np.percentile(d, 2.5) > 0 or np.percentile(d, 97.5) < 0)}


def per_group_table(y, p, groups, min_n=50, thr=0.5):
    table = {}
    for g in np.unique(groups):
        mask = groups == g
        if mask.sum() >= min_n and len(np.unique(y[mask])) == 2:
            gm = base_metrics(y[mask], p[mask], thr)
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
    paired = {}
    for seed in args.seeds:
        kwargs = {"seed": seed}
        if args.cache_file:
            kwargs["cache_file"] = args.cache_file
        _, trainer, _, demo = build_federation(**kwargs)
        y = trainer.y_test.flatten().astype(int)
        eth = demo["ethnicity"].values
        gen = demo["gender"].values
        subj = demo["subject_id"].values
        rng = np.random.default_rng(seed)
        probs_by_method = {}

        for name, run in method_registry().items():
            print(f"[seed {seed}] training {name} ...")
            run(trainer, seed)
            # v5: metrics at each method's own validation-selected threshold
            fm = trainer.evaluate_global_model()
            p = fm["prediction_probs"].flatten()
            tau = fm["threshold"]
            probs_by_method[name] = p

            entry = {
                "point": {**base_metrics(y, p, tau), "threshold": tau,
                          "auprc": fm["auprc"], "ppv": fm["ppv"], "npv": fm["npv"]},
                "bootstrap": bootstrap(y, p, eth, gen, subj, args.n_boot, rng, tau),
                "by_ethnicity": per_group_table(y, p, eth, thr=tau),
                "by_gender": per_group_table(y, p, gen, thr=tau),
            }
            entry["worst_group"] = worst_group(
                {**entry["by_ethnicity"], **entry["by_gender"]})
            results.setdefault(name, {})[str(seed)] = entry

            b = entry["bootstrap"]
            print(f"    AUROC {b['auroc']['mean']:.3f} "
                  f"[{b['auroc']['ci95_low']:.3f}, {b['auroc']['ci95_high']:.3f}]  "
                  f"thr={tau:.3f}  worst-group sens "
                  f"{entry['worst_group'].get('worst_group_sensitivity', float('nan')):.3f}")

        # Paired AUROC comparison: MOOP-HealthFL vs Fixed-eps(1.0), same patients
        if "MOOP-HealthFL" in probs_by_method and "Fixed-eps(1.0)" in probs_by_method:
            paired[str(seed)] = paired_auroc_diff(
                y, probs_by_method["MOOP-HealthFL"],
                probs_by_method["Fixed-eps(1.0)"], subj, args.n_boot, rng)
            print(f"    [paired] MOOP - Fixed(1.0) AUROC diff "
                  f"{paired[str(seed)]['mean_diff']:+.4f} "
                  f"[{paired[str(seed)]['ci95_low']:+.4f}, "
                  f"{paired[str(seed)]['ci95_high']:+.4f}] "
                  f"sig={paired[str(seed)]['excludes_zero']}")

    save_json({"per_method": results, "paired_moop_vs_fixed1": paired}, OUT)


if __name__ == "__main__":
    main()
