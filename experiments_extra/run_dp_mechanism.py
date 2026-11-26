"""
What drives the score shift under certified DP, and is the DP configuration
reasonable? (REVISION_PLAN.md 3.4 and 3.5; reviewer comments R2.3, R2.4.)

  mechanism   Clipping versus noise, on the deployment cohort (test split):
              non-private; Poisson sampling without clipping or noise;
              clipping only (noise switched off) at several clipping norms;
              clipping + certified noise at the same norms; and the same
              arms without class weighting. Runs with the noise switched off
              carry NO privacy guarantee; they exist to attribute the effect.

  tuning      Sensitivity of certified training to clipping norm, lot size
              and learning rate. Candidates are trained on the DESIGN cohort
              and scored on validation, so the deployment patients' budget
              is untouched (docs/PRIVACY_LEDGER.md). The best design-cohort
              configuration is then run once per budget on the deployment
              cohort next to the default configuration.

    python experiments_extra/run_dp_mechanism.py mechanism --n-jobs 31
    python experiments_extra/run_dp_mechanism.py tuning --n-jobs 31
"""

import argparse
import itertools
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import moop_healthfl as m                      # noqa: E402
from analysis import common as C               # noqa: E402

NO_CLIP = 1e9          # clipping norm large enough never to bind
NONPRIVATE = 1e6


def _run(task):
    """Train one configuration in a worker. task = dict(label, cohort, eps,
    seed, overrides). Config overrides are restored afterwards because
    worker processes are reused."""
    saved = {k: getattr(m.Config, k) for k in task["overrides"]}
    for k, v in task["overrides"].items():
        setattr(m.Config, k, v)
    try:
        ctx = m._CTX
        if task["cohort"] == "design":
            trainer = m.HealthcareFLTrainer(ctx["design_clients"], ctx["X_val"],
                                            ctx["y_val"], seed=task["seed"])
            y_eval = ctx["y_val"]
        else:
            trainer = m._context_trainer(task["seed"])
            y_eval = ctx["y_test"]
        trainer.reset(task["seed"])
        alloc = {c.client_id: task["eps"] for c in trainer.clients}
        t0 = time.time()
        fm = trainer.train_federated(num_rounds=task.get("rounds"),
                                     epsilon_allocations=alloc)["final_metrics"]
        p = np.asarray(fm["prediction_probs"]).ravel().astype(np.float64)
        y = np.asarray(y_eval).ravel().astype(int)
        row = {k: v for k, v in task.items() if k != "overrides"}
        row.update(task["overrides"])
        row.update(auroc=fm["auc"], auprc=fm["auprc"], w_pos=trainer.w_pos,
                   elapsed_s=time.time() - t0)
        if task["cohort"] == "deployment":
            cal = C.calibration_metrics(y, p)
            matched = C.counts(y, p, C.threshold_at_sensitivity(y, p))
            row.update(threshold=fm["threshold"], sensitivity=fm["sensitivity"],
                       specificity=fm["specificity"], ppv=fm["ppv"],
                       mean_pred=cal["mean_pred"], event_rate=cal["event_rate"],
                       median_pred=float(np.median(p)), ece=cal["ece"],
                       cal_slope=cal["cal_slope"], citl=cal["citl"],
                       spec_at_sens70=matched["specificity"],
                       false_alerts_per_1000_at_sens70=matched["false_alerts_per_1000"])
        return row
    finally:
        for k, v in saved.items():
            setattr(m.Config, k, v)


def _execute(tasks, pool, out_csv):
    rows = []
    it = pool.imap_unordered(_run, tasks, chunksize=1) if pool else map(_run, tasks)
    for i, row in enumerate(it, 1):
        rows.append(row)
        print(f"  [{i}/{len(tasks)}] {row['label']} seed={row['seed']} "
              f"AUROC={row['auroc']:.3f} ({row['elapsed_s']:.0f}s)", flush=True)
    df = pd.DataFrame(rows).sort_values(["label", "seed"]).reset_index(drop=True)
    df.to_csv(out_csv, index=False)
    return df


def mechanism_tasks(seeds, eps, clips=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0)):
    unweighted = {"MORTALITY_PRIOR": 0.5}          # w_pos = 1
    arms = [("nonprivate", NONPRIVATE, {}),
            ("nonprivate_unweighted", NONPRIVATE, unweighted),
            ("sampling_only", eps, {"CLIPPING_NORM": NO_CLIP, "NOISE_SCALE": 0.0}),
            ("sampling_only_unweighted", eps,
             {"CLIPPING_NORM": NO_CLIP, "NOISE_SCALE": 0.0, **unweighted}),
            ("clip_only_C1_unweighted", eps,
             {"CLIPPING_NORM": 1.0, "NOISE_SCALE": 0.0, **unweighted}),
            ("clip_noise_C1_unweighted", eps, {"CLIPPING_NORM": 1.0, **unweighted})]
    for c in clips:
        arms.append((f"clip_only_C{c:g}", eps, {"CLIPPING_NORM": c, "NOISE_SCALE": 0.0}))
        arms.append((f"clip_noise_C{c:g}", eps, {"CLIPPING_NORM": c}))
    return [dict(label=a, cohort="deployment", eps=e, seed=s, overrides=o)
            for a, e, o in arms for s in seeds]


def tuning_tasks(seeds, budgets, clips, lots, lrs):
    grid = itertools.product(clips, lots, lrs)
    tasks = [dict(label=f"C{c:g}_B{b}_lr{lr:g}", cohort="design", eps=e, seed=s,
                  overrides={"CLIPPING_NORM": c, "BATCH_SIZE": b,
                             "DP_LEARNING_RATE": lr})
             for c, b, lr in grid for e in budgets for s in seeds]
    # Reference: does the non-private model also gain from a larger step?
    tasks += [dict(label=f"nonprivate_lr{lr:g}", cohort="design", eps=NONPRIVATE,
                   seed=s, overrides={"LEARNING_RATE": lr})
              for lr in lrs for s in seeds]
    return tasks


def design_fraction_experiment(args, tag):
    """How large should the design cohort be? For each fraction: (a) the
    cost to the deployment model (full training, test split) and (b) how
    well the search proxy (EVAL_ROUNDS rounds on the design cohort, scored
    on validation) separates uniform budgets, relative to its seed noise."""
    data = pd.read_parquet(args.cache_file)
    rows = []
    for frac in (0.25, 0.5):
        fed = m.prepare_federation(data, split_seed=args.split_seed,
                                   design_fraction=frac)
        pool = m.make_pool(m.make_context(fed), args.n_jobs)
        tasks = [dict(label=f"deployment_f{frac}", cohort="deployment", eps=e,
                      seed=s, overrides={})
                 for e in (1.0, 5.0) for s in args.seeds]
        tasks += [dict(label=f"proxy_f{frac}", cohort="design", eps=e, seed=s,
                       rounds=m.Config.EVAL_ROUNDS, overrides={})
                  for e in (1.0, 2.0, 5.0, 10.0) for s in (42, 123, 456, 789)]
        try:
            df = _execute(tasks, pool, os.path.join(
                args.out, f"design_fraction_runs_f{frac}{tag}.csv"))
        finally:
            if pool:
                pool.close(); pool.join()
        df["design_fraction"] = frac
        df["n_design_patients"] = sum(len(np.unique(c.subjects))
                                      for c in fed["design_clients"])
        rows.append(df)
    df = pd.concat(rows)
    df["kind"] = df["label"].str.split("_").str[0]
    summ = df.groupby(["kind", "design_fraction", "eps"])["auroc"].agg(["mean", "std"])
    summ.to_csv(os.path.join(args.out, f"design_fraction_table{tag}.csv"))
    print(summ.round(4).to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("experiment", choices=["mechanism", "tuning", "design-fraction"])
    ap.add_argument("--cache-file", default="output/mimic_iv_features_v2.parquet")
    ap.add_argument("--out", default="results_revision")
    ap.add_argument("--split-seed", type=int, default=42)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 123])
    ap.add_argument("--n-jobs", type=int, default=m.Config.N_JOBS)
    ap.add_argument("--clips", type=float, nargs="+", default=[1, 3, 10, 30, 100, 300])
    ap.add_argument("--mechanism-clips", type=float, nargs="+",
                    default=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
                    help="clipping norms of the mechanism ablation")
    ap.add_argument("--lots", type=int, nargs="+", default=[64, 256, 1024])
    ap.add_argument("--lrs", type=float, nargs="+", default=[1e-3, 3e-3, 1e-2])
    ap.add_argument("--confirm", type=float, nargs=3, default=None,
                    metavar=("CLIP", "LOT", "LR"),
                    help="tuning: skip the grid and confirm this configuration "
                         "against the Config defaults on the deployment cohort")
    ap.add_argument("--tag", default="", help="suffix of the output file names")
    ap.add_argument("--nonprivate-only", action="store_true",
                    help="tuning: run only the non-private learning-rate reference")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    tag = f"_{args.tag}" if args.tag else ""

    if args.experiment == "design-fraction":
        design_fraction_experiment(args, tag)
        return
    fed = m.prepare_federation(pd.read_parquet(args.cache_file),
                               split_seed=args.split_seed)
    pool = m.make_pool(m.make_context(fed), args.n_jobs)
    try:
        if args.experiment == "mechanism":
            df = _execute(mechanism_tasks(args.seeds, 1.0, args.mechanism_clips), pool,
                          os.path.join(args.out, f"dp_mechanism_runs{tag}.csv"))
            cols = ["auroc", "auprc", "mean_pred", "median_pred", "threshold",
                    "ece", "cal_slope", "citl", "spec_at_sens70",
                    "false_alerts_per_1000_at_sens70", "w_pos"]
            summ = df.groupby("label")[cols].mean()
            summ.to_csv(os.path.join(args.out, f"dp_mechanism_table{tag}.csv"))
            print(summ.round(4).to_string())
        elif args.confirm is None:
            budgets = (1.0, 5.0)
            tasks = tuning_tasks(args.seeds, budgets, args.clips, args.lots, args.lrs)
            if args.nonprivate_only:
                tasks = [t for t in tasks if t["label"].startswith("nonprivate")]
            df = _execute(tasks, pool,
                          os.path.join(args.out, f"dp_tuning_design_runs{tag}.csv"))
            ref = df[df["label"].str.startswith("nonprivate")]
            print(ref.groupby("label")["auroc"].mean().round(4).to_string())
            if args.nonprivate_only:
                return
            dp = df[~df["label"].str.startswith("nonprivate")]
            keys = ["CLIPPING_NORM", "BATCH_SIZE", "DP_LEARNING_RATE"]
            grid = (dp.pivot_table(index=keys, columns="eps", values="auroc")
                    .add_prefix("val_auroc_eps"))
            grid["val_auroc_mean"] = grid.mean(axis=1)
            grid = grid.sort_values("val_auroc_mean", ascending=False)
            grid.to_csv(os.path.join(args.out, f"dp_tuning_design_table{tag}.csv"))
            print(grid.round(4).head(15).to_string())
        else:
            c, b, lr = args.confirm
            best = {"CLIPPING_NORM": c, "BATCH_SIZE": int(b), "DP_LEARNING_RATE": lr}
            confirm = [dict(label=name, cohort="deployment", eps=e, seed=s, overrides=o)
                       for name, o in (("default", {}), ("tuned", best))
                       for e in (1.0, 2.0, 5.0, 10.0) for s in args.seeds]
            cf = _execute(confirm, pool,
                          os.path.join(args.out, f"dp_tuning_confirm_runs{tag}.csv"))
            cols = ["auroc", "auprc", "threshold", "mean_pred", "spec_at_sens70",
                    "false_alerts_per_1000_at_sens70"]
            summ = cf.groupby(["label", "eps"])[cols].mean()
            summ.to_csv(os.path.join(args.out, f"dp_tuning_confirm_table{tag}.csv"))
            print(summ.round(4).to_string())
    finally:
        if pool:
            pool.close(); pool.join()


if __name__ == "__main__":
    main()
