"""
Optimiser validity (Phase 5 of REVISION_PLAN.md, reviewer comments R2.2, R1.1).

All candidate evaluations use the search's own fitness worker
(moop_healthfl._moop_eval_worker): training on the DESIGN cohort, scoring on
validation with cross-fitted thresholds. Final models are trained on the
DEPLOYMENT cohort and scored once on the test split.

  fidelity    5.2  the same Latin-hypercube allocations evaluated at the
                   search horizon (15 rounds) and the full horizon (50
                   rounds), under several training seeds. Gives the rank
                   agreement of the proxy with the full horizon, the agreement
                   of the full horizon with itself across seeds (the ceiling
                   any proxy can reach) and the share of variance that is due
                   to the allocation rather than to the training seed.
  two-stage   5.3  shortlist from the saved 15-round archive of the main run,
                   re-evaluated at 50 rounds, final choice on the 50-round
                   values; the chosen allocation is trained on the deployment
                   cohort. Each selected allocation is paired with a UNIFORM
                   allocation of the same mean budget (matched-uniform
                   control): the difference is what heterogeneity buys.
  uniform-grid 5.4 the same selection rule applied to a grid of uniform
                   budgets only (19 candidate runs instead of about 1,270).

Usage:
    python experiments_extra/run_search_fidelity.py fidelity --split-seeds 42 43
    python experiments_extra/run_search_fidelity.py two-stage --root output_master_r2
    python experiments_extra/run_search_fidelity.py uniform-grid
"""

import argparse
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common_setup import m, _resolve_cache, CODE_DIR  # noqa: E402

K = m.Config.N_CLIENTS
SEARCH_KEYS = {"moop_healthfl": "MOOP-HealthFL", "lhs_search": "LHS search"}


# ---------------------------------------------------------------------------
# Shared plumbing
# ---------------------------------------------------------------------------

def open_split(data, split_seed, n_jobs):
    fed = m.prepare_federation(data, split_seed=split_seed)
    return m.make_pool(m.make_context(fed), n_jobs)


def evaluate(pool, X, rounds, seed):
    """Search fitness of allocations X at a given horizon -> DataFrame."""
    cfg = m._make_cfg()
    raw = m._pool_map(pool, m._moop_eval_worker,
                      [(np.asarray(x, dtype=float), rounds, seed, cfg) for x in X])
    F = np.array([r[0] for r in raw])
    G = np.array([r[1] for r in raw])
    return F, G


def fitness_frame(X, F, G):
    d = pd.DataFrame({f"eps_client{j}": np.asarray(X)[:, j] for j in range(K)})
    d["mean_eps"] = np.asarray(X).mean(axis=1)
    d["auroc"] = 1.0 - F[:, 1]
    d["fairness_objective"] = F[:, 2]
    d["sensitivity"] = m.Config.SENSITIVITY_MIN - G[:, 0]
    d["fnr_gap"] = G[:, 2] + m.Config.FAIRNESS_GAP_MAX
    d["violation"] = np.clip(G, 0, None).sum(axis=1)
    return d


def _deploy_worker(task):
    """Train one allocation on the deployment cohort; test metrics."""
    alloc, seed = task
    trainer = m._context_trainer(seed)
    trainer.reset(seed)
    eps = {j: float(alloc[j]) for j in range(K)}
    result = trainer.train_federated(num_rounds=m.Config.GLOBAL_ROUNDS,
                                     epsilon_allocations=eps)
    result["best_allocation"] = eps
    result = m.finalize_result(result, trainer)
    fm = result["final_metrics"]
    return {"auroc": fm["auc"], "auprc": fm["auprc"],
            "sensitivity": fm["sensitivity"], "specificity": fm["specificity"],
            "ppv": fm["ppv"], "threshold": fm["threshold"],
            "max_fnr_gap": result["fairness"]["max_fnr_gap"],
            "fairness_unified": result["fairness"]["fairness_unified"]}


def choose(X, F, G):
    """The main run's selection rule; index into X."""
    idx, n_feasible = m.select_front(np.asarray(X), np.asarray(F), np.asarray(G))
    return int(idx[m._best_compromise(np.asarray(F)[idx])]), n_feasible


# ---------------------------------------------------------------------------
# 5.2 Fidelity study
# ---------------------------------------------------------------------------

def run_fidelity(args):
    data = pd.read_parquet(_resolve_cache(args.cache_file))
    rows = []
    for split in args.split_seeds:
        pool = open_split(data, split, args.n_jobs)
        try:
            X = m.lhs_candidates(args.n_candidates, 1000 + split)
            for rounds in args.rounds:
                for seed in args.seeds:
                    t0 = time.time()
                    F, G = evaluate(pool, X, rounds, seed)
                    d = fitness_frame(X, F, G)
                    d.insert(0, "candidate", np.arange(len(X)))
                    d.insert(0, "seed", seed)
                    d.insert(0, "rounds", rounds)
                    d.insert(0, "split", split)
                    rows.append(d)
                    print(f"[fidelity] split {split} rounds {rounds} seed {seed}: "
                          f"{len(X)} candidates in {time.time() - t0:.0f}s", flush=True)
                    pd.concat(rows).to_csv(
                        os.path.join(args.out, "fidelity_runs.csv"), index=False)
        finally:
            if pool is not None:
                pool.close(); pool.join()
    summarise_fidelity(pd.concat(rows), args.out)


def _spearman_ci(a, b, n_boot=2000, seed=0):
    from scipy.stats import spearmanr
    a, b = np.asarray(a), np.asarray(b)
    rho = spearmanr(a, b)[0]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), size=(n_boot, len(a)))
    reps = np.array([spearmanr(a[i], b[i])[0] for i in idx])
    lo, hi = np.quantile(reps, [0.025, 0.975])
    return float(rho), float(lo), float(hi)


def summarise_fidelity(df, out):
    """Rank agreement and variance decomposition, per split and metric."""
    metrics = ["auroc", "fairness_objective", "fnr_gap", "sensitivity"]
    rounds = sorted(df["rounds"].unique())
    lo_r, hi_r = rounds[0], rounds[-1]
    agree, var = [], []
    for split, ds in df.groupby("split"):
        seeds = sorted(ds["seed"].unique())
        for metric in metrics:
            wide = ds.pivot_table(index="candidate", columns=["rounds", "seed"],
                                  values=metric)
            # proxy vs full horizon, same seed (what the search does)
            for s in seeds:
                rho, lo, hi = _spearman_ci(wide[(lo_r, s)], wide[(hi_r, s)])
                agree.append({"split": split, "metric": metric,
                              "comparison": f"{lo_r} vs {hi_r} rounds, same seed",
                              "seed_a": s, "seed_b": s, "spearman": rho,
                              "ci_lo": lo, "ci_hi": hi})
            # the same horizon under two seeds: the reproducibility ceiling
            for r in (lo_r, hi_r):
                for i, a in enumerate(seeds):
                    for b in seeds[i + 1:]:
                        rho, lo, hi = _spearman_ci(wide[(r, a)], wide[(r, b)])
                        agree.append({"split": split, "metric": metric,
                                      "comparison": f"{r} rounds, different seeds",
                                      "seed_a": a, "seed_b": b, "spearman": rho,
                                      "ci_lo": lo, "ci_hi": hi})
            # proxy averaged over seeds vs full horizon averaged over seeds
            rho, lo, hi = _spearman_ci(wide[lo_r].mean(axis=1), wide[hi_r].mean(axis=1))
            agree.append({"split": split, "metric": metric,
                          "comparison": f"{lo_r} vs {hi_r} rounds, seed-averaged",
                          "seed_a": -1, "seed_b": -1, "spearman": rho,
                          "ci_lo": lo, "ci_hi": hi})
            # one-way random-effects decomposition: allocation vs seed
            for r in rounds:
                w = wide[r].values
                n_s = w.shape[1]
                within = w.var(axis=1, ddof=1).mean()
                between = max(w.mean(axis=1).var(ddof=1) - within / n_s, 0.0)
                mean_eps = ds[ds["rounds"] == r].groupby("candidate")["mean_eps"].first()
                cand_mean = w.mean(axis=1)
                slope, icpt = np.polyfit(np.log(mean_eps.values), cand_mean, 1)
                resid = cand_mean - (slope * np.log(mean_eps.values) + icpt)
                var.append({
                    "split": split, "metric": metric, "rounds": r,
                    "sd_between_allocations": float(np.sqrt(between)),
                    "sd_between_seeds": float(np.sqrt(within)),
                    "share_allocation": float(between / (between + within))
                    if between + within > 0 else float("nan"),
                    "range_of_allocation_means": float(np.ptp(cand_mean)),
                    "r2_explained_by_mean_budget": float(
                        1.0 - resid.var() / cand_mean.var()),
                    "sd_residual_after_mean_budget": float(resid.std(ddof=1)),
                })
    agree = pd.DataFrame(agree)
    var = pd.DataFrame(var)
    agree.to_csv(os.path.join(out, "fidelity_rank_agreement_runs.csv"), index=False)
    var.to_csv(os.path.join(out, "fidelity_variance_table.csv"), index=False)
    tab = agree.groupby(["metric", "comparison"], sort=False).agg(
        spearman_mean=("spearman", "mean"), spearman_min=("spearman", "min"),
        spearman_max=("spearman", "max"), ci_lo_min=("ci_lo", "min"),
        ci_hi_max=("ci_hi", "max"), n=("spearman", "size")).reset_index()
    tab.to_csv(os.path.join(out, "fidelity_rank_agreement_table.csv"), index=False)
    print(tab.to_string(index=False))
    print(var.to_string(index=False))


# ---------------------------------------------------------------------------
# 5.3 Two-stage search and matched-uniform control
# ---------------------------------------------------------------------------

def _shortlist(X, F, G, size):
    """The `size` best-compromise members of the selected front."""
    idx, n_feasible = m.select_front(X, F, G)
    Fn = F[idx] / (F[idx].max(axis=0) + 1e-8)
    order = np.argsort(Fn.sum(axis=1), kind="stable")[:size]
    return idx[order], n_feasible


def run_two_stage(args):
    data = pd.read_parquet(_resolve_cache(args.cache_file))
    rows, short_rows = [], []
    for split in args.split_seeds:
        ckpt = os.path.join(CODE_DIR, args.root, f"split{split}", "checkpoints",
                            "experiment_results.pkl")
        with open(ckpt, "rb") as f:
            saved = pickle.load(f)
        pool = open_split(data, split, args.n_jobs)
        try:
            deploy = []          # (label dict, allocation, seed)
            for seed_s, per_method in saved.items():
                seed = int(seed_s)
                for key, name in SEARCH_KEYS.items():
                    r = per_method[key]
                    X, F, G = (np.asarray(r[k]) for k in
                               ("search_X", "search_F", "search_G"))
                    one = np.array([r["best_allocation"][j] for j in range(K)])
                    sl, n_feas = _shortlist(X, F, G, args.shortlist)
                    F50, G50 = evaluate(pool, X[sl], m.Config.GLOBAL_ROUNDS, seed)
                    pick, n_feas50 = choose(X[sl], F50, G50)
                    two = X[sl][pick]
                    from scipy.stats import spearmanr
                    d15, d50 = fitness_frame(X[sl], F[sl], G[sl]), fitness_frame(X[sl], F50, G50)
                    for i in range(len(sl)):
                        short_rows.append({
                            "split": split, "seed": seed, "search": key,
                            "rank_in_shortlist": i, "chosen_two_stage": i == pick,
                            **{f"{c}_15": d15.loc[i, c] for c in
                               ("auroc", "fairness_objective", "fnr_gap", "violation")},
                            **{f"{c}_50": d50.loc[i, c] for c in
                               ("auroc", "fairness_objective", "fnr_gap", "violation")},
                            "mean_eps": d15.loc[i, "mean_eps"]})
                    info = {"split": split, "seed": seed, "search": key,
                            "n_feasible_15": n_feas, "n_feasible_50_in_shortlist": n_feas50,
                            "shortlist": len(sl),
                            "two_stage_rank_in_shortlist": pick,
                            "shortlist_spearman_auroc": float(
                                spearmanr(d15["auroc"], d50["auroc"])[0]),
                            "shortlist_spearman_fairness": float(
                                spearmanr(d15["fairness_objective"],
                                          d50["fairness_objective"])[0])}
                    for stage, alloc in (("single_stage", one), ("two_stage", two)):
                        deploy.append(({**info, "stage": stage, "allocation": "selected"},
                                       alloc, seed))
                        deploy.append(({**info, "stage": stage,
                                        "allocation": "uniform_same_mean"},
                                       np.full(K, alloc.mean()), seed))
                    print(f"[two-stage] split {split} seed {seed} {key}: shortlist "
                          f"{len(sl)}, chosen rank {pick}", flush=True)
            out = m._pool_map(pool, _deploy_worker, [(a, s) for _, a, s in deploy])
            for (info, alloc, _), res in zip(deploy, out):
                rows.append({**info, "mean_eps": float(np.mean(alloc)),
                             **{f"eps_client{j}": float(alloc[j]) for j in range(K)},
                             **res})
        finally:
            if pool is not None:
                pool.close(); pool.join()
        pd.DataFrame(rows).to_csv(os.path.join(args.out, "two_stage_runs.csv"), index=False)
        pd.DataFrame(short_rows).to_csv(
            os.path.join(args.out, "two_stage_shortlists.csv"), index=False)
    summarise_two_stage(pd.DataFrame(rows), args.out)


def summarise_two_stage(df, out):
    keys = ["split", "seed", "search", "stage"]
    w = df.pivot_table(index=keys, columns="allocation",
                       values=["auroc", "sensitivity", "specificity",
                               "max_fnr_gap", "mean_eps"])
    paired = pd.DataFrame({
        "mean_eps": w[("mean_eps", "selected")],
        "auroc_selected": w[("auroc", "selected")],
        "auroc_uniform_same_mean": w[("auroc", "uniform_same_mean")],
        "d_auroc": w[("auroc", "selected")] - w[("auroc", "uniform_same_mean")],
        "d_specificity": w[("specificity", "selected")] - w[("specificity", "uniform_same_mean")],
        "d_fnr_gap": w[("max_fnr_gap", "selected")] - w[("max_fnr_gap", "uniform_same_mean")],
    }).reset_index()
    paired.to_csv(os.path.join(out, "matched_uniform_runs.csv"), index=False)
    g = paired.groupby(["search", "stage"], sort=False)
    tab = g.agg(runs=("d_auroc", "size"), mean_eps=("mean_eps", "mean"),
                auroc_selected=("auroc_selected", "mean"),
                auroc_uniform_same_mean=("auroc_uniform_same_mean", "mean"),
                d_auroc_mean=("d_auroc", "mean"), d_auroc_sd=("d_auroc", "std"),
                d_auroc_min=("d_auroc", "min"), d_auroc_max=("d_auroc", "max"),
                runs_selected_better=("d_auroc", lambda v: int((v > 0).sum())),
                d_specificity_mean=("d_specificity", "mean"),
                d_fnr_gap_mean=("d_fnr_gap", "mean")).reset_index()
    tab.to_csv(os.path.join(out, "matched_uniform_table.csv"), index=False)
    print(tab.to_string(index=False))


# ---------------------------------------------------------------------------
# 5.4 Uniform-budget grid under the same selection rule
# ---------------------------------------------------------------------------

def run_uniform_grid(args):
    data = pd.read_parquet(_resolve_cache(args.cache_file))
    grid = np.arange(m.Config.EPSILON_MIN, m.Config.EPSILON_MAX + 1e-9, args.step)
    X = np.repeat(grid[:, None], K, axis=1)
    rows, cand_rows = [], []
    for split in args.split_seeds:
        pool = open_split(data, split, args.n_jobs)
        try:
            picks = []
            for seed in args.seeds:
                F, G = evaluate(pool, X, m.Config.EVAL_ROUNDS, seed)
                pick, n_feas = choose(X, F, G)
                d = fitness_frame(X, F, G)
                d.insert(0, "chosen", np.arange(len(X)) == pick)
                d.insert(0, "seed", seed); d.insert(0, "split", split)
                cand_rows.append(d)
                picks.append((seed, pick, n_feas))
            out = m._pool_map(pool, _deploy_worker, [(X[p], s) for s, p, _ in picks])
            for (seed, pick, n_feas), res in zip(picks, out):
                rows.append({"split": split, "seed": seed, "method": "uniform_grid",
                             "n_candidates": len(X), "n_feasible": n_feas,
                             "mean_eps": float(grid[pick]), **res})
                print(f"[uniform-grid] split {split} seed {seed}: eps {grid[pick]:.1f}, "
                      f"test AUROC {res['auroc']:.4f}", flush=True)
        finally:
            if pool is not None:
                pool.close(); pool.join()
    pd.DataFrame(rows).to_csv(os.path.join(args.out, "uniform_grid_runs.csv"), index=False)
    pd.concat(cand_rows).to_csv(
        os.path.join(args.out, "uniform_grid_candidates.csv"), index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("experiment", choices=["fidelity", "two-stage", "uniform-grid",
                                           "summarise-fidelity"])
    ap.add_argument("--cache-file", default=None)
    ap.add_argument("--out", default=os.path.join(CODE_DIR, "results_revision_r2"))
    ap.add_argument("--root", default="output_master_r2")
    ap.add_argument("--split-seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 123])
    ap.add_argument("--rounds", type=int, nargs="+",
                    default=[m.Config.EVAL_ROUNDS, m.Config.GLOBAL_ROUNDS])
    ap.add_argument("--n-candidates", type=int, default=200)
    ap.add_argument("--shortlist", type=int, default=20)
    ap.add_argument("--step", type=float, default=0.5)
    ap.add_argument("--n-jobs", type=int, default=m.Config.N_JOBS)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    if args.experiment == "fidelity":
        run_fidelity(args)
    elif args.experiment == "summarise-fidelity":
        summarise_fidelity(pd.read_csv(os.path.join(args.out, "fidelity_runs.csv")),
                           args.out)
    elif args.experiment == "two-stage":
        run_two_stage(args)
    else:
        run_uniform_grid(args)


if __name__ == "__main__":
    main()
