"""
Shared setup for the complementary evaluation harnesses.

Rebuilds the exact federation of the main experiment through
moop_healthfl.prepare_federation (the single implementation of the split,
imputation, patient->client assignment and scaling), and provides a
PATIENT-CLUSTERED bootstrap (resample subject_id, keep all their stays) for
confidence intervals and paired method comparisons.
"""

import os
import sys
import json
import numpy as np
import pandas as pd

CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, CODE_DIR)

import moop_healthfl as m  # noqa: E402
import dp_core_v5 as dpc                    # noqa: E402

DEFAULT_CACHE = os.path.join(CODE_DIR, "output", m.CACHE_NAME)

SEEDS = [42, 123, 456]


def _resolve_cache(cache_file):
    if cache_file and os.path.exists(cache_file):
        return cache_file
    if os.path.exists(DEFAULT_CACHE):
        return DEFAULT_CACHE
    raise FileNotFoundError(
        "No feature cache found. Run moop_healthfl.py once "
        "with --mimic-path, or pass --cache-file.")


def load_v5_allocation():
    """Read the seed-42 selected MOOP allocation of the main run (split 42).
    The run folder is $MOOP_RUN_DIR if set, else the second master rerun
    (output_master_r2/split42), else the single-run default (output)."""
    import pickle
    candidates = [os.environ.get("MOOP_RUN_DIR"),
                  os.path.join(CODE_DIR, "output_master_r2", "split42"),
                  os.path.join(CODE_DIR, "output")]
    ckpt = None
    for c in candidates:
        if c and os.path.exists(os.path.join(c, "checkpoints",
                                             "experiment_results.pkl")):
            ckpt = os.path.join(c, "checkpoints", "experiment_results.pkl")
            break
    if ckpt is None:
        raise FileNotFoundError(
            "no main-run checkpoint found (looked in $MOOP_RUN_DIR, "
            "output_master_r2/split42, output); run the main experiment "
            "before the harnesses so the selected allocation exists.")
    with open(ckpt, "rb") as f:
        d = pickle.load(f)
    try:
        alloc = d["42"]["moop_healthfl"]["best_allocation"]
    except KeyError:
        raise FileNotFoundError(
            f"{ckpt} has no seed-42 MOOP-HealthFL result yet; finish the "
            "main experiment before running the harnesses.") from None
    print(f"[alloc] MOOP allocation from {ckpt}")
    return {int(k): float(v) for k, v in alloc.items()}


def build_federation(cache_file: str = None, seed: int = 42,
                     split_seed: int = 42, split_unit: str = "patient"):
    """Recreate clients, trainer, test set and fairness evaluator exactly as
    in moop_healthfl.run_experiments (deployment cohort; privacy unit,
    preprocessing and design fraction as set in moop_healthfl.Config). Returns
    (clients, trainer, fairness_eval, test_demographics). trainer carries
    X_val/y_val (used for the uniform threshold rule); test_demographics
    holds ethnicity, gender and subject_id per test row."""
    data = pd.read_parquet(_resolve_cache(cache_file))
    if "subject_id" not in data.columns:
        raise ValueError("feature cache has no subject_id column.")
    fed = m.prepare_federation(data, split_seed=split_seed,
                               split_unit=split_unit)
    test_demographics = fed["meta_test"][["ethnicity", "gender", "subject_id"]]
    fairness_eval = m.FairnessEvaluator(test_demographics)
    trainer = m.HealthcareFLTrainer(fed["clients"], fed["X_test"], fed["y_test"],
                                    seed=seed, X_val=fed["X_val"],
                                    y_val=fed["y_val"])
    trainer._test_subjects = test_demographics["subject_id"].values
    return fed["clients"], trainer, fairness_eval, test_demographics


def method_registry(alloc=None):
    """Headline methods. alloc = v5 seed-42 MOOP allocation (dict)."""
    def fedavg(tr, s):
        return m.run_fedavg(tr, s), None

    def fixed(eps):
        def _run(tr, s):
            return m.run_fixed_eps(tr, eps, s), {c.client_id: eps for c in tr.clients}
        return _run

    def moop(tr, s):
        a = alloc if alloc is not None else load_v5_allocation()
        tr.reset(s)
        res = tr.train_federated(num_rounds=m.Config.GLOBAL_ROUNDS,
                                 epsilon_allocations=a)
        return res, dict(a)

    return {
        "FedAvg": fedavg,
        "Fixed-eps(1.0)": fixed(1.0),
        "Fixed-eps(2.0)": fixed(2.0),
        "Fixed-eps(5.0)": fixed(5.0),
        "MOOP-HealthFL": moop,
    }


# ---------------------------------------------------------------------------
# Patient-clustered bootstrap (resample subjects, keep all their stays)
# ---------------------------------------------------------------------------

def clustered_bootstrap_indices(subject_ids: np.ndarray, rng):
    """Return row indices for one clustered bootstrap resample: sample the
    unique subjects with replacement, concatenate all rows of each drawn
    subject (so the resample size varies slightly, as usual for cluster
    bootstrap)."""
    uniq = np.unique(subject_ids)
    drawn = rng.choice(uniq, size=len(uniq), replace=True)
    by_subj = {s: np.where(subject_ids == s)[0] for s in uniq}
    return np.concatenate([by_subj[s] for s in drawn])


def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=float)
    print(f"[saved] {path}")
