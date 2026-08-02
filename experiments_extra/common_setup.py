"""
Shared setup for the post-review evaluation harnesses — v5 CERTIFIED-DP PROTOCOL.

Rebuilds the exact experimental environment of
moop_healthfl.py (output_v5_real/):

  - patient-level 70/10/20 split, RandomState(42);
  - each PATIENT assigned to exactly one client (modal care unit of their
    training stays) -> no cross-client patient overlap;
  - one-hot encoding (19 features); continuous features standardized with a
    training-partition-only scaler;
  - certified DP-SGD (per-example clipping + RDP accountant) via the v5
    module; epsilon_k are TOTAL-training budgets in [1, 10].

Also provides a PATIENT-CLUSTERED bootstrap (resample subject_id, keep all
their stays) for honest CIs and paired method comparisons.
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

DEFAULT_CACHE = os.path.join(
    CODE_DIR, "output", "mimic_iv_features_v2.parquet")
# Fall back to the v4 cache (identical features + subject_id) if v5 not yet run.
_V4_CACHE = os.path.join(CODE_DIR, "output_v4_real", "mimic_iv_features_v2.parquet")

# v5 seed-42 best-compromise TOTAL-budget allocation.
# Placeholder until the v5 run completes; update_from_v5() overwrites it.
MOOP_ALLOCATION_SEED42 = None

SEEDS = [42, 123, 456]


def _resolve_cache(cache_file):
    if cache_file and os.path.exists(cache_file):
        return cache_file
    if os.path.exists(DEFAULT_CACHE):
        return DEFAULT_CACHE
    if os.path.exists(_V4_CACHE):
        return _V4_CACHE
    raise FileNotFoundError(
        "No feature cache found. Run moop_healthfl.py once "
        "with --mimic-path, or pass --cache-file.")


def load_v5_allocation():
    """Read the v5 seed-42 best-compromise allocation from the checkpoint."""
    import pickle
    ckpt = os.path.join(CODE_DIR, "output", "checkpoints",
                        "experiment_results.pkl")
    if not os.path.exists(ckpt):
        raise FileNotFoundError(
            f"v5 checkpoint not found ({ckpt}); run the main v5 experiment "
            "before the harnesses so the published allocation exists.")
    d = pickle.load(open(ckpt, "rb"))
    alloc = d["42"]["moop_healthfl"]["best_allocation"]
    return {int(k): float(v) for k, v in alloc.items()}


def build_federation(cache_file: str = None, seed: int = 42):
    """Recreate clients, trainer, test set and fairness evaluator, matching
    moop_healthfl.run_experiments exactly. Returns
    (clients, trainer, fairness_eval, test_demographics). trainer carries
    X_val/y_val (used for the uniform threshold rule) and each client's
    subject ids."""
    from sklearn.preprocessing import StandardScaler

    cache_file = _resolve_cache(cache_file)
    data = pd.read_parquet(cache_file)
    if "subject_id" not in data.columns:
        raise ValueError(f"{cache_file} has no subject_id column (v3 cache?).")

    subjects = np.sort(data["subject_id"].unique())
    rng_split = np.random.RandomState(42)
    perm = rng_split.permutation(len(subjects))
    n_test = int(round(m.Config.TEST_SIZE * len(subjects)))
    n_val = int(round(m.Config.VAL_SIZE * len(subjects)))
    test_subjects = set(subjects[perm[:n_test]])
    val_subjects = set(subjects[perm[n_test:n_test + n_val]])
    part = np.where(data["subject_id"].isin(test_subjects), "test",
                    np.where(data["subject_id"].isin(val_subjects), "val", "train"))
    data = data.assign(_partition=part)

    train_rows = data[data["_partition"] == "train"].copy()
    modal = (train_rows.groupby("subject_id")["hospital_id"]
             .agg(lambda s: s.value_counts().index[0]))
    train_rows["client_id"] = train_rows["subject_id"].map(modal).astype(int)

    scaler = StandardScaler()
    Xtr, _, _ = dpc.encode_features(train_rows)
    scaler.fit(Xtr[:, :dpc.N_CONTINUOUS])

    clients = []
    for i in sorted(train_rows["client_id"].unique())[:m.Config.N_CLIENTS]:
        cdf = train_rows[train_rows["client_id"] == i]
        clients.append(m.FederatedClient(int(i), cdf.reset_index(drop=True), scaler))

    def _extract(partition):
        pdf = data[data["_partition"] == partition]
        X, _, _ = dpc.encode_features(pdf)
        nan = np.isnan(X)
        if nan.any():
            med = np.nanmedian(X, axis=0)
            X[nan] = np.take(med, np.where(nan)[1])
        X[:, :dpc.N_CONTINUOUS] = scaler.transform(X[:, :dpc.N_CONTINUOUS])
        return (X.astype(np.float32), pdf["mortality"].values.astype(np.float32),
                pdf["ethnicity"].values, pdf["gender"].values,
                pdf["subject_id"].values)

    Xv, yv, ethv, genv, subv = _extract("val")
    Xt, yt, etht, gent, subt = _extract("test")

    test_demographics = pd.DataFrame({"ethnicity": etht, "gender": gent,
                                      "subject_id": subt})
    fairness_eval = m.FairnessEvaluator(test_demographics)
    trainer = m.HealthcareFLTrainer(clients, Xt, yt, seed=seed, X_val=Xv, y_val=yv)
    trainer._test_subjects = subt
    return clients, trainer, fairness_eval, test_demographics


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
