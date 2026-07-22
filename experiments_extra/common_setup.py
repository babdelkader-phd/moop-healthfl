"""
Shared setup for the complementary evaluation harnesses (membership inference,
bootstrap confidence intervals, participation-rate ablation).

Rebuilds the exact experimental environment of moop_healthfl.py:
same cached MIMIC-IV features, same StandardScaler fit, same care-unit client
partition, same global test split (seed 42, stratified).

All scripts in this folder import from here so that every number they produce
is computed on the identical federation used in the paper.

The cached feature file is produced automatically by the first run of
    python moop_healthfl.py --mimic-path /path/to/mimic-iv
and lands at output/mimic_iv_features.parquet. Point --cache-file at it if
you used a non-default --output-dir.
"""

import os
import sys
import json
import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Import the main experiment module (repository root, one directory up)
# --------------------------------------------------------------------------
REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_DIR)

import moop_healthfl as m  # noqa: E402

# Cached parquet produced by the main experiment run
DEFAULT_CACHE = os.path.join(REPO_DIR, "output", "mimic_iv_features.parquet")

# Best-compromise per-client allocation, seed 42 (paper Table "Best Compromise
# Per-Client Privacy Allocation"; avg eps = 0.887). The harnesses evaluate this
# published operating point rather than re-running the NSGA-II search.
MOOP_ALLOCATION_SEED42 = {0: 0.892, 1: 0.918, 2: 0.736, 3: 1.310, 4: 0.581}

SEEDS = [42, 123, 456]


def build_federation(cache_file: str = DEFAULT_CACHE, seed: int = 42):
    """Recreate clients, trainer, test set and fairness evaluator.

    Mirrors run_experiments() in moop_healthfl.py exactly:
    - median imputation over feature columns,
    - StandardScaler fit on ALL rows,
    - hospital_id partition into N_CLIENTS clients,
    - global stratified 20% test split with random_state=42.
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split

    if not os.path.exists(cache_file):
        raise FileNotFoundError(
            f"Cached features not found: {cache_file}\n"
            "Run the main experiment once to build the cache:\n"
            "    python moop_healthfl.py --mimic-path /path/to/mimic-iv\n"
            "or pass --cache-file pointing at an existing "
            "mimic_iv_features.parquet.")

    data = pd.read_parquet(cache_file)

    na_count = data[m.Config.FEATURE_COLS].isnull().sum().sum()
    if na_count > 0:
        for col in m.Config.FEATURE_COLS:
            data[col] = data[col].fillna(data[col].median())

    scaler = StandardScaler()
    scaler.fit(data[m.Config.FEATURE_COLS].values.astype(np.float32))

    hospital_ids = sorted(data["hospital_id"].unique())[:m.Config.N_CLIENTS]
    clients = [
        m.FederatedClient(i, data[data["hospital_id"] == hid].reset_index(drop=True), scaler)
        for i, hid in enumerate(hospital_ids)
    ]

    all_X = np.vstack([c.X for c in clients])
    all_y = np.concatenate([c.y for c in clients])
    all_eth = np.concatenate([c.ethnicity for c in clients])
    all_gen = np.concatenate([c.gender for c in clients])

    _, X_test, _, y_test, _, eth_te, _, gen_te = train_test_split(
        all_X, all_y, all_eth, all_gen,
        test_size=m.Config.TEST_SIZE, random_state=42, stratify=all_y)

    test_demographics = pd.DataFrame({"ethnicity": eth_te, "gender": gen_te})
    fairness_eval = m.FairnessEvaluator(test_demographics)
    trainer = m.HealthcareFLTrainer(clients, X_test, y_test, seed=seed)
    return clients, trainer, fairness_eval, test_demographics


def method_registry():
    """Headline methods evaluated by the extra harnesses.

    Each entry: name -> callable(trainer, seed) -> (result_dict, eps_alloc)
    eps_alloc is the per-client epsilon dict actually used (None for non-DP).
    """
    def fedavg(tr, s):
        return m.run_fedavg(tr, s), None

    def fixed(eps):
        def _run(tr, s):
            return m.run_fixed_eps(tr, eps, s), {c.client_id: eps for c in tr.clients}
        return _run

    def moop(tr, s):
        # Full 50-round training with the paper's best-compromise allocation.
        tr.reset(s)
        res = tr.train_federated(num_rounds=m.Config.GLOBAL_ROUNDS,
                                 epsilon_allocations=MOOP_ALLOCATION_SEED42)
        return res, dict(MOOP_ALLOCATION_SEED42)

    return {
        "FedAvg": fedavg,
        "Fixed-eps(0.5)": fixed(0.5),
        "Fixed-eps(1.0)": fixed(1.0),
        "Fixed-eps(5.0)": fixed(5.0),
        "MOOP-HealthFL": moop,
    }


def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=float)
    print(f"[saved] {path}")
