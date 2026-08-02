"""
Split-strategy ablation: stay-level vs patient-level data splitting.

turn the central "split leakage" finding
into its own controlled experiment. Trains an identical model (same
architecture, seeds, preprocessing, and the uniform validation threshold
rule) under two splitting protocols that differ ONLY in the split unit:

  * stay-level   : 70/10/20 split over ICU STAYS (a patient's stays may
                   straddle train/val/test) -- the leaky protocol;
  * patient-level: 70/10/20 split over SUBJECTS (all of a patient's stays
                   stay together) -- the v5 protocol.

Both use the same one-hot features, the same train-only scaler, and the
same clients-from-training-rows construction. Only the partition boundary
moves. Reports test AUROC and sensitivity for FedAvg, Fixed-eps(1.0) [now
certified DP], and MOOP-HealthFL, isolating how much apparent performance
(especially sensitivity) is manufactured by stay-level leakage.

Usage:
    cd moop-healthfl-release
    python3 experiments_extra/run_split_ablation.py
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common_setup import (m, dpc, _resolve_cache, load_v5_allocation,  # noqa: E402
                          save_json, SEEDS)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "results", "split_ablation_results.json")


def build_split(data, split_unit, seed_split=42):
    """Return (clients, X_val, y_val, X_test, y_test) for the given split
    unit ('patient' or 'stay'). Patient-level assigns each patient to one
    client (modal care unit); stay-level keeps the raw care-unit client."""
    from sklearn.preprocessing import StandardScaler
    rng = np.random.RandomState(seed_split)

    if split_unit == "patient":
        units = np.sort(data["subject_id"].unique())
    else:
        units = np.sort(data.index.values)
    perm = rng.permutation(len(units))
    n_test = int(round(m.Config.TEST_SIZE * len(units)))
    n_val = int(round(m.Config.VAL_SIZE * len(units)))
    test_u = set(units[perm[:n_test]]); val_u = set(units[perm[n_test:n_test + n_val]])

    if split_unit == "patient":
        part = np.where(data["subject_id"].isin(test_u), "test",
                        np.where(data["subject_id"].isin(val_u), "val", "train"))
    else:
        idx = data.index.values
        part = np.where(np.isin(idx, list(test_u)), "test",
                        np.where(np.isin(idx, list(val_u)), "val", "train"))
    data = data.assign(_partition=part)
    train_rows = data[data["_partition"] == "train"].copy()

    if split_unit == "patient":
        modal = (train_rows.groupby("subject_id")["hospital_id"]
                 .agg(lambda s: s.value_counts().index[0]))
        train_rows["client_id"] = train_rows["subject_id"].map(modal).astype(int)
    else:
        train_rows["client_id"] = train_rows["hospital_id"]

    scaler = StandardScaler()
    Xtr, _, _ = dpc.encode_features(train_rows)
    scaler.fit(Xtr[:, :dpc.N_CONTINUOUS])
    clients = [m.FederatedClient(int(i),
               train_rows[train_rows["client_id"] == i].reset_index(drop=True), scaler)
               for i in sorted(train_rows["client_id"].unique())[:m.Config.N_CLIENTS]]

    def _extract(p):
        pdf = data[data["_partition"] == p]
        X, _, _ = dpc.encode_features(pdf)
        nan = np.isnan(X)
        if nan.any():
            med = np.nanmedian(X, axis=0); X[nan] = np.take(med, np.where(nan)[1])
        X[:, :dpc.N_CONTINUOUS] = scaler.transform(X[:, :dpc.N_CONTINUOUS])
        return X.astype(np.float32), pdf["mortality"].values.astype(np.float32)

    Xv, yv = _extract("val"); Xt, yt = _extract("test")
    return clients, Xv, yv, Xt, yt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--cache-file", default=None)
    args = ap.parse_args()

    data0 = pd.read_parquet(_resolve_cache(args.cache_file)).reset_index(drop=True)
    try:
        alloc = load_v5_allocation()
    except FileNotFoundError:
        alloc = {i: 1.0 for i in range(m.Config.N_CLIENTS)}
        print("[warn] v5 allocation not found; MOOP uses uniform eps=1.0")

    methods = {
        "FedAvg": lambda tr, s: m.run_fedavg(tr, s),
        "Fixed-eps(1.0)": lambda tr, s: m.run_fixed_eps(tr, 1.0, s),
        "MOOP-HealthFL": lambda tr, s: (tr.reset(s), tr.train_federated(
            num_rounds=m.Config.GLOBAL_ROUNDS, epsilon_allocations=alloc))[1],
    }

    results = {}
    for unit in ["stay", "patient"]:
        for seed in args.seeds:
            clients, Xv, yv, Xt, yt = build_split(data0, unit, seed_split=42)
            trainer = m.HealthcareFLTrainer(clients, Xt, yt, seed=seed,
                                            X_val=Xv, y_val=yv)
            for name, run in methods.items():
                print(f"[{unit}][seed {seed}] {name} ...")
                run(trainer, seed)
                fm = trainer.evaluate_global_model()
                results.setdefault(unit, {}).setdefault(name, {})[str(seed)] = {
                    "auroc": fm["auc"], "sensitivity": fm["sensitivity"],
                    "specificity": fm["specificity"], "threshold": fm["threshold"],
                }
                print(f"    AUROC={fm['auc']:.4f} sens={fm['sensitivity']:.3f} "
                      f"(thr={fm['threshold']:.3f})")

    # Summary table (mean over seeds)
    summary = {}
    for unit in results:
        for name in results[unit]:
            vals = results[unit][name].values()
            summary.setdefault(name, {})[unit] = {
                "auroc_mean": float(np.mean([v["auroc"] for v in vals])),
                "auroc_std": float(np.std([v["auroc"] for v in vals])),
                "sens_mean": float(np.mean([v["sensitivity"] for v in vals])),
                "sens_std": float(np.std([v["sensitivity"] for v in vals])),
            }
    print("\n=== Split-strategy ablation (mean over seeds) ===")
    for name, u in summary.items():
        s = u.get("stay", {}); p = u.get("patient", {})
        print(f"{name:16s} stay AUROC {s.get('auroc_mean',0):.3f}/sens "
              f"{s.get('sens_mean',0):.3f}  ->  patient AUROC "
              f"{p.get('auroc_mean',0):.3f}/sens {p.get('sens_mean',0):.3f}")

    save_json({"per_seed": results, "summary": summary}, OUT)


if __name__ == "__main__":
    main()
