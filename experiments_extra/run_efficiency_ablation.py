"""
Client participation-rate ablation (efficiency objective strengthening).

Addresses reviewer requirement: "Compare different client participation
rates: 20%, 40%, 60%, 80%, 100%" so that the efficiency objective becomes a
genuine trade-off rather than a constant.

For each participation rate r, every federated round samples
max(1, round(r*K)) clients uniformly at random; only sampled clients train
and upload updates that round. Communication cost therefore scales ~ r, while
utility may degrade at low r -- producing a real efficiency-utility curve.

Evaluated for the two headline configurations:
  - Fixed-eps(1.0)   (uniform budget baseline)
  - MOOP-HealthFL    (best-compromise seed-42 allocation)

Runtime: 2 methods x 5 rates x 3 seeds x (~3 min) ~= 90 min on 1 core.

Usage:
    python3 experiments_extra/run_efficiency_ablation.py
"""

import argparse
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common_setup import (build_federation, save_json, SEEDS,  # noqa: E402
                          MOOP_ALLOCATION_SEED42, m)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "results", "efficiency_ablation_results.json")

PARTICIPATION_RATES = [0.2, 0.4, 0.6, 0.8, 1.0]
MODEL_MB = 3009 * 4 / 1e6  # float32 parameters, one direction


def train_with_participation(trainer, eps_alloc, rate, seed):
    """Federated training where each round samples a fraction of clients.

    Mirrors HealthcareFLTrainer.train_federated() but with per-round client
    sampling; aggregation stays sample-size-weighted over the sampled subset.
    Returns (final_metrics, comm_mb_total).
    """
    rng = np.random.default_rng(seed)
    trainer.reset(seed)
    trainer.global_model = m.create_model()
    global_weights = trainer.global_model.get_weights()

    K = len(trainer.clients)
    k_sample = max(1, int(round(rate * K)))
    comm_mb = 0.0

    for _ in range(m.Config.GLOBAL_ROUNDS):
        sampled = rng.choice(K, size=k_sample, replace=False)
        client_updates = []
        for ci in sampled:
            client = trainer.clients[ci]
            eps = (eps_alloc or {}).get(client.client_id, 1e6)
            weights, meta = trainer.local_training_with_dp(client, global_weights, eps)
            client_updates.append((weights, meta["num_samples"]))
        global_weights = trainer.federated_averaging(client_updates)
        trainer.global_model.set_weights(global_weights)
        comm_mb += k_sample * 2 * MODEL_MB  # down + up per sampled client

    return trainer.evaluate_global_model(), comm_mb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--rates", type=float, nargs="+", default=PARTICIPATION_RATES)
    ap.add_argument("--cache-file", default=None)
    args = ap.parse_args()

    configs = {
        "Fixed-eps(1.0)": {c: 1.0 for c in range(m.Config.N_CLIENTS)},
        "MOOP-HealthFL": dict(MOOP_ALLOCATION_SEED42),
    }

    results = {}
    for seed in args.seeds:
        kwargs = {"seed": seed}
        if args.cache_file:
            kwargs["cache_file"] = args.cache_file
        _, trainer, _, _ = build_federation(**kwargs)
        for name, alloc in configs.items():
            for rate in args.rates:
                print(f"[seed {seed}] {name} @ participation {rate:.0%} ...")
                final, comm = train_with_participation(trainer, alloc, rate, seed)
                results.setdefault(name, {}).setdefault(f"{rate:.1f}", {})[str(seed)] = {
                    "auroc": final["auc"],
                    "sensitivity": final["sensitivity"],
                    "specificity": final["specificity"],
                    "comm_mb_total": comm,
                }
                print(f"    AUROC={final['auc']:.4f} sens={final['sensitivity']:.4f} "
                      f"comm={comm:.2f} MB")

    # Aggregate
    summary = {}
    for name, per_rate in results.items():
        summary[name] = {}
        for rate, per_seed in per_rate.items():
            aucs = [v["auroc"] for v in per_seed.values()]
            comm = [v["comm_mb_total"] for v in per_seed.values()]
            summary[name][rate] = {
                "auroc_mean": float(np.mean(aucs)), "auroc_std": float(np.std(aucs)),
                "comm_mb": float(np.mean(comm)),
            }
    save_json({"per_seed": results, "summary": summary}, OUT)


if __name__ == "__main__":
    main()
