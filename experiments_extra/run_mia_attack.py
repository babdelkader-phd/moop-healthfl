"""
Empirical membership inference attack (MIA) evaluation.

Addresses reviewer requirement: "add at least one empirical privacy attack".

Attack: Yeom et al. (CSF 2018) loss-threshold membership inference, plus a
prediction-confidence variant. For each trained global model we score every
sample by its per-sample binary cross-entropy loss (and max-confidence); an
attacker predicts "member" for low-loss samples. We report:

  - attack AUC   : ROC-AUC of the score separating members from non-members
                   (0.5 = no leakage, higher = more leakage)
  - advantage    : max over thresholds of (TPR - FPR)  [Yeom advantage]

Membership ground truth uses the per-client train/validation splits that the
federated pipeline itself uses (client.get_train_test_split(seed)): the local
80% partitions are the true training members; the local 20% partitions are
same-distribution non-members. These sets are disjoint by construction.

Expected outcome: FedAvg (no DP) leaks most; attack AUC should decrease toward
0.5 as allocated epsilon decreases; MOOP-HealthFL (avg eps=0.887) should sit
near Fixed-eps(1.0). Balanced member/non-member sampling keeps base rate 50%.

Runtime: ~5 methods x 3 seeds x (~3 min full training) ~= 45-60 min on 1 core.

Usage:
    python3 experiments_extra/run_mia_attack.py [--seeds 42 123 456]
"""

import argparse
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common_setup import build_federation, method_registry, save_json, SEEDS  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "results", "mia_results.json")
EPSILON = 1e-7


def per_sample_bce(y_true, p):
    p = np.clip(p.flatten(), EPSILON, 1 - EPSILON)
    y = y_true.flatten()
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def attack_metrics(member_scores, nonmember_scores):
    """Scores oriented so that HIGHER = more member-like."""
    from sklearn.metrics import roc_auc_score, roc_curve
    y = np.concatenate([np.ones_like(member_scores), np.zeros_like(nonmember_scores)])
    s = np.concatenate([member_scores, nonmember_scores])
    auc = roc_auc_score(y, s)
    fpr, tpr, _ = roc_curve(y, s)
    return {"attack_auc": float(auc), "advantage": float(np.max(tpr - fpr))}


def evaluate_mia(trainer, seed, rng):
    """Run both attacks against trainer.global_model."""
    model = trainer.global_model
    member_X, member_y, non_X, non_y = [], [], [], []
    for client in trainer.clients:
        X_tr, X_val, y_tr, y_val = client.get_train_test_split(seed=seed)
        member_X.append(X_tr); member_y.append(y_tr)
        non_X.append(X_val);   non_y.append(y_val)
    member_X = np.vstack(member_X); member_y = np.concatenate(member_y)
    non_X = np.vstack(non_X);       non_y = np.concatenate(non_y)

    # Balance member set down to non-member size (50% base rate)
    idx = rng.choice(len(member_y), size=len(non_y), replace=False)
    member_X, member_y = member_X[idx], member_y[idx]

    p_mem = model.predict(member_X, verbose=0).flatten()
    p_non = model.predict(non_X, verbose=0).flatten()

    # Attack 1: loss threshold (members have LOWER loss -> score = -loss)
    loss_attack = attack_metrics(-per_sample_bce(member_y, p_mem),
                                 -per_sample_bce(non_y, p_non))
    # Attack 2: confidence threshold (members have HIGHER max-confidence)
    conf_attack = attack_metrics(np.maximum(p_mem, 1 - p_mem),
                                 np.maximum(p_non, 1 - p_non))
    return {"loss_attack": loss_attack, "confidence_attack": conf_attack,
            "n_members": int(len(member_y)), "n_nonmembers": int(len(non_y))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--cache-file", default=None)
    args = ap.parse_args()

    results = {}
    for seed in args.seeds:
        kwargs = {"seed": seed}
        if args.cache_file:
            kwargs["cache_file"] = args.cache_file
        _, trainer, _, _ = build_federation(**kwargs)
        rng = np.random.default_rng(seed)
        for name, run in method_registry().items():
            print(f"[seed {seed}] training {name} ...")
            run(trainer, seed)                    # trains trainer.global_model
            mia = evaluate_mia(trainer, seed, rng)
            results.setdefault(name, {})[str(seed)] = mia
            print(f"    loss-attack AUC={mia['loss_attack']['attack_auc']:.4f} "
                  f"adv={mia['loss_attack']['advantage']:.4f}")

    # Aggregate across seeds
    summary = {}
    for name, per_seed in results.items():
        aucs = [v["loss_attack"]["attack_auc"] for v in per_seed.values()]
        advs = [v["loss_attack"]["advantage"] for v in per_seed.values()]
        summary[name] = {
            "loss_attack_auc_mean": float(np.mean(aucs)),
            "loss_attack_auc_std": float(np.std(aucs)),
            "advantage_mean": float(np.mean(advs)),
            "advantage_std": float(np.std(advs)),
        }
        print(f"{name:18s} attack AUC = {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")

    save_json({"per_seed": results, "summary": summary}, OUT)


if __name__ == "__main__":
    main()
