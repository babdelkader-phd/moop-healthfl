"""
dp_core_v5.py — Certified DP-SGD core for MOOP-HealthFL v5.

Self-contained (numpy/scipy-free) implementation of:

  1. RDP accountant for the Poisson-subsampled Gaussian mechanism
     (Mironov et al., 2019 integer-alpha upper bound — the same bound used
     by TensorFlow Privacy's `_compute_log_a_int`), with RDP->(eps, delta)
     conversion following Opacus/Balle et al.
  2. Noise-multiplier calibration: binary search for the smallest sigma
     meeting a target TOTAL-TRAINING (eps, delta) over a fixed number of
     steps at sampling rate q.
  3. Per-example gradient utilities for the 2-hidden-layer MLP: vectorized
     per-example backprop, per-example l2 clipping, Gaussian sum
     perturbation.
  4. Uniform validation-based decision-threshold selection.
  5. One-hot feature encoding for the MIMIC-IV cohort.

Privacy unit: one training record = one ICU stay (add/remove adjacency).
A patient-level guarantee follows from group privacy with factor equal to
the maximum number of stays one patient contributes to a client.
"""

import math
import numpy as np

# ---------------------------------------------------------------------------
# 1. RDP accountant (Poisson-subsampled Gaussian)
# ---------------------------------------------------------------------------

DEFAULT_ALPHAS = tuple(list(range(2, 65)) + [80, 96, 128, 160, 192, 256, 384, 512])


def _log_comb(n: int, k: int) -> float:
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def _logsumexp(xs) -> float:
    m = max(xs)
    if m == -math.inf:
        return -math.inf
    return m + math.log(sum(math.exp(x - m) for x in xs))


def rdp_step_subsampled_gaussian(q: float, sigma: float, alpha: int) -> float:
    """RDP of ONE step of the Poisson-subsampled Gaussian mechanism at
    integer order alpha >= 2 (Mironov et al. 2019 upper bound)."""
    if q == 0.0:
        return 0.0
    if sigma <= 0:
        return math.inf
    if q == 1.0:
        # Plain Gaussian mechanism
        return alpha / (2.0 * sigma ** 2)
    terms = []
    log_q, log_1mq = math.log(q), math.log1p(-q)
    for k in range(alpha + 1):
        terms.append(
            _log_comb(alpha, k)
            + k * log_q
            + (alpha - k) * log_1mq
            + (k * k - k) / (2.0 * sigma ** 2)
        )
    return _logsumexp(terms) / (alpha - 1)


def eps_from_rdp(q: float, sigma: float, steps: int, delta: float,
                 alphas=DEFAULT_ALPHAS) -> float:
    """Total-training epsilon at the given delta after `steps` compositions."""
    best = math.inf
    for alpha in alphas:
        rdp = steps * rdp_step_subsampled_gaussian(q, sigma, alpha)
        if math.isinf(rdp):
            continue
        # Opacus / Balle et al. conversion
        eps = rdp + math.log((alpha - 1) / alpha) \
            - (math.log(delta) + math.log(alpha)) / (alpha - 1)
        if eps < best:
            best = eps
    return best


def calibrate_sigma(target_eps: float, delta: float, q: float, steps: int,
                    sigma_min: float = 0.3, sigma_max: float = 500.0,
                    tol: float = 1e-3) -> float:
    """Smallest sigma such that eps(sigma) <= target_eps (binary search;
    eps is monotone decreasing in sigma)."""
    if eps_from_rdp(q, sigma_min, steps, delta) <= target_eps:
        return sigma_min
    if eps_from_rdp(q, sigma_max, steps, delta) > target_eps:
        raise ValueError(
            f"target eps={target_eps} unreachable with sigma<={sigma_max} "
            f"(q={q}, steps={steps})")
    lo, hi = sigma_min, sigma_max
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if eps_from_rdp(q, mid, steps, delta) <= target_eps:
            hi = mid
        else:
            lo = mid
    return hi


_SIGMA_CACHE: dict = {}


def calibrate_sigma_cached(target_eps: float, delta: float, q: float,
                           steps: int) -> float:
    key = (round(float(target_eps), 3), delta, round(float(q), 6), int(steps))
    if key not in _SIGMA_CACHE:
        _SIGMA_CACHE[key] = calibrate_sigma(target_eps, delta, q, steps)
    return _SIGMA_CACHE[key]


# ---------------------------------------------------------------------------
# 2. Per-example gradients for the 2-hidden-layer MLP
# ---------------------------------------------------------------------------

def per_example_grads(model, y: np.ndarray, cache, w_pos: float):
    """Vectorized per-example gradients of the class-weighted BCE loss.

    `cache` is the tuple returned by model._forward(Xb, training=...).
    Weighting: positive examples weighted by the GLOBAL factor w_pos
    (n_neg_train / n_pos_train of the training partition), negatives by 1.
    Returns a list of arrays with leading batch dimension, matching the
    parameter order [W1, b1, W2, b2, W3, b3].
    """
    X, z1, a1, z2, a2, z3 = cache
    out = 1.0 / (1.0 + np.exp(-np.clip(z3, -20, 20)))
    w = np.where(y > 0.5, w_pos, 1.0).astype(np.float32)

    dz3 = (out - y) * w                                # (B, 1)
    dW3 = np.einsum('bi,bj->bij', a2, dz3)             # (B, 32, 1)
    db3 = dz3                                          # (B, 1)

    da2 = dz3 @ model.W3.T                             # (B, 32)
    dz2 = da2 * (z2 > 0)
    dW2 = np.einsum('bi,bj->bij', a1, dz2)             # (B, 64, 32)
    db2 = dz2                                          # (B, 32)

    da1 = dz2 @ model.W2.T
    dz1 = da1 * (z1 > 0)
    dW1 = np.einsum('bi,bj->bij', X, dz1)              # (B, in, 64)
    db1 = dz1                                          # (B, 64)

    return [dW1, db1, dW2, db2, dW3, db3]


def clip_and_noise(peg, clip_norm: float, sigma: float, lot_size: float,
                   rng: np.random.Generator):
    """Per-example l2-clip (over the full concatenated gradient), sum,
    add N(0, sigma^2 * C^2) to every coordinate of the SUM, divide by the
    expected lot size. Returns averaged noisy gradients in parameter order."""
    B = peg[0].shape[0]
    flat = np.concatenate([g.reshape(B, -1) for g in peg], axis=1)  # (B, P)
    norms = np.linalg.norm(flat, axis=1) + 1e-12
    factors = np.minimum(1.0, clip_norm / norms).astype(np.float32)
    out = []
    for g in peg:
        clipped_sum = np.tensordot(factors, g, axes=(0, 0))  # sum over batch
        noise = rng.normal(0.0, sigma * clip_norm, size=clipped_sum.shape)
        out.append(((clipped_sum + noise) / max(lot_size, 1.0)).astype(np.float32))
    return out


# ---------------------------------------------------------------------------
# 3. Uniform decision-threshold protocol
# ---------------------------------------------------------------------------

def select_threshold(val_probs: np.ndarray, val_y: np.ndarray,
                     target_sens: float = 0.70) -> float:
    """Prespecified rule, identical for every method:
    among validation thresholds achieving sensitivity >= target_sens,
    choose the one maximizing specificity; if none exists, fall back to the
    threshold maximizing Youden's J. Deterministic given (probs, labels)."""
    from sklearn.metrics import roc_curve
    y = val_y.ravel().astype(int)
    fpr, tpr, thr = roc_curve(y, val_probs.ravel())
    ok = tpr >= target_sens
    if ok.any():
        idx = np.argmin(fpr[ok])          # max specificity among qualifying
        chosen = thr[ok][idx]
    else:
        chosen = thr[np.argmax(tpr - fpr)]  # Youden fallback
    return float(np.clip(chosen, 1e-6, 1 - 1e-6))


def threshold_metrics(y_true: np.ndarray, probs: np.ndarray, tau: float) -> dict:
    from sklearn.metrics import roc_auc_score, average_precision_score
    y = y_true.ravel().astype(int)
    p = probs.ravel()
    yhat = (p >= tau).astype(int)
    tp = int(np.sum((yhat == 1) & (y == 1)))
    fn = int(np.sum((yhat == 0) & (y == 1)))
    tn = int(np.sum((yhat == 0) & (y == 0)))
    fp = int(np.sum((yhat == 1) & (y == 0)))
    sens = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    ppv = tp / (tp + fp) if (tp + fp) else 0.0
    npv = tn / (tn + fn) if (tn + fn) else 0.0
    try:
        auc = roc_auc_score(y, p) if len(np.unique(y)) > 1 else 0.5
        auprc = average_precision_score(y, p) if len(np.unique(y)) > 1 else 0.0
    except ValueError:
        auc, auprc = 0.5, 0.0
    return {"auc": float(auc), "auprc": float(auprc), "threshold": float(tau),
            "sensitivity": float(sens), "specificity": float(spec),
            "ppv": float(ppv), "npv": float(npv),
            "accuracy": float((tp + tn) / max(len(y), 1)),
            "predictions": yhat, "prediction_probs": p.reshape(-1, 1)}


# ---------------------------------------------------------------------------
# 4. One-hot feature encoding
# ---------------------------------------------------------------------------

CONTINUOUS_COLS = ["age", "heart_rate", "systolic_bp", "temperature",
                   "resp_rate", "spo2", "creatinine", "lactate", "wbc",
                   "hemoglobin"]
ETHNICITY_LEVELS = [0, 1, 2, 3, 4]   # White, Black, Hispanic, Asian, Other
INSURANCE_LEVELS = [0, 1, 2]         # Medicare, Medicaid, Other


def encode_features(df) -> tuple:
    """Return (X_raw float32 [N, 19], feature_names, nonprotected_idx).

    Layout: 10 continuous | gender | 5 ethnicity dummies | 3 insurance dummies.
    Continuous columns are standardized later (train-partition scaler);
    binary columns are left as 0/1. nonprotected_idx excludes gender,
    ethnicity, and insurance dummies (used by the individual-fairness kNN)."""
    parts = [df[CONTINUOUS_COLS].values.astype(np.float32),
             df[["gender"]].values.astype(np.float32)]
    names = list(CONTINUOUS_COLS) + ["gender"]
    eth = df["ethnicity"].values.astype(int)
    for lv in ETHNICITY_LEVELS:
        parts.append((eth == lv).astype(np.float32).reshape(-1, 1))
        names.append(f"ethnicity_{lv}")
    ins = df["insurance"].values.astype(int)
    for lv in INSURANCE_LEVELS:
        parts.append((ins == lv).astype(np.float32).reshape(-1, 1))
        names.append(f"insurance_{lv}")
    X = np.concatenate(parts, axis=1)
    nonprotected_idx = list(range(len(CONTINUOUS_COLS)))  # continuous only
    return X, names, nonprotected_idx


N_FEATURES = len(CONTINUOUS_COLS) + 1 + len(ETHNICITY_LEVELS) + len(INSURANCE_LEVELS)  # 19
N_CONTINUOUS = len(CONTINUOUS_COLS)
