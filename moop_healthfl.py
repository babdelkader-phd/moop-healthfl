#!/usr/bin/env python3
"""
MOOP-HealthFL v3 — Parallel Command-Line Version
Multi-Objective Optimization for Privacy-Aware Healthcare Federated Learning

Converted from moop_healthfl_v3_local.ipynb, extended with multiprocessing.Pool
so NSGA-II population evaluations run in parallel across all machine cores.

Objectives (4):
  1. Privacy Loss    = mean(eps_k) / eps_max
  2. Utility Loss    = 1 - AUC
  3. Fairness Loss   = unified 5-level fairness metric
  4. Efficiency Loss = communication + compute cost

Constraints (3):
  g1: sensitivity >= SENSITIVITY_MIN (0.70)
  g2: AUC >= AUC_MIN (0.60)
  g3: fairness_gap <= FAIRNESS_GAP_MAX (0.20)

Methods (11):
  FedAvg, Fixed-eps(0.5/1.0/2.0/5.0), DP-SGD, FairFed, FCFL,
  MO-Privacy, MO-Efficiency, MOOP-HealthFL

Usage:
  python moop_healthfl_v3_parallel.py [--fast] [--n-jobs N]
                                      [--mimic-path PATH] [--output-dir DIR]
                                      [--seed SEED]
"""

import argparse
import json
import logging
import os
import pickle
import sys
import time
import warnings
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(log_dir: str) -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "experiment.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)],
    )
    return logging.getLogger(__name__)


log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

class Config:
    N_CLIENTS = 5
    TEST_SIZE = 0.2
    SEEDS = [42, 123, 456]

    FEATURE_COLS = [
        "age", "gender", "ethnicity", "insurance",
        "heart_rate", "systolic_bp", "temperature", "resp_rate", "spo2",
        "creatinine", "lactate", "wbc", "hemoglobin",
    ]

    # Training
    GLOBAL_ROUNDS = 50
    LOCAL_EPOCHS = 5
    BATCH_SIZE = 64
    LEARNING_RATE = 0.001

    # Privacy
    EPSILON_MIN = 0.5
    EPSILON_MAX = 5.0
    DELTA = 1e-5
    CLIPPING_NORM = 1.0

    # NSGA-II (full mode)
    POP_SIZE = 50
    N_GENERATIONS = 50
    N_OFFSPRINGS = 25
    EVAL_ROUNDS = 15

    # Constraints
    SENSITIVITY_MIN = 0.70
    AUC_MIN = 0.60
    FAIRNESS_GAP_MAX = 0.20

    # Fairness weights
    W_OUTCOME = 0.30
    W_GROUP = 0.25
    W_CLIENT = 0.20
    W_INDIVIDUAL = 0.15
    W_SYSTEM = 0.10

    # Parallelism
    N_JOBS = max(1, (os.cpu_count() or 2) - 1)

    @classmethod
    def apply_fast_mode(cls):
        cls.SEEDS = [42]
        cls.GLOBAL_ROUNDS = 50
        cls.POP_SIZE = 20
        cls.N_GENERATIONS = 30
        cls.N_OFFSPRINGS = 10
        cls.EVAL_ROUNDS = 10


# ---------------------------------------------------------------------------
# TensorFlow / sklearn setup
# ---------------------------------------------------------------------------

TF_AVAILABLE = False
try:
    import tensorflow as tf
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if gpus:
        try:
            policy = tf.keras.mixed_precision.Policy("mixed_float16")
            tf.keras.mixed_precision.set_global_policy(policy)
        except Exception:
            pass
    TF_AVAILABLE = True
except ImportError:
    pass


class NumpyMLPWrapper:
    """Pure-numpy MLP (13→64→32→1) with full DP-SGD support.

    Replaces TensorFlow as the fallback model — no extra dependencies,
    works on Python 3.13 / Apple Silicon.  Implements:
      - He-initialised weights
      - ReLU activations + Dropout (training mode only)
      - Class-balanced BCE loss (handles ~12% mortality imbalance)
      - Adam optimiser
      - get_weights / set_weights compatible with FedAvg
      - Exposed gradients for DP-SGD noise injection
    """

    def __init__(self, input_dim: int = 13, lr: float = 0.001):
        self.lr = lr
        self._init_weights(input_dim)

    def _init_weights(self, input_dim: int):
        rng = np.random.default_rng()
        self.W1 = (rng.standard_normal((input_dim, 64)) * np.sqrt(2 / input_dim)).astype(np.float32)
        self.b1 = np.zeros(64, dtype=np.float32)
        self.W2 = (rng.standard_normal((64, 32)) * np.sqrt(2 / 64)).astype(np.float32)
        self.b2 = np.zeros(32, dtype=np.float32)
        self.W3 = (rng.standard_normal((32, 1)) * np.sqrt(2 / 32)).astype(np.float32)
        self.b3 = np.zeros(1, dtype=np.float32)
        self._reset_adam()

    def _reset_adam(self):
        self._t = 0
        self._m = [np.zeros_like(p) for p in self._params()]
        self._v = [np.zeros_like(p) for p in self._params()]

    def _params(self):
        return [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3]

    # ------------------------------------------------------------------
    # Forward / backward
    # ------------------------------------------------------------------

    def _forward(self, X: np.ndarray, training: bool = False):
        z1 = X @ self.W1 + self.b1
        a1 = np.maximum(0, z1)
        if training:
            m1 = (np.random.rand(*a1.shape) > 0.3).astype(np.float32) / 0.7
            a1 = a1 * m1
        z2 = a1 @ self.W2 + self.b2
        a2 = np.maximum(0, z2)
        if training:
            m2 = (np.random.rand(*a2.shape) > 0.2).astype(np.float32) / 0.8
            a2 = a2 * m2
        z3 = a2 @ self.W3 + self.b3
        out = 1.0 / (1.0 + np.exp(-np.clip(z3, -20, 20)))
        return out, (X, z1, a1, z2, a2, z3)

    def _backward(self, y: np.ndarray, cache):
        X, z1, a1, z2, a2, z3 = cache
        out = 1.0 / (1.0 + np.exp(-np.clip(z3, -20, 20)))
        n = len(X)

        # Class-balanced weight: up-weight the minority positive class
        n_pos = max(float(y.sum()), 1.0)
        n_neg = max(float(n - n_pos), 1.0)
        w = np.where(y > 0.5, n_neg / n_pos, 1.0)

        dz3 = (out - y) * w / n            # (n, 1)
        dW3 = a2.T @ dz3
        db3 = dz3.sum(axis=0)

        da2 = dz3 @ self.W3.T
        dz2 = da2 * (z2 > 0)              # ReLU derivative
        dW2 = a1.T @ dz2
        db2 = dz2.sum(axis=0)

        da1 = dz2 @ self.W2.T
        dz1 = da1 * (z1 > 0)
        dW1 = X.T @ dz1
        db1 = dz1.sum(axis=0)

        return [dW1, db1, dW2, db2, dW3, db3]

    def _apply_adam(self, grads, beta1: float = 0.9, beta2: float = 0.999,
                    eps: float = 1e-8):
        self._t += 1
        t = self._t
        params = self._params()
        for i, g in enumerate(grads):
            self._m[i] = beta1 * self._m[i] + (1 - beta1) * g
            self._v[i] = beta2 * self._v[i] + (1 - beta2) * g ** 2
            m_hat = self._m[i] / (1 - beta1 ** t)
            v_hat = self._v[i] / (1 - beta2 ** t)
            params[i] -= self.lr * m_hat / (np.sqrt(v_hat) + eps)

    # ------------------------------------------------------------------
    # Public Keras-compatible interface
    # ------------------------------------------------------------------

    def predict(self, X, verbose=0) -> np.ndarray:
        out, _ = self._forward(np.array(X, dtype=np.float32), training=False)
        return out.astype(np.float32)

    def evaluate(self, X, y, verbose=0):
        from sklearn.metrics import roc_auc_score, accuracy_score
        probs = self.predict(X).flatten()
        preds = (probs > 0.5).astype(int)
        y_f = y.ravel()
        loss = float(-np.mean(y_f * np.log(probs + 1e-8) + (1 - y_f) * np.log(1 - probs + 1e-8)))
        auc = roc_auc_score(y_f, probs) if len(np.unique(y_f)) > 1 else 0.5
        acc = accuracy_score(y_f, preds)
        return loss, float(acc), float(auc)

    def get_weights(self):
        return [p.copy() for p in self._params()]

    def set_weights(self, weights):
        if not weights or len(weights) < 6:
            return
        (self.W1, self.b1, self.W2, self.b2,
         self.W3, self.b3) = [np.array(w, dtype=np.float32) for w in weights[:6]]
        self._reset_adam()  # fresh Adam state after weight injection

    def fit(self, X, y, **kwargs):
        """One-epoch fit without DP (used as a plain fallback)."""
        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.float32).reshape(-1, 1)
        bs = 64
        idx = np.random.permutation(len(X))
        for b in range(max(1, len(X) // bs)):
            s, e = b * bs, min((b + 1) * bs, len(X))
            Xb, yb = X[idx[s:e]], y[idx[s:e]]
            _, cache = self._forward(Xb, training=True)
            grads = self._backward(yb, cache)
            self._apply_adam(grads)
        return self

    def count_params(self):
        return sum(p.size for p in self._params())


def create_model(input_dim: int = 13, learning_rate: float = Config.LEARNING_RATE):
    if TF_AVAILABLE:
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation="relu", input_shape=(input_dim,),
                                  kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(32, activation="relu",
                                  kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(1, activation="sigmoid", dtype="float32"),
        ])
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss="binary_crossentropy",
            metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
        )
        return model
    return NumpyMLPWrapper(input_dim, learning_rate)


# ---------------------------------------------------------------------------
# MIMIC-IV data loading
# ---------------------------------------------------------------------------

def load_mimic_data(mimic_path: str, cache_file: str, logger: logging.Logger) -> pd.DataFrame:
    """Load MIMIC-IV from local files, using a parquet cache if available."""
    if os.path.exists(cache_file):
        logger.info(f"Loading cached features from {cache_file}")
        data = pd.read_parquet(cache_file)
        logger.info(f"Loaded {len(data):,} rows from cache.")
        return data

    logger.info("No cache found. Reading local MIMIC-IV files (15-30 min)...")
    CHUNK_SIZE = 1_000_000
    ICU_PATH = Path(mimic_path) / "icu"
    HOSP_PATH = Path(mimic_path) / "hosp"

    # Step 1: Demographics + ICU stays
    logger.info("[1/4] Loading demographics and ICU stays...")
    icu = pd.read_csv(ICU_PATH / "icustays.csv.gz",
                      usecols=["stay_id", "subject_id", "hadm_id", "first_careunit",
                               "intime", "outtime"],
                      parse_dates=["intime", "outtime"])
    patients = pd.read_csv(HOSP_PATH / "patients.csv.gz",
                           usecols=["subject_id", "gender", "anchor_age", "anchor_year", "dod"])
    admissions = pd.read_csv(HOSP_PATH / "admissions.csv.gz",
                             usecols=["hadm_id", "subject_id", "race", "insurance",
                                      "hospital_expire_flag"])

    demo = icu.merge(patients, on="subject_id", how="left")
    demo = demo.merge(admissions, on=["hadm_id", "subject_id"], how="left")
    demo["age"] = (demo["anchor_age"] + (demo["intime"].dt.year - demo["anchor_year"])).clip(18, 91)
    demo["gender"] = (demo["gender"] == "F").astype(int)
    demo["mortality"] = demo["hospital_expire_flag"].fillna(0).astype(int)
    demo["intime_24h"] = demo["intime"] + pd.Timedelta(hours=24)
    logger.info(f"  ICU stays: {len(demo):,}, Mortality: {demo['mortality'].mean():.1%}")

    # Step 2: Vital signs
    VITAL_ITEMS = {220045, 220050, 220179, 220051, 220052, 223761, 223762, 220210, 220277}
    VITAL_NAMES = {220045: "heart_rate", 220050: "systolic_bp", 220179: "systolic_bp",
                   223761: "temperature_f", 223762: "temperature_c",
                   220210: "resp_rate", 220277: "spo2"}
    logger.info("[2/4] Extracting vital signs (chunked)...")
    vitals_agg: Dict = {}
    for chunk in pd.read_csv(ICU_PATH / "chartevents.csv.gz",
                              usecols=["stay_id", "itemid", "charttime", "valuenum"],
                              chunksize=CHUNK_SIZE, parse_dates=["charttime"]):
        chunk = chunk[chunk["itemid"].isin(VITAL_ITEMS) & chunk["valuenum"].notna()]
        if len(chunk) == 0:
            continue
        chunk = chunk.merge(demo[["stay_id", "intime", "intime_24h"]], on="stay_id", how="inner")
        chunk = chunk[(chunk["charttime"] >= chunk["intime"]) &
                      (chunk["charttime"] <= chunk["intime_24h"])]
        chunk["feature"] = chunk["itemid"].map(VITAL_NAMES)
        f_mask = chunk["itemid"] == 223761
        chunk.loc[f_mask, "valuenum"] = (chunk.loc[f_mask, "valuenum"] - 32) * 5 / 9
        chunk.loc[chunk["itemid"].isin({223761, 223762}), "feature"] = "temperature"
        for (sid, feat), grp in chunk.groupby(["stay_id", "feature"]):
            key = (sid, feat)
            vals = grp["valuenum"].values
            vitals_agg[key] = np.concatenate([vitals_agg[key], vals]) if key in vitals_agg else vals
    vitals_records: Dict = {}
    for (sid, feat), vals in vitals_agg.items():
        vitals_records.setdefault(sid, {})[feat] = float(np.mean(vals))
    vitals_df = pd.DataFrame.from_dict(vitals_records, orient="index")
    vitals_df.index.name = "stay_id"
    vitals_df = vitals_df.reset_index()

    # Step 3: Lab values
    LAB_ITEMS = {50912, 50813, 51301, 50811}
    LAB_NAMES = {50912: "creatinine", 50813: "lactate", 51301: "wbc", 50811: "hemoglobin"}
    logger.info("[3/4] Extracting lab values (chunked)...")
    labs_agg: Dict = {}
    for chunk in pd.read_csv(HOSP_PATH / "labevents.csv.gz",
                              usecols=["hadm_id", "itemid", "charttime", "valuenum"],
                              chunksize=CHUNK_SIZE, parse_dates=["charttime"]):
        chunk = chunk[chunk["itemid"].isin(LAB_ITEMS) & chunk["valuenum"].notna()]
        if len(chunk) == 0:
            continue
        chunk = chunk.merge(demo[["stay_id", "hadm_id", "intime", "intime_24h"]],
                            on="hadm_id", how="inner")
        chunk = chunk[(chunk["charttime"] >= chunk["intime"]) &
                      (chunk["charttime"] <= chunk["intime_24h"])]
        chunk["feature"] = chunk["itemid"].map(LAB_NAMES)
        for (sid, feat), grp in chunk.groupby(["stay_id", "feature"]):
            key = (sid, feat)
            vals = grp["valuenum"].values
            labs_agg[key] = np.concatenate([labs_agg[key], vals]) if key in labs_agg else vals
    labs_records: Dict = {}
    for (sid, feat), vals in labs_agg.items():
        labs_records.setdefault(sid, {})[feat] = float(np.mean(vals))
    labs_df = pd.DataFrame.from_dict(labs_records, orient="index")
    labs_df.index.name = "stay_id"
    labs_df = labs_df.reset_index()

    # Step 4: Merge and finalize
    logger.info("[4/4] Merging and finalizing...")
    data = demo.merge(vitals_df, on="stay_id", how="left")
    data = data.merge(labs_df, on="stay_id", how="left")

    def _map_ethnicity(race):
        if pd.isna(race):
            return 4
        r = str(race).upper()
        if "WHITE" in r: return 0
        if "BLACK" in r or "AFRICAN" in r: return 1
        if "HISPANIC" in r or "LATINO" in r: return 2
        if "ASIAN" in r: return 3
        return 4

    def _map_insurance(ins):
        if pd.isna(ins):
            return 0
        i = str(ins).upper()
        if "MEDICARE" in i: return 0
        if "MEDICAID" in i: return 1
        return 2

    CAREUNIT_MAP = {
        "Medical Intensive Care Unit (MICU)": 0,
        "Medical/Surgical Intensive Care Unit (MICU/SICU)": 1,
        "Cardiac Vascular Intensive Care Unit (CVICU)": 2,
        "Surgical Intensive Care Unit (SICU)": 3,
        "Coronary Care Unit (CCU)": 3,
        "Trauma SICU (TSICU)": 4,
        "Neuro Surgical Intensive Care Unit (Neuro SICU)": 4,
    }

    data["ethnicity"] = data["race"].apply(_map_ethnicity)
    data["insurance"] = data["insurance"].apply(_map_insurance)
    data["hospital_id"] = data["first_careunit"].map(CAREUNIT_MAP).fillna(4).astype(int)

    clinical_cols = ["heart_rate", "systolic_bp", "temperature", "resp_rate", "spo2",
                     "creatinine", "lactate", "wbc", "hemoglobin"]
    missing_frac = data[clinical_cols].isnull().mean(axis=1)
    data = data[missing_frac <= 0.5].copy()
    for col in clinical_cols:
        data[col].fillna(data[col].median(), inplace=True)

    keep = ["stay_id", "hospital_id", "mortality",
            "age", "gender", "ethnicity", "insurance"] + clinical_cols
    data = data[keep].copy()
    data.to_parquet(cache_file, index=False)
    logger.info(f"Cached to {cache_file}")
    return data


def generate_synthetic_data(n_samples: int = 94_458, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    hospital_sizes = [26513, 23606, 16886, 16532, 10921]
    hospital_ids = np.repeat(np.arange(5), hospital_sizes)[:n_samples]
    data = {
        "hospital_id": hospital_ids,
        "age": rng.integers(18, 92, n_samples).astype(float),
        "gender": rng.integers(0, 2, n_samples).astype(float),
        "ethnicity": rng.choice([0, 1, 2, 3], n_samples, p=[0.6, 0.2, 0.1, 0.1]).astype(float),
        "insurance": rng.choice([0, 1, 2], n_samples, p=[0.5, 0.3, 0.2]).astype(float),
        "heart_rate": rng.normal(80, 15, n_samples).clip(40, 180),
        "systolic_bp": rng.normal(120, 20, n_samples).clip(70, 220),
        "temperature": rng.normal(37.0, 0.8, n_samples).clip(35.0, 40.5),
        "resp_rate": rng.normal(18, 4, n_samples).clip(8, 40),
        "spo2": rng.normal(97, 3, n_samples).clip(80, 100),
        "creatinine": rng.exponential(1.0, n_samples).clip(0.3, 15),
        "lactate": rng.exponential(1.5, n_samples).clip(0.5, 20),
        "wbc": rng.normal(10, 4, n_samples).clip(1, 50),
        "hemoglobin": rng.normal(11, 2, n_samples).clip(5, 18),
    }
    df = pd.DataFrame(data)
    risk = (0.03 * (df["age"] - 65) / 40 + 0.05 * df["lactate"] / 5
            + 0.02 * (df["creatinine"] - 1) / 5 + rng.normal(0, 0.05, n_samples))
    df["mortality"] = (risk > np.percentile(risk, 95)).astype(int)
    return df


# ---------------------------------------------------------------------------
# Differential Privacy
# ---------------------------------------------------------------------------

class ProperDPMechanism:
    def __init__(self, epsilon: float, delta: float = Config.DELTA,
                 clipping_norm: float = Config.CLIPPING_NORM):
        self.epsilon = epsilon
        self.delta = delta
        self.clipping_norm = clipping_norm
        self.noise_multiplier = np.sqrt(2 * np.log(1.25 / delta)) / epsilon

    def clip_gradients(self, gradients):
        total_norm = np.sqrt(sum(np.sum(g ** 2) for g in gradients))
        clip_factor = min(1.0, self.clipping_norm / (total_norm + 1e-8))
        return [g * clip_factor for g in gradients]

    def add_noise(self, gradients, num_samples):
        noise_std = (self.clipping_norm * self.noise_multiplier) / max(num_samples, 1)
        return [g + np.random.normal(0, noise_std, g.shape) for g in gradients]

    def privatize_gradients(self, gradients, num_samples):
        return self.add_noise(self.clip_gradients(gradients), num_samples)


# ---------------------------------------------------------------------------
# Federated Client
# ---------------------------------------------------------------------------

class FederatedClient:
    def __init__(self, client_id: int, client_data: pd.DataFrame, scaler):
        self.client_id = client_id
        self.n_samples = len(client_data)
        self.mortality_rate = client_data["mortality"].mean()
        X_raw = client_data[Config.FEATURE_COLS].values.astype(np.float32)
        # Safety: column-median imputation for any NaN that survived upstream
        nan_mask = np.isnan(X_raw)
        if nan_mask.any():
            col_medians = np.nanmedian(X_raw, axis=0)
            X_raw[nan_mask] = np.take(col_medians, np.where(nan_mask)[1])
        self.X = scaler.transform(X_raw).astype(np.float32)
        self.y = client_data["mortality"].values.astype(np.float32)
        self.ethnicity = client_data["ethnicity"].values
        self.gender = client_data["gender"].values

    def get_train_test_split(self, test_size: float = 0.2, seed: int = 42):
        from sklearn.model_selection import train_test_split
        return train_test_split(self.X, self.y,
                                test_size=test_size, random_state=seed, stratify=self.y)


# ---------------------------------------------------------------------------
# Config snapshot for worker processes
# ---------------------------------------------------------------------------

def _make_cfg() -> dict:
    return {
        "epsilon_max":      Config.EPSILON_MAX,
        "epsilon_min":      Config.EPSILON_MIN,
        "auc_min":          Config.AUC_MIN,
        "sensitivity_min":  Config.SENSITIVITY_MIN,
        "fairness_gap_max": Config.FAIRNESS_GAP_MAX,
        "n_clients":        Config.N_CLIENTS,
        "batch_size":       Config.BATCH_SIZE,
        "local_epochs":     Config.LOCAL_EPOCHS,
        "delta":            Config.DELTA,
        "clipping_norm":    Config.CLIPPING_NORM,
        "learning_rate":    Config.LEARNING_RATE,
        "w_outcome":        Config.W_OUTCOME,
        "w_group":          Config.W_GROUP,
        "w_individual":     Config.W_INDIVIDUAL,
        "w_client":         Config.W_CLIENT,
        "w_system":         Config.W_SYSTEM,
    }


# ---------------------------------------------------------------------------
# Module-level worker functions (must be top-level to pickle with spawn)
# ---------------------------------------------------------------------------

def _v3_moop_eval_worker(args):
    """Evaluate one individual for the 4-objective MOOP-HealthFL problem."""
    (xi, clients, X_test, y_test,
     eth_test, gen_test,
     eval_rounds, seed, cfg) = args

    trainer = HealthcareFLTrainer(clients, X_test, y_test, seed=seed)
    trainer.reset(seed)

    n_clients = len(clients)
    client_sizes = {c.client_id: c.n_samples for c in clients}
    eps_alloc = {j: float(xi[j]) for j in range(n_clients)}

    demographics = pd.DataFrame({"ethnicity": eth_test, "gender": gen_test})
    fairness_eval = FairnessEvaluator(demographics)

    try:
        t0 = time.time()
        result = trainer.train_federated(num_rounds=eval_rounds,
                                         epsilon_allocations=eps_alloc)
        elapsed = time.time() - t0
        m = result["final_metrics"]
        y_true = y_test.flatten().astype(int)

        f = np.zeros(4)
        f[0] = float(np.mean(xi)) / cfg["epsilon_max"]
        f[1] = 1.0 - m["auc"]

        fairness = fairness_eval.evaluate_all(
            y_true, m["predictions"], m["prediction_probs"],
            trainer.X_test, result["client_aucs"], eps_alloc, client_sizes,
        )
        f[2] = fairness["fairness_unified"]

        comm = (sum(w.nbytes for w in trainer.global_model.get_weights()) / 1e6
                if trainer.global_model else 0.01)
        f[3] = 0.4 * min(comm / 10, 1.0) + 0.6 * min(elapsed / 60, 1.0)

        g = np.zeros(3)
        g[0] = cfg["sensitivity_min"] - m["sensitivity"]
        g[1] = cfg["auc_min"] - m["auc"]
        g[2] = fairness["fairness_unified"] - cfg["fairness_gap_max"]

        return f, g
    except Exception:
        return np.ones(4), np.ones(3)


def _v3_mo_privacy_eval_worker(args):
    xi, clients, X_test, y_test, eval_rounds, seed, cfg = args
    trainer = HealthcareFLTrainer(clients, X_test, y_test, seed=seed)
    trainer.reset(seed)
    eps_alloc = {j: float(xi[j]) for j in range(len(xi))}
    try:
        result = trainer.train_federated(num_rounds=eval_rounds, epsilon_allocations=eps_alloc)
        m = result["final_metrics"]
        f = np.array([float(np.mean(xi)) / cfg["epsilon_max"], 1.0 - m["auc"]])
        g = np.array([cfg["sensitivity_min"] - m["sensitivity"], cfg["auc_min"] - m["auc"]])
        return f, g
    except Exception:
        return np.ones(2), np.ones(2)


def _v3_mo_efficiency_eval_worker(args):
    xi, clients, X_test, y_test, eval_rounds, seed, cfg = args
    trainer = HealthcareFLTrainer(clients, X_test, y_test, seed=seed)
    trainer.reset(seed)
    eps_alloc = {j: float(xi[j]) for j in range(len(xi))}
    try:
        t0 = time.time()
        result = trainer.train_federated(num_rounds=eval_rounds, epsilon_allocations=eps_alloc)
        elapsed = time.time() - t0
        m = result["final_metrics"]
        comm = (sum(w.nbytes for w in trainer.global_model.get_weights()) / 1e6
                if trainer.global_model else 0.01)
        f = np.array([0.4 * min(comm / 10, 1.0) + 0.6 * min(elapsed / 60, 1.0),
                      1.0 - m["auc"]])
        g = np.array([cfg["auc_min"] - m["auc"]])
        return f, g
    except Exception:
        return np.ones(2), np.ones(1)


# ---------------------------------------------------------------------------
# Federated Trainer
# ---------------------------------------------------------------------------

class HealthcareFLTrainer:
    def __init__(self, clients: List[FederatedClient],
                 X_test: np.ndarray, y_test: np.ndarray, seed: int = 42):
        self.clients = clients
        self.X_test = X_test
        self.y_test = y_test.reshape(-1, 1)
        self.seed = seed
        self.global_model = None
        self.training_history: List[Dict] = []

    def reset(self, seed: Optional[int] = None):
        if seed is not None:
            self.seed = seed
        np.random.seed(self.seed)
        if TF_AVAILABLE:
            tf.random.set_seed(self.seed)
        self.global_model = None
        self.training_history = []

    def local_training_with_dp(self, client: FederatedClient,
                                global_weights, epsilon: float,
                                local_epochs: Optional[int] = None):
        if local_epochs is None:
            local_epochs = Config.LOCAL_EPOCHS

        local_model = create_model()
        if global_weights is not None:
            local_model.set_weights(global_weights)

        X_train, X_val, y_train, y_val = client.get_train_test_split(seed=self.seed)
        y_train = y_train.reshape(-1, 1)
        y_val = y_val.reshape(-1, 1)

        dp = ProperDPMechanism(epsilon=epsilon)
        n_batches = max(1, len(X_train) // Config.BATCH_SIZE)

        if TF_AVAILABLE:
            for _ in range(local_epochs):
                indices = np.random.permutation(len(X_train))
                for b in range(n_batches):
                    s = b * Config.BATCH_SIZE
                    e = min(s + Config.BATCH_SIZE, len(X_train))
                    Xb = X_train[indices[s:e]]
                    yb = y_train[indices[s:e]]
                    with tf.GradientTape() as tape:
                        preds = local_model(Xb, training=True)
                        loss = tf.reduce_mean(
                            tf.keras.losses.binary_crossentropy(yb, preds))
                    grads = tape.gradient(loss, local_model.trainable_variables)
                    grads_np = [g.numpy() for g in grads]
                    priv_grads = dp.privatize_gradients(grads_np, e - s)
                    local_model.optimizer.apply_gradients(
                        zip([tf.constant(g.astype(np.float32)) for g in priv_grads],
                            local_model.trainable_variables))
        else:
            # Numpy DP-SGD: same gradient-clip+noise pattern, no TF required
            for _ in range(local_epochs):
                indices = np.random.permutation(len(X_train))
                for b in range(n_batches):
                    s = b * Config.BATCH_SIZE
                    e = min(s + Config.BATCH_SIZE, len(X_train))
                    Xb = X_train[indices[s:e]].astype(np.float32)
                    yb = y_train[indices[s:e]]
                    _, cache = local_model._forward(Xb, training=True)
                    grads = local_model._backward(yb, cache)
                    priv_grads = dp.privatize_gradients(grads, e - s)
                    local_model._apply_adam(priv_grads)

        val_loss, val_acc, val_auc = local_model.evaluate(X_val, y_val, verbose=0)
        return local_model.get_weights(), {
            "val_loss": float(val_loss), "val_accuracy": float(val_acc),
            "val_auc": float(val_auc), "num_samples": len(X_train),
        }

    def federated_averaging(self, client_updates):
        total = sum(n for _, n in client_updates)
        avg = [np.zeros_like(w) for w in client_updates[0][0]]
        for weights, n in client_updates:
            factor = n / total
            for i, w in enumerate(weights):
                avg[i] += w * factor
        return avg

    def weighted_federated_averaging(self, client_updates, custom_weights):
        total_w = sum(custom_weights)
        avg = [np.zeros_like(w) for w in client_updates[0][0]]
        for (weights, _), cw in zip(client_updates, custom_weights):
            factor = cw / total_w
            for i, w in enumerate(weights):
                avg[i] += w * factor
        return avg

    def evaluate_global_model(self) -> Dict:
        from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
        if self.global_model is None:
            return {"accuracy": 0.0, "auc": 0.5, "loss": 1.0,
                    "sensitivity": 0.0, "specificity": 0.0,
                    "predictions": np.array([]), "prediction_probs": np.array([])}
        loss, accuracy, auc = self.global_model.evaluate(
            self.X_test, self.y_test, verbose=0)
        y_pred_prob = self.global_model.predict(self.X_test, verbose=0)
        y_pred = (y_pred_prob > 0.5).astype(int).flatten()
        y_true = self.y_test.flatten().astype(int)
        try:
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        except ValueError:
            sens, spec = 0.0, 0.0
        return {
            "loss": float(loss), "accuracy": float(accuracy), "auc": float(auc),
            "sensitivity": float(sens), "specificity": float(spec),
            "predictions": y_pred, "prediction_probs": y_pred_prob,
        }

    def evaluate_per_client(self) -> List[float]:
        from sklearn.metrics import roc_auc_score
        aucs = []
        for client in self.clients:
            _, X_val, _, y_val = client.get_train_test_split(seed=self.seed)
            probs = self.global_model.predict(X_val, verbose=0).flatten()
            try:
                aucs.append(float(roc_auc_score(y_val, probs)))
            except Exception:
                aucs.append(0.5)
        return aucs

    def train_federated(self, num_rounds: Optional[int] = None,
                        epsilon_allocations: Optional[Dict] = None,
                        aggregation_fn=None, verbose: bool = False) -> Dict:
        if num_rounds is None:
            num_rounds = Config.GLOBAL_ROUNDS

        self.global_model = create_model()
        global_weights = self.global_model.get_weights()
        self.training_history = []

        for rnd in range(num_rounds):
            client_updates = []
            for client in self.clients:
                eps = (epsilon_allocations or {}).get(client.client_id, 1e6)
                weights, meta = self.local_training_with_dp(client, global_weights, eps)
                client_updates.append((weights, meta["num_samples"]))

            if aggregation_fn:
                global_weights = aggregation_fn(client_updates, rnd, self)
            else:
                global_weights = self.federated_averaging(client_updates)
            self.global_model.set_weights(global_weights)

            metrics = self.evaluate_global_model()
            self.training_history.append({
                "round": rnd + 1, "auc": metrics["auc"],
                "accuracy": metrics["accuracy"], "loss": metrics["loss"],
                "sensitivity": metrics["sensitivity"],
            })

            if verbose and (rnd + 1) % 10 == 0:
                log.info(f"  Round {rnd+1}/{num_rounds}: AUC={metrics['auc']:.3f}, "
                         f"Sens={metrics['sensitivity']:.3f}")

        final = self.evaluate_global_model()
        return {
            "final_metrics": final,
            "client_aucs": self.evaluate_per_client(),
            "training_history": self.training_history,
        }


# ---------------------------------------------------------------------------
# 5-Level Fairness Evaluator
# ---------------------------------------------------------------------------

class FairnessEvaluator:
    def __init__(self, test_demographics: pd.DataFrame):
        self.demographics = test_demographics

    def client_level(self, client_aucs) -> float:
        variance = np.var(list(client_aucs) if isinstance(client_aucs, dict)
                          else client_aucs)
        return max(0.0, 1.0 - variance / 0.25)

    def group_level(self, y_true, y_pred, y_pred_prob) -> float:
        dp_gaps, eo_gaps = [], []
        for attr in ["ethnicity", "gender"]:
            values = self.demographics[attr].values
            groups = np.unique(values)
            rates = [np.mean(y_pred[values == g]) for g in groups
                     if (values == g).sum() > 10]
            if len(rates) >= 2:
                dp_gaps.append(max(rates) - min(rates))
            for yv in [0, 1]:
                cond = [np.mean(y_pred[(values == g) & (y_true == yv)])
                        for g in groups if ((values == g) & (y_true == yv)).sum() > 10]
                if len(cond) >= 2:
                    eo_gaps.append(max(cond) - min(cond))
        avg_dp = np.mean(dp_gaps) if dp_gaps else 0
        avg_eo = np.mean(eo_gaps) if eo_gaps else 0
        return max(0.0, 1.0 - 0.5 * (avg_dp + avg_eo))

    def individual_level(self, X_test: np.ndarray, y_pred_prob: np.ndarray,
                         k: int = 10, n_sub: int = 5000) -> float:
        from sklearn.neighbors import NearestNeighbors
        n = min(len(X_test), n_sub)
        idx = np.random.choice(len(X_test), n, replace=False)
        X_sub, p_sub = X_test[idx], y_pred_prob.flatten()[idx]
        nn = NearestNeighbors(n_neighbors=k + 1, algorithm="ball_tree").fit(X_sub)
        _, indices = nn.kneighbors(X_sub)
        return max(0.0, float(np.mean([1.0 - np.var(p_sub[indices[i, 1:]]) for i in range(n)])))

    def outcome_level(self, y_true, y_pred) -> float:
        fnr_gaps = []
        for attr in ["ethnicity", "gender"]:
            values = self.demographics[attr].values
            groups = np.unique(values)
            fnrs = []
            for g in groups:
                mask = (values == g) & (y_true == 1)
                if mask.sum() > 5:
                    fnrs.append(np.mean(y_pred[mask] == 0))
            if len(fnrs) >= 2:
                fnr_gaps.append(max(fnrs) - min(fnrs))
        return max(0.0, 1.0 - (np.mean(fnr_gaps) if fnr_gaps else 0))

    def system_level(self, eps_alloc: Dict, client_sizes: Dict) -> float:
        if not eps_alloc:
            return 1.0
        ratios = np.array(sorted(
            eps_alloc.get(k, 1.0) / max(client_sizes.get(k, 1), 1) for k in eps_alloc))
        n = len(ratios)
        if n == 0 or np.sum(ratios) == 0:
            return 1.0
        index = np.arange(1, n + 1)
        gini = (2 * np.sum(index * ratios) / (n * np.sum(ratios))) - (n + 1) / n
        return max(0.0, 1.0 - abs(gini))

    def evaluate_all(self, y_true, y_pred, y_pred_prob, X_test,
                     client_aucs, eps_alloc, client_sizes) -> Dict:
        f_client = self.client_level(client_aucs)
        f_group = self.group_level(y_true, y_pred, y_pred_prob)
        f_individual = self.individual_level(X_test, y_pred_prob)
        f_system = self.system_level(eps_alloc or {}, client_sizes)
        f_outcome = self.outcome_level(y_true, y_pred)
        unified = (Config.W_OUTCOME * f_outcome + Config.W_GROUP * f_group
                   + Config.W_CLIENT * f_client + Config.W_INDIVIDUAL * f_individual
                   + Config.W_SYSTEM * f_system)
        return {
            "fairness_client": float(f_client), "fairness_group": float(f_group),
            "fairness_individual": float(f_individual), "fairness_system": float(f_system),
            "fairness_outcome": float(f_outcome), "fairness_unified": float(1.0 - unified),
        }


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------

def run_fedavg(trainer: HealthcareFLTrainer, seed: int = 42) -> Dict:
    trainer.reset(seed)
    eps = {c.client_id: 1e6 for c in trainer.clients}
    result = trainer.train_federated(num_rounds=Config.GLOBAL_ROUNDS, epsilon_allocations=eps)
    result["method"] = "FedAvg"
    result["avg_epsilon"] = float("inf")
    return result


def run_fixed_eps(trainer: HealthcareFLTrainer, epsilon: float, seed: int = 42) -> Dict:
    trainer.reset(seed)
    eps = {c.client_id: epsilon for c in trainer.clients}
    result = trainer.train_federated(num_rounds=Config.GLOBAL_ROUNDS, epsilon_allocations=eps)
    result["method"] = f"Fixed-eps({epsilon})"
    result["avg_epsilon"] = epsilon
    return result


def run_dpsgd_standalone(trainer: HealthcareFLTrainer, epsilon: float = 1.0,
                          seed: int = 42) -> Dict:
    trainer.reset(seed)
    n_avg = np.mean([c.n_samples for c in trainer.clients])
    total_steps = Config.GLOBAL_ROUNDS * Config.LOCAL_EPOCHS * int(n_avg * 0.8 / Config.BATCH_SIZE)
    eps_per_step = epsilon / np.sqrt(2 * total_steps * np.log(1 / Config.DELTA))
    eps = {c.client_id: max(eps_per_step * np.sqrt(total_steps / Config.GLOBAL_ROUNDS), 0.1)
           for c in trainer.clients}
    result = trainer.train_federated(num_rounds=Config.GLOBAL_ROUNDS, epsilon_allocations=eps)
    result["method"] = f"DP-SGD(eps={epsilon})"
    result["avg_epsilon"] = epsilon
    return result


def run_fairfed(trainer: HealthcareFLTrainer, seed: int = 42, lambda_fair: float = 1.0) -> Dict:
    from sklearn.model_selection import train_test_split as tts
    trainer.reset(seed)

    def _agg(updates, rnd, t):
        if t.global_model is None:
            return t.federated_averaging(updates)
        gaps = []
        for client in t.clients:
            _, X_val, _, y_val = client.get_train_test_split(seed=t.seed)
            preds = (t.global_model.predict(X_val, verbose=0) > 0.5).astype(int).flatten()
            _, _, _, eth_val = tts(client.ethnicity, client.ethnicity,
                                   test_size=0.2, random_state=t.seed)
            rates = [np.mean(preds[eth_val == g])
                     for g in np.unique(eth_val) if (eth_val == g).sum() > 5]
            gaps.append((max(rates) - min(rates)) if len(rates) >= 2 else 0)
        weights = [n * np.exp(-lambda_fair * gap) for (_, n), gap in zip(updates, gaps)]
        return t.weighted_federated_averaging(updates, weights)

    eps = {c.client_id: 1e6 for c in trainer.clients}
    result = trainer.train_federated(num_rounds=Config.GLOBAL_ROUNDS,
                                      epsilon_allocations=eps, aggregation_fn=_agg)
    result["method"] = "FairFed"
    result["avg_epsilon"] = float("inf")
    return result


def run_fcfl(trainer: HealthcareFLTrainer, seed: int = 42) -> Dict:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix as cm, roc_curve

    trainer.reset(seed)
    eps = {c.client_id: 1e6 for c in trainer.clients}
    result = trainer.train_federated(num_rounds=Config.GLOBAL_ROUNDS, epsilon_allocations=eps)

    # Pool validation probabilities and labels from all clients into one dataset.
    # Fitting per-client Platts and averaging produces inconsistent distributions
    # that collapse entirely below 0.5; a single pooled fit avoids this.
    all_val_probs, all_val_y = [], []
    for client in trainer.clients:
        _, X_val, _, y_val = client.get_train_test_split(seed=trainer.seed)
        val_probs = trainer.global_model.predict(X_val, verbose=0).flatten()
        all_val_probs.append(val_probs)
        all_val_y.append(y_val)
    pool_probs = np.concatenate(all_val_probs)
    pool_y = np.concatenate(all_val_y)

    # Single Platt scaling fit on pooled validation data
    lr = LogisticRegression(class_weight="balanced", max_iter=300)
    lr.fit(pool_probs.reshape(-1, 1), pool_y)

    # Calibrate test predictions
    raw_probs = trainer.global_model.predict(trainer.X_test, verbose=0).flatten()
    cal_probs = lr.predict_proba(raw_probs.reshape(-1, 1))[:, 1]

    # Optimal decision threshold via Youden's J on calibrated validation probs.
    # Hardcoding 0.5 fails when calibrated scores are systematically low/high.
    cal_val_probs = lr.predict_proba(pool_probs.reshape(-1, 1))[:, 1]
    fpr, tpr, thresholds = roc_curve(pool_y, cal_val_probs)
    best_thresh = float(np.clip(thresholds[np.argmax(tpr - fpr)], 0.05, 0.95))

    y_pred = (cal_probs >= best_thresh).astype(int)
    y_true = trainer.y_test.flatten().astype(int)
    auc = roc_auc_score(y_true, cal_probs) if len(np.unique(y_true)) > 1 else 0.5
    try:
        tn, fp, fn, tp = cm(y_true, y_pred).ravel()
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    except ValueError:
        sens, spec = 0.0, 0.0

    result["final_metrics"].update({
        "auc": float(auc), "accuracy": float(accuracy_score(y_true, y_pred)),
        "sensitivity": float(sens), "specificity": float(spec),
        "predictions": y_pred, "prediction_probs": cal_probs.reshape(-1, 1),
    })
    result["method"] = "FCFL"
    result["avg_epsilon"] = float("inf")
    return result


# ---------------------------------------------------------------------------
# MO-Privacy (parallel)
# ---------------------------------------------------------------------------

def run_mo_privacy(trainer: HealthcareFLTrainer, seed: int = 42,
                   pool: Optional[Pool] = None) -> Dict:
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.core.problem import Problem
    from pymoo.optimize import minimize
    from pymoo.operators.crossover.sbx import SBX
    from pymoo.operators.mutation.pm import PM
    from pymoo.operators.sampling.rnd import FloatRandomSampling

    cfg = _make_cfg()

    class _Prob(Problem):
        def __init__(self, trainer, pool, cfg):
            super().__init__(n_var=Config.N_CLIENTS, n_obj=2, n_constr=2,
                             xl=Config.EPSILON_MIN, xu=Config.EPSILON_MAX)
            self.clients = trainer.clients
            self.X_test = trainer.X_test
            self.y_test = trainer.y_test
            self.pool = pool
            self.cfg = cfg
            self.cache: Dict = {}

        def _evaluate(self, X, out, *args, **kwargs):
            n = len(X)
            F, G = np.zeros((n, 2)), np.zeros((n, 2))
            to_eval = [(i, xi) for i, xi in enumerate(X)
                       if tuple(np.round(xi, 2)) not in self.cache]
            for i, xi in enumerate(X):
                key = tuple(np.round(xi, 2))
                if key in self.cache:
                    F[i], G[i] = self.cache[key]
            if to_eval:
                tasks = [(xi, self.clients, self.X_test, self.y_test,
                          Config.EVAL_ROUNDS, seed, self.cfg)
                         for (_, xi) in to_eval]
                raw = (self.pool.map(_v3_mo_privacy_eval_worker, tasks)
                       if self.pool and len(tasks) > 1
                       else [_v3_mo_privacy_eval_worker(t) for t in tasks])
                for (i, xi), (fi, gi) in zip(to_eval, raw):
                    F[i], G[i] = fi, gi
                    self.cache[tuple(np.round(xi, 2))] = (fi.copy(), gi.copy())
            out["F"], out["G"] = F, G

    trainer.reset(seed)
    prob = _Prob(trainer, pool, cfg)
    algo = NSGA2(pop_size=Config.POP_SIZE, n_offsprings=Config.N_OFFSPRINGS,
                 sampling=FloatRandomSampling(), crossover=SBX(prob=0.9, eta=15),
                 mutation=PM(eta=20), eliminate_duplicates=True)
    res = minimize(prob, algo, termination=("n_gen", Config.N_GENERATIONS),
                   seed=seed, verbose=False)

    if res.X is not None and len(res.X) > 0:
        F_norm = res.F / (res.F.max(axis=0) + 1e-8)
        best_idx = int(np.argmin(F_norm.sum(axis=1)))
        best_eps = {j: res.X[best_idx, j] for j in range(Config.N_CLIENTS)}
        trainer.reset(seed)
        result = trainer.train_federated(num_rounds=Config.GLOBAL_ROUNDS,
                                          epsilon_allocations=best_eps)
        result["pareto_X"] = res.X
        result["pareto_F"] = res.F
    else:
        result = run_fixed_eps(trainer, 1.0, seed)
        best_eps = {j: 1.0 for j in range(Config.N_CLIENTS)}
        result["pareto_X"] = None
        result["pareto_F"] = None

    result["method"] = "MO-Privacy"
    result["avg_epsilon"] = float(np.mean(list(best_eps.values())))
    return result


# ---------------------------------------------------------------------------
# MO-Efficiency (parallel)
# ---------------------------------------------------------------------------

def run_mo_efficiency(trainer: HealthcareFLTrainer, seed: int = 42,
                      pool: Optional[Pool] = None) -> Dict:
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.core.problem import Problem
    from pymoo.optimize import minimize
    from pymoo.operators.crossover.sbx import SBX
    from pymoo.operators.mutation.pm import PM
    from pymoo.operators.sampling.rnd import FloatRandomSampling

    cfg = _make_cfg()

    class _Prob(Problem):
        def __init__(self, trainer, pool, cfg):
            super().__init__(n_var=Config.N_CLIENTS, n_obj=2, n_constr=1,
                             xl=Config.EPSILON_MIN, xu=Config.EPSILON_MAX)
            self.clients = trainer.clients
            self.X_test = trainer.X_test
            self.y_test = trainer.y_test
            self.pool = pool
            self.cfg = cfg
            self.cache: Dict = {}

        def _evaluate(self, X, out, *args, **kwargs):
            n = len(X)
            F, G = np.zeros((n, 2)), np.zeros((n, 1))
            to_eval = [(i, xi) for i, xi in enumerate(X)
                       if tuple(np.round(xi, 2)) not in self.cache]
            for i, xi in enumerate(X):
                key = tuple(np.round(xi, 2))
                if key in self.cache:
                    F[i], G[i] = self.cache[key]
            if to_eval:
                tasks = [(xi, self.clients, self.X_test, self.y_test,
                          Config.EVAL_ROUNDS, seed, self.cfg)
                         for (_, xi) in to_eval]
                raw = (self.pool.map(_v3_mo_efficiency_eval_worker, tasks)
                       if self.pool and len(tasks) > 1
                       else [_v3_mo_efficiency_eval_worker(t) for t in tasks])
                for (i, xi), (fi, gi) in zip(to_eval, raw):
                    F[i], G[i] = fi, gi
                    self.cache[tuple(np.round(xi, 2))] = (fi.copy(), gi.copy())
            out["F"], out["G"] = F, G

    trainer.reset(seed)
    prob = _Prob(trainer, pool, cfg)
    algo = NSGA2(pop_size=Config.POP_SIZE, n_offsprings=Config.N_OFFSPRINGS,
                 sampling=FloatRandomSampling(), crossover=SBX(prob=0.9, eta=15),
                 mutation=PM(eta=20), eliminate_duplicates=True)
    res = minimize(prob, algo, termination=("n_gen", Config.N_GENERATIONS),
                   seed=seed, verbose=False)

    if res.X is not None and len(res.X) > 0:
        F_norm = res.F / (res.F.max(axis=0) + 1e-8)
        best_idx = int(np.argmin(F_norm.sum(axis=1)))
        best_eps = {j: res.X[best_idx, j] for j in range(Config.N_CLIENTS)}
        trainer.reset(seed)
        result = trainer.train_federated(num_rounds=Config.GLOBAL_ROUNDS,
                                          epsilon_allocations=best_eps)
        result["pareto_X"] = res.X
        result["pareto_F"] = res.F
    else:
        result = run_fixed_eps(trainer, 1.0, seed)
        best_eps = {j: 1.0 for j in range(Config.N_CLIENTS)}
        result["pareto_X"] = None
        result["pareto_F"] = None

    result["method"] = "MO-Efficiency"
    result["avg_epsilon"] = float(np.mean(list(best_eps.values())))
    return result


# ---------------------------------------------------------------------------
# MOOP-HealthFL 4-objective (parallel)
# ---------------------------------------------------------------------------

def run_moop_healthfl(trainer: HealthcareFLTrainer,
                      fairness_evaluator: FairnessEvaluator,
                      seed: int = 42,
                      pool: Optional[Pool] = None,
                      logger: logging.Logger = None) -> Dict:
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.core.problem import Problem
    from pymoo.optimize import minimize
    from pymoo.operators.crossover.sbx import SBX
    from pymoo.operators.mutation.pm import PM
    from pymoo.operators.sampling.rnd import FloatRandomSampling

    if logger is None:
        logger = log

    cfg = _make_cfg()
    eth_test = fairness_evaluator.demographics["ethnicity"].values
    gen_test = fairness_evaluator.demographics["gender"].values

    class _MOOPProblem(Problem):
        def __init__(self, trainer, pool, cfg):
            super().__init__(n_var=Config.N_CLIENTS, n_obj=4, n_constr=3,
                             xl=Config.EPSILON_MIN, xu=Config.EPSILON_MAX)
            self.clients = trainer.clients
            self.X_test = trainer.X_test
            self.y_test = trainer.y_test
            self.pool = pool
            self.cfg = cfg
            self.cache: Dict = {}
            self.eval_count = 0

        def _evaluate(self, X, out, *args, **kwargs):
            n = len(X)
            F, G = np.zeros((n, 4)), np.zeros((n, 3))
            to_eval = []
            for i, xi in enumerate(X):
                key = tuple(np.round(xi, 2))
                if key in self.cache:
                    F[i], G[i] = self.cache[key]
                else:
                    to_eval.append((i, xi))
                    self.eval_count += 1

            if to_eval:
                tasks = [
                    (xi, self.clients, self.X_test, self.y_test,
                     eth_test, gen_test,
                     Config.EVAL_ROUNDS, seed, self.cfg)
                    for (_, xi) in to_eval
                ]
                raw = (self.pool.map(_v3_moop_eval_worker, tasks)
                       if self.pool and len(tasks) > 1
                       else [_v3_moop_eval_worker(t) for t in tasks])
                for (i, xi), (fi, gi) in zip(to_eval, raw):
                    F[i], G[i] = fi, gi
                    self.cache[tuple(np.round(xi, 2))] = (fi.copy(), gi.copy())

            out["F"], out["G"] = F, G

    trainer.reset(seed)
    n_workers = pool._processes if pool is not None else 1
    logger.info(f"  [MOOP] NSGA-II pop={Config.POP_SIZE}, gen={Config.N_GENERATIONS}, "
                f"eval_rounds={Config.EVAL_ROUNDS}, workers={n_workers}")

    prob = _MOOPProblem(trainer, pool, cfg)
    algo = NSGA2(pop_size=Config.POP_SIZE, n_offsprings=Config.N_OFFSPRINGS,
                 sampling=FloatRandomSampling(), crossover=SBX(prob=0.9, eta=15),
                 mutation=PM(eta=20), eliminate_duplicates=True)
    res = minimize(prob, algo, termination=("n_gen", Config.N_GENERATIONS),
                   seed=seed, verbose=False)

    if res.X is not None and len(res.X) > 0:
        F_norm = res.F / (res.F.max(axis=0) + 1e-8)
        best_idx = int(np.argmin(F_norm.sum(axis=1)))
        best_eps = {j: res.X[best_idx, j] for j in range(Config.N_CLIENTS)}

        logger.info("  [MOOP] Running full training with best allocation...")
        trainer.reset(seed)
        result = trainer.train_federated(num_rounds=Config.GLOBAL_ROUNDS,
                                          epsilon_allocations=best_eps)
        result["pareto_X"] = res.X
        result["pareto_F"] = res.F
        result["best_allocation"] = best_eps
        result["n_pareto"] = len(res.X)
        result["n_evaluations"] = prob.eval_count
    else:
        logger.warning("  [MOOP] NSGA-II returned empty Pareto — fallback to Fixed-eps(1.0)")
        result = run_fixed_eps(trainer, 1.0, seed)
        best_eps = {j: 1.0 for j in range(Config.N_CLIENTS)}
        result["pareto_X"] = None
        result["pareto_F"] = None
        result["best_allocation"] = best_eps

    result["method"] = "MOOP-HealthFL"
    result["avg_epsilon"] = float(np.mean(list(best_eps.values())))
    return result


# ---------------------------------------------------------------------------
# Checkpoint manager
# ---------------------------------------------------------------------------

class CheckpointManager:
    def __init__(self, checkpoint_dir: str):
        self.dir = checkpoint_dir
        os.makedirs(self.dir, exist_ok=True)

    def save(self, results, name: str = "experiment_results"):
        path = os.path.join(self.dir, f"{name}.pkl")
        clean = self._clean(results)
        with open(path, "wb") as f:
            pickle.dump(clean, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, name: str = "experiment_results"):
        path = os.path.join(self.dir, f"{name}.pkl")
        if os.path.exists(path):
            with open(path, "rb") as f:
                return pickle.load(f)
        return None

    def _clean(self, obj):
        if isinstance(obj, dict):
            return {k: self._clean(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._clean(x) for x in obj]
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        return str(obj)


# ---------------------------------------------------------------------------
# Results aggregation
# ---------------------------------------------------------------------------

def aggregate_results(all_results: Dict) -> Dict:
    seeds = [s for s in all_results if not s.startswith("_")]
    if not seeds:
        return {}
    methods = [k for k in all_results[seeds[0]] if not k.startswith("_")]
    aggregated = {}
    for method in methods:
        metrics: Dict = {}
        for metric in ["auc", "accuracy", "sensitivity", "specificity"]:
            vals = [all_results[s][method]["final_metrics"].get(metric, np.nan)
                    for s in seeds if method in all_results[s]]
            metrics[metric] = (float(np.nanmean(vals)), float(np.nanstd(vals)))
        for fk in ["fairness_client", "fairness_group", "fairness_individual",
                   "fairness_system", "fairness_outcome", "fairness_unified"]:
            vals = [all_results[s][method].get("fairness", {}).get(fk, np.nan)
                    for s in seeds if method in all_results[s]]
            metrics[fk] = (float(np.nanmean(vals)), float(np.nanstd(vals)))
        metrics["avg_epsilon"] = (
            all_results[seeds[0]][method].get("avg_epsilon", float("inf"))
            if method in all_results[seeds[0]] else float("inf")
        )
        metrics["method_name"] = (
            all_results[seeds[0]][method].get("method", method)
            if method in all_results[seeds[0]] else method
        )
        aggregated[method] = metrics
    return aggregated


# ---------------------------------------------------------------------------
# Figure generation
# ---------------------------------------------------------------------------

def generate_figures(agg: Dict, all_results: Dict, figures_dir: str,
                     results_dir: str, logger: logging.Logger):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matplotlib.rcParams.update({
        "font.family": "serif", "font.size": 10,
        "axes.labelsize": 11, "axes.titlesize": 12,
        "legend.fontsize": 9, "figure.dpi": 150,
        "savefig.dpi": 300, "savefig.bbox": "tight",
    })

    COLORS = {
        "fedavg": "#1f77b4", "fixed_eps": "#ff7f0e",
        "dpsgd": "#2ca02c", "fairfed": "#d62728", "fcfl": "#9467bd",
        "mo_privacy": "#8c564b", "mo_efficiency": "#e377c2",
        "moop_healthfl": "#17becf",
    }

    def _m(method, key):
        v = agg.get(method, {}).get(key, (0.0, 0.0))
        return v if isinstance(v, tuple) else (float(v), 0.0)

    def _save(name):
        for d in [figures_dir, results_dir]:
            plt.savefig(os.path.join(d, name), bbox_inches="tight")
        plt.close()
        logger.info(f"  Saved {name}")

    # Figure 1: Privacy-Utility tradeoff
    fig, ax = plt.subplots(figsize=(7, 5))
    eps_vals = [0.5, 1.0, 2.0, 5.0]
    keys = ["fixed_eps_0.5", "fixed_eps_1.0", "fixed_eps_2.0", "fixed_eps_5.0"]
    aucs_m = [_m(k, "auc")[0] for k in keys]
    aucs_s = [_m(k, "auc")[1] for k in keys]
    ax.errorbar(eps_vals, aucs_m, yerr=aucs_s, marker="s", color=COLORS["fixed_eps"],
                label="Fixed-$\\varepsilon$ DP", capsize=3, linewidth=1.5)
    for mk, label, marker in [
        ("fedavg", "FedAvg (no DP)", "o"), ("dpsgd", "DP-SGD", "^"),
        ("fairfed", "FairFed", "D"), ("fcfl", "FCFL", "v"),
        ("mo_privacy", "MO-Privacy", "p"), ("mo_efficiency", "MO-Efficiency", "h"),
        ("moop_healthfl", "MOOP-HealthFL", "*"),
    ]:
        if mk not in agg:
            continue
        eps = agg[mk].get("avg_epsilon", float("inf"))
        eps_plot = min(eps, 6.0) if eps != float("inf") else 6.0
        auc_m, auc_s = _m(mk, "auc")
        color = COLORS.get(mk.split("_")[0] if "fixed" not in mk else "moop_healthfl",
                           COLORS.get(mk, "#333"))
        ax.errorbar(eps_plot, auc_m, yerr=auc_s, marker=marker, color=color,
                    label=label, capsize=3, markersize=8)
    ax.set_xlabel("Average $\\varepsilon$")
    ax.set_ylabel("AUROC")
    ax.set_title("Privacy-Utility Tradeoff")
    ax.legend(loc="lower right", framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 6.5)
    _save("fig_privacy_utility.pdf")

    # Figure 2: Convergence
    first_seed = str(Config.SEEDS[0])
    fig, ax = plt.subplots(figsize=(7, 5))
    for mk, label, color in [
        ("fedavg", "FedAvg", COLORS["fedavg"]),
        ("fixed_eps_1.0", "Fixed-$\\varepsilon$=1.0", COLORS["fixed_eps"]),
        ("dpsgd", "DP-SGD", COLORS["dpsgd"]),
        ("fairfed", "FairFed", COLORS["fairfed"]),
        ("moop_healthfl", "MOOP-HealthFL", COLORS["moop_healthfl"]),
    ]:
        if mk not in all_results.get(first_seed, {}):
            continue
        history = all_results[first_seed][mk].get("training_history", [])
        if history:
            ax.plot([h["round"] for h in history],
                    [h["auc"] for h in history],
                    label=label, color=color, linewidth=1.5)
    ax.set_xlabel("Communication Round")
    ax.set_ylabel("AUROC")
    ax.set_title("Training Convergence")
    ax.legend(loc="lower right", framealpha=0.9)
    ax.grid(True, alpha=0.3)
    _save("fig_convergence.pdf")

    # Figure 3: Pareto front — single clean panel (Privacy x Utility, colour = Fairness).
    # The efficiency objective is near-constant at this model scale, so a Fairness x
    # Efficiency panel would appear degenerate; we visualise the three informative
    # objectives (privacy, utility, fairness) in one scatter instead.
    moop_res = all_results.get(first_seed, {}).get("moop_healthfl", {})
    pf = moop_res.get("pareto_F")
    if pf is not None:
        pareto_F = np.array(pf)
        fig, ax = plt.subplots(figsize=(7, 5.5))
        sc = ax.scatter(pareto_F[:, 0], pareto_F[:, 1],
                        c=pareto_F[:, 2], cmap="viridis_r", s=90, alpha=0.85,
                        edgecolors="black", linewidths=0.4)
        cbar = plt.colorbar(sc, ax=ax, label="Fairness Loss")
        cbar.ax.tick_params(labelsize=9)
        ax.set_xlabel(r"Privacy Loss  ($\bar{\varepsilon}/\varepsilon_{\max}$)")
        ax.set_ylabel(r"Utility Loss  ($1 - \mathrm{AUROC}$)")
        ax.set_title("MOOP-HealthFL Pareto Front\n(Privacy--Utility--Fairness Trade-off)")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        _save("fig_pareto_front.pdf")

    # Figure 4: Per-client epsilon allocation
    fig, ax = plt.subplots(figsize=(8, 5))
    hospitals = [f"H{i}" for i in range(Config.N_CLIENTS)]
    x = np.arange(len(hospitals))
    width = 0.25
    ax.bar(x - width, [1.0] * Config.N_CLIENTS, width,
           label="Fixed-$\\varepsilon$=1.0", color=COLORS["fixed_eps"], alpha=0.8)
    mo_priv = all_results.get(first_seed, {}).get("mo_privacy", {})
    if mo_priv.get("pareto_X") is not None:
        pareto_X = np.array(mo_priv["pareto_X"])
        ax.bar(x, pareto_X[0, :Config.N_CLIENTS], width,
               label="MO-Privacy", color=COLORS["mo_privacy"], alpha=0.8)
    if moop_res.get("best_allocation"):
        moop_eps = [moop_res["best_allocation"].get(i, 1.0) for i in range(Config.N_CLIENTS)]
        ax.bar(x + width, moop_eps, width, label="MOOP-HealthFL",
               color=COLORS["moop_healthfl"], alpha=0.8)
    ax.set_xlabel("Hospital")
    ax.set_ylabel("Privacy Budget ($\\varepsilon$)")
    ax.set_title("Per-Client Privacy Allocation")
    ax.set_xticks(x)
    ax.set_xticklabels(hospitals)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    _save("fig_client_allocation.pdf")

    # Figure 5: Fairness radar
    categories = ["Client", "Group", "Individual", "System", "Outcome"]
    fair_keys = ["fairness_client", "fairness_group", "fairness_individual",
                 "fairness_system", "fairness_outcome"]
    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    plt.xticks(angles[:-1], categories, size=8)
    for mk, label, color in [
        ("fedavg", "FedAvg", COLORS["fedavg"]),
        ("fairfed", "FairFed", COLORS["fairfed"]),
        ("fcfl", "FCFL", COLORS["fcfl"]),
        ("moop_healthfl", "MOOP-HealthFL", COLORS["moop_healthfl"]),
    ]:
        if mk not in agg:
            continue
        vals = [_m(mk, fk)[0] for fk in fair_keys] + [_m(mk, fair_keys[0])[0]]
        ax.plot(angles, vals, linewidth=1.5, label=label, color=color)
        ax.fill(angles, vals, alpha=0.1, color=color)
    ax.set_title("Five-Level Fairness", size=11, y=1.08)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    _save("fig_fairness_radar.pdf")

    # Figure 6: Training dashboard
    history = moop_res.get("training_history", [])
    if history:
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        rounds = [h["round"] for h in history]
        axes[0, 0].plot(rounds, [h["auc"] for h in history], "b-", lw=1.5)
        axes[0, 0].set_ylabel("AUROC")
        axes[0, 0].set_title("Global AUROC")
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 1].plot(rounds, [h["accuracy"] for h in history], "g-", lw=1.5)
        axes[0, 1].set_ylabel("Accuracy")
        axes[0, 1].set_title("Global Accuracy")
        axes[0, 1].grid(True, alpha=0.3)
        axes[1, 0].plot(rounds, [h["loss"] for h in history], "r-", lw=1.5)
        axes[1, 0].set_xlabel("Round")
        axes[1, 0].set_ylabel("Loss")
        axes[1, 0].set_title("Global Loss")
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 1].plot(rounds, [h["sensitivity"] for h in history], "m-", lw=1.5)
        axes[1, 1].set_xlabel("Round")
        axes[1, 1].set_ylabel("Sensitivity")
        axes[1, 1].set_title("Sensitivity (Mortality Detection)")
        axes[1, 1].grid(True, alpha=0.3)
        plt.suptitle("MOOP-HealthFL Training Dashboard", fontsize=13)
        plt.tight_layout()
        _save("fig_training_dashboard.pdf")

    logger.info(f"All figures saved to {figures_dir}")


# ---------------------------------------------------------------------------
# LaTeX tables
# ---------------------------------------------------------------------------

def print_latex_tables(agg: Dict):
    def fmt(mean, std):
        return f"{mean:.3f}$\\pm${std:.3f}"

    order = ["fedavg", "fixed_eps_0.5", "fixed_eps_1.0", "fixed_eps_2.0",
             "fixed_eps_5.0", "dpsgd", "fairfed", "fcfl",
             "mo_privacy", "mo_efficiency", "moop_healthfl"]
    order = [m for m in order if m in agg]

    print("\n% Table IV: Main Results")
    print(f"{'Method':30s} {'AUC':15s} {'Sens':15s} {'Fairness(unif)':20s} Avg-eps")
    print("-" * 90)
    for mk in order:
        m = agg[mk]
        eps = m["avg_epsilon"]
        eps_str = f"{eps:.2f}" if eps != float("inf") else "inf"
        print(f"{m['method_name']:30s} {fmt(*m['auc']):15s} {fmt(*m['sensitivity']):15s} "
              f"{fmt(*m['fairness_unified']):20s} {eps_str}")


# ---------------------------------------------------------------------------
# Main experiment runner
# ---------------------------------------------------------------------------

def run_experiments(args, logger: logging.Logger):
    output_dir = args.output_dir
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    figures_dir = os.path.join(output_dir, "figures")
    results_dir = os.path.join(output_dir, "results")
    for d in [output_dir, checkpoint_dir, figures_dir, results_dir]:
        os.makedirs(d, exist_ok=True)

    if args.fast:
        Config.apply_fast_mode()
        logger.info("FAST MODE enabled.")

    Config.N_JOBS = args.n_jobs
    if args.seed is not None:
        Config.SEEDS = [args.seed]

    logger.info("=" * 70)
    logger.info("MOOP-HealthFL v3 Parallel")
    logger.info(f"Seeds:          {Config.SEEDS}")
    logger.info(f"Global rounds:  {Config.GLOBAL_ROUNDS}")
    logger.info(f"NSGA-II:        pop={Config.POP_SIZE}, gen={Config.N_GENERATIONS}, "
                f"eval_rounds={Config.EVAL_ROUNDS}")
    logger.info(f"Workers:        {Config.N_JOBS}")
    logger.info("=" * 70)

    # Data
    if args.mimic_path and os.path.exists(args.mimic_path):
        cache_file = os.path.join(output_dir, "mimic_iv_features.parquet")
        data = load_mimic_data(args.mimic_path, cache_file, logger)
    else:
        logger.warning("MIMIC-IV path not found — using synthetic data.")
        data = generate_synthetic_data()

    logger.info(f"Dataset: {len(data):,} stays, mortality={data['mortality'].mean():.1%}")

    # Impute NaN in ALL feature columns before scaling.
    # MIMIC-IV merges can leave NaN in demographic columns (age, gender, etc.)
    # even after clinical-column imputation in load_mimic_data.
    na_count = data[Config.FEATURE_COLS].isnull().sum().sum()
    if na_count > 0:
        for col in Config.FEATURE_COLS:
            median_val = data[col].median()
            data[col] = data[col].fillna(median_val)
        logger.warning(f"Imputed {na_count} NaN values across feature columns.")

    # Hospital split
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split

    scaler = StandardScaler()
    all_X_raw = data[Config.FEATURE_COLS].values.astype(np.float32)
    scaler.fit(all_X_raw)

    hospital_ids = sorted(data["hospital_id"].unique())[:Config.N_CLIENTS]
    client_dfs = [data[data["hospital_id"] == hid] for hid in hospital_ids]

    clients = []
    for i, cdf in enumerate(client_dfs):
        c = FederatedClient(i, cdf.reset_index(drop=True), scaler)
        clients.append(c)
        logger.info(f"  Client {i}: {c.n_samples:,} samples, "
                    f"mortality={c.mortality_rate:.1%}")

    all_X = np.vstack([c.X for c in clients])
    all_y = np.concatenate([c.y for c in clients])
    all_eth = np.concatenate([c.ethnicity for c in clients])
    all_gen = np.concatenate([c.gender for c in clients])

    _, X_test, _, y_test, _, eth_te, _, gen_te = train_test_split(
        all_X, all_y, all_eth, all_gen,
        test_size=Config.TEST_SIZE, random_state=42, stratify=all_y,
    )
    test_demographics = pd.DataFrame({"ethnicity": eth_te, "gender": gen_te})
    fairness_eval = FairnessEvaluator(test_demographics)
    logger.info(f"Global test set: {len(y_test):,} samples")

    ckpt = CheckpointManager(checkpoint_dir)
    all_results = ckpt.load() or {}

    pool_ctx = Pool(Config.N_JOBS) if Config.N_JOBS > 1 else None
    logger.info(f"Pool: {'%d workers' % Config.N_JOBS if pool_ctx else 'sequential'}")

    METHODS = [
        ("fedavg",        "FedAvg",           lambda t, s: run_fedavg(t, s)),
        ("fixed_eps_0.5", "Fixed-eps(0.5)",   lambda t, s: run_fixed_eps(t, 0.5, s)),
        ("fixed_eps_1.0", "Fixed-eps(1.0)",   lambda t, s: run_fixed_eps(t, 1.0, s)),
        ("fixed_eps_2.0", "Fixed-eps(2.0)",   lambda t, s: run_fixed_eps(t, 2.0, s)),
        ("fixed_eps_5.0", "Fixed-eps(5.0)",   lambda t, s: run_fixed_eps(t, 5.0, s)),
        ("dpsgd",         "DP-SGD",           lambda t, s: run_dpsgd_standalone(t, 1.0, s)),
        ("fairfed",       "FairFed",          lambda t, s: run_fairfed(t, s)),
        ("fcfl",          "FCFL",             lambda t, s: run_fcfl(t, s)),
        ("mo_privacy",    "MO-Privacy",       lambda t, s: run_mo_privacy(t, s, pool=pool_ctx)),
        ("mo_efficiency", "MO-Efficiency",    lambda t, s: run_mo_efficiency(t, s, pool=pool_ctx)),
        ("moop_healthfl", "MOOP-HealthFL",    lambda t, s: run_moop_healthfl(
            t, fairness_eval, s, pool=pool_ctx, logger=logger)),
    ]

    try:
        for seed in Config.SEEDS:
            sk = str(seed)
            if sk not in all_results:
                all_results[sk] = {}

            logger.info(f"\n{'='*70}\nSEED {seed}\n{'='*70}")
            trainer = HealthcareFLTrainer(clients, X_test, y_test, seed=seed)
            client_sizes = {c.client_id: c.n_samples for c in clients}

            for method_key, method_name, method_fn in METHODS:
                if method_key in all_results[sk]:
                    logger.info(f"  [{method_name}] already done — skipping.")
                    continue

                logger.info(f"\n  Running [{method_name}]...")
                t0 = time.time()
                try:
                    result = method_fn(trainer, seed)
                    elapsed = time.time() - t0

                    # Compute 5-level fairness
                    m = result["final_metrics"]
                    y_true = y_test.flatten().astype(int)
                    eps_alloc = result.get("best_allocation") or (
                        {c.client_id: result.get("avg_epsilon", 1.0) for c in clients}
                        if result.get("avg_epsilon", float("inf")) != float("inf")
                        else None
                    )
                    result["fairness"] = fairness_eval.evaluate_all(
                        y_true, m["predictions"], m["prediction_probs"],
                        X_test, result.get("client_aucs", [0.5] * Config.N_CLIENTS),
                        eps_alloc, client_sizes,
                    )

                    all_results[sk][method_key] = result
                    ckpt.save(all_results)

                    auc = m.get("auc", 0.0)
                    sens = m.get("sensitivity", 0.0)
                    fair = result["fairness"]["fairness_unified"]
                    logger.info(f"  [{method_name}] {elapsed:.0f}s — "
                                f"AUC={auc:.4f}, Sens={sens:.3f}, Fair={fair:.3f}")
                except Exception as exc:
                    logger.error(f"  [{method_name}] FAILED: {exc}", exc_info=True)
                    all_results[sk][method_key] = {"method": method_name, "error": str(exc),
                                                    "final_metrics": {}, "fairness": {}}
                    ckpt.save(all_results)

    finally:
        if pool_ctx is not None:
            pool_ctx.close()
            pool_ctx.join()

    # Aggregate
    logger.info("\n" + "=" * 70)
    logger.info("AGGREGATING RESULTS")
    agg = aggregate_results(all_results)

    json_path = os.path.join(results_dir, "aggregated_results.json")
    agg_json: Dict = {}
    for method, metrics in agg.items():
        agg_json[method] = {}
        for k, v in metrics.items():
            if isinstance(v, tuple):
                agg_json[method][k] = {"mean": v[0], "std": v[1]}
            elif v == float("inf"):
                agg_json[method][k] = "inf"
            else:
                agg_json[method][k] = v
    with open(json_path, "w") as f:
        json.dump(agg_json, f, indent=2, default=str)
    logger.info(f"Results saved to {json_path}")

    # Print table
    print_latex_tables(agg)

    # Figures
    logger.info("\nGenerating figures...")
    try:
        generate_figures(agg, all_results, figures_dir, results_dir, logger)
    except Exception as exc:
        logger.warning(f"Figure generation failed: {exc}")

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)
    if "moop_healthfl" in agg:
        m = agg["moop_healthfl"]
        logger.info(f"MOOP-HealthFL: AUC={m['auc'][0]:.3f}±{m['auc'][1]:.3f}, "
                    f"Sens={m['sensitivity'][0]:.3f}±{m['sensitivity'][1]:.3f}, "
                    f"Fair={m['fairness_unified'][0]:.3f}±{m['fairness_unified'][1]:.3f}, "
                    f"ε={m['avg_epsilon']:.2f}")
    n_figs = len([f for f in os.listdir(figures_dir) if f.endswith(".pdf")])
    logger.info(f"Figures: {n_figs} PDFs saved to {figures_dir}")
    logger.info(f"Results: {results_dir}")
    logger.info("\nAll done.")
    return agg


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="MOOP-HealthFL: multi-objective privacy-preserving federated learning on MIMIC-IV")
    parser.add_argument("--fast", action="store_true",
                        help="Fast mode (pop=20, gen=30, eval_rounds=10, seed=42 only)")
    parser.add_argument("--n-jobs", type=int,
                        default=max(1, (os.cpu_count() or 2) - 1),
                        help="Parallel workers for NSGA-II (default: cpu_count - 1)")
    parser.add_argument("--mimic-path", type=str, default=None,
                        help="Path to MIMIC-IV directory (optional; synthetic fallback if absent)")
    parser.add_argument("--output-dir", type=str, default="output",
                        help="Output directory")
    parser.add_argument("--seed", type=int, default=None,
                        help="Run a single seed (default: all seeds)")
    args = parser.parse_args()

    log_dir = os.path.join(args.output_dir, "logs")
    logger = setup_logging(log_dir)
    run_experiments(args, logger)


if __name__ == "__main__":
    main()
