#!/usr/bin/env python3
"""
MOOP-HealthFL -- certified differential privacy in clinical federated learning
with multi-objective per-client privacy-budget selection.

Protocol
  * Patient-level 70/10/20 split over subject_id; each training patient is
    assigned to exactly one client (modal care unit).
  * 19 one-hot features; continuous features preprocessed with public
    clinical reference values only (no statistic fitted on patient data).
  * Certified DP-SGD (dp_core_v5) at PATIENT-level adjacency: Poisson
    sampling of patients at q = B / n_patients, all stays of a sampled
    patient form one contribution whose summed gradient is l2-clipped,
    Gaussian noise on the per-step sum, noise multiplier calibrated by an
    RDP accountant so that the full GLOBAL_ROUNDS protocol meets each
    client's total (epsilon, delta) budget. Decision variables are total
    budgets eps_k in [1, 10].
  * Design/deployment separation: budget searches train only on a disjoint
    design cohort of patients; every reported model is trained once on the
    deployment cohort, whose guarantee is therefore the single-run
    certificate. The composed cost of the searches is reported separately.
  * One prespecified decision-threshold rule for every method: maximum
    specificity subject to validation sensitivity >= 0.70 (Youden fallback).

Search
  Objectives (3): privacy loss = mean(eps_k) / eps_max; utility loss =
  1 - AUROC; fairness loss = unified five-level fairness deficit.
  Constraints (3): sensitivity >= 0.70; AUROC >= 0.60; max per-attribute
  FNR gap <= 0.20. Fitness is evaluated on the validation split.

Methods (10)
  FedAvg, FairFed, FCFL, Fixed-eps(1/2/5/10), MO-Privacy (2-objective
  NSGA-II), LHS-Search (random-search control), MOOP-HealthFL (3-objective
  NSGA-II).

Reproducibility
  A run is fully determined by its seed: weight initialisation, dropout,
  Poisson sampling and Gaussian noise all derive from the NumPy state set
  by HealthcareFLTrainer.reset(seed). Every trained model's validation and
  test probabilities are written to <output-dir>/predictions/.

Usage
  python moop_healthfl.py --mimic-path PATH [--output-dir DIR] [--n-jobs N]
                          [--seed SEED] [--fast]
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

# One BLAS thread per process: the model is tiny (3,393 parameters), so
# intra-op threading only adds overhead and oversubscribes the worker pool.
# Must be set before NumPy is imported; respects values already exported.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Also cap BLAS at runtime, for callers that imported NumPy before this module.
from threadpoolctl import threadpool_limits
threadpool_limits(limits=1)

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
    VAL_SIZE = 0.10  # v4: patient-level validation fraction (NSGA-II fitness)
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
    LEARNING_RATE = 0.001         # non-private training
    # Hidden-layer widths. (64, 32) is the model of the paper (3,393
    # parameters); wider settings are used only by the capacity experiment
    # (experiments_extra/run_capacity.py).
    HIDDEN = (64, 32)
    DP_LEARNING_RATE = 0.003      # certified (clipped, noised) training

    # Privacy boundary (see docs/PRIVACY_LEDGER.md). The second value of each
    # switch reproduces the original submission and exists only for ablation.
    PRIVACY_UNIT = "patient"      # "patient": clip per patient | "record": per ICU stay
    PREPROCESSING = "reference"   # "reference": public constants | "cohort": fitted statistics
    CLASS_WEIGHT = "prior"        # "prior": from MORTALITY_PRIOR | "pooled": from training labels
    MORTALITY_PRIOR = 0.10        # public prior for ICU in-hospital mortality
    DESIGN_FRACTION = 0.50        # share of training patients reserved for budget search (0 = none)

    # Privacy (TOTAL-training budgets, certified via RDP accountant)
    EPSILON_MIN = 1.0
    EPSILON_MAX = 10.0
    DELTA = 1e-5
    # Clipping norm and DP learning rate were selected on the DESIGN cohort
    # (experiments_extra/run_dp_mechanism.py tuning; best of 45 configurations
    # on two patient splits). The submission used C = 1.0 and lr = 0.001.
    CLIPPING_NORM = 30.0
    # Ablation only (experiments_extra/run_dp_mechanism.py): multiplies the
    # calibrated noise multiplier. Any value other than 1.0 VOIDS the
    # certificate (0.0 = clipping without noise).
    NOISE_SCALE = 1.0
    SENS_TARGET = 0.70   # validation threshold-rule target sensitivity

    # NSGA-II (full mode)
    POP_SIZE = 50
    N_GENERATIONS = 50
    N_OFFSPRINGS = 25
    EVAL_ROUNDS = 15

    # Constraints (evaluated on cross-fitted validation decisions, i.e. with a
    # threshold chosen on other validation patients). The threshold rule
    # targets sensitivity 0.70; out of sample that is met only on average, so
    # the constraint allows a 5-point shortfall instead of rejecting about
    # half of all candidates by chance.
    SENSITIVITY_MIN = 0.65
    AUC_MIN = 0.60
    FAIRNESS_GAP_MAX = 0.20

    # Fairness objective of the search: "five_level" (composite score) or
    # "fnr_gap" (largest subgroup false-negative-rate gap, a direct clinical
    # quantity; used for the objective sensitivity analysis).
    FAIRNESS_OBJECTIVE = "five_level"

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
# Model (NumPy backend: exact control of clipping, sampling, and noise)
# ---------------------------------------------------------------------------

import dp_core_v5 as dpc

NONPRIVATE_EPS = 1e5   # allocations >= this run the non-private path


def _apply_env_overrides():
    """MOOP_CONFIG='{"HIDDEN": [256, 128], "CLIPPING_NORM": 10}' overrides
    Config attributes in every script and harness that imports this module
    (used by the capacity experiment). Overridden values are part of the
    saved run configuration."""
    raw = os.environ.get("MOOP_CONFIG")
    if not raw:
        return
    for k, v in json.loads(raw).items():
        if not hasattr(Config, k):
            raise KeyError(f"MOOP_CONFIG: unknown Config attribute {k!r}")
        setattr(Config, k, tuple(v) if isinstance(v, list) else v)


_apply_env_overrides()


class NumpyMLPWrapper:
    """Pure-NumPy MLP (19-h1-h2-1; Config.HIDDEN = (64, 32) gives the 3,393
    parameters of the paper).

    He-initialised weights, ReLU activations, dropout 0.3/0.2 in training
    mode, Adam optimiser. Gradients of the class-weighted BCE loss come from
    dp_core_v5 (clipped_noisy_grads on the private path, batch_grads
    otherwise); there is no weight decay. get_weights / set_weights are used by federated averaging.
    """

    def __init__(self, input_dim: int = dpc.N_FEATURES, lr: float = 0.001):
        self.lr = lr
        self._init_weights(input_dim)

    def _init_weights(self, input_dim: int):
        # Seeded from the global NumPy state (set by trainer.reset(seed)) so
        # that a run is fully determined by its seed.
        rng = np.random.default_rng(np.random.randint(0, 2**31 - 1))
        h1, h2 = Config.HIDDEN
        self.W1 = (rng.standard_normal((input_dim, h1)) * np.sqrt(2 / input_dim)).astype(np.float32)
        self.b1 = np.zeros(h1, dtype=np.float32)
        self.W2 = (rng.standard_normal((h1, h2)) * np.sqrt(2 / h1)).astype(np.float32)
        self.b2 = np.zeros(h2, dtype=np.float32)
        self.W3 = (rng.standard_normal((h2, 1)) * np.sqrt(2 / h2)).astype(np.float32)
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
        """Returns (probabilities, cache). cache = (X, a1, a2, z3, g1, g2):
        hidden activations as fed forward (after dropout) and their
        derivatives w.r.t. the pre-activations (ReLU gate x dropout scale),
        which is what backpropagation needs (dp_core_v5.backprop_deltas)."""
        z1 = X @ self.W1 + self.b1
        g1 = (z1 > 0).astype(X.dtype)
        if training:
            g1 = g1 * ((np.random.rand(*z1.shape) > 0.3).astype(np.float32) / 0.7)
        a1 = z1 * g1
        z2 = a1 @ self.W2 + self.b2
        g2 = (z2 > 0).astype(X.dtype)
        if training:
            g2 = g2 * ((np.random.rand(*z2.shape) > 0.2).astype(np.float32) / 0.8)
        a2 = z2 * g2
        z3 = a2 @ self.W3 + self.b3
        out = 1.0 / (1.0 + np.exp(-np.clip(z3, -20, 20)))
        return out, (X, a1, a2, z3, g1, g2)

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
    # Public interface
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

    def count_params(self):
        return sum(p.size for p in self._params())


def create_model(input_dim: int = dpc.N_FEATURES, learning_rate: Optional[float] = None):
    # Read at call time so that harness overrides of Config.LEARNING_RATE apply.
    if learning_rate is None:
        learning_rate = Config.LEARNING_RATE
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
    # Missing values are deliberately LEFT missing in the cache: imputation is
    # part of preprocessing (public reference values; Config.PREPROCESSING),
    # so that no cohort statistic is baked into the feature table.

    keep = ["stay_id", "subject_id", "hospital_id", "mortality",
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
        "subject_id": np.arange(n_samples),
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
# Federated Client
# ---------------------------------------------------------------------------

class FederatedClient:
    def __init__(self, client_id: int, client_data: pd.DataFrame, preprocessor):
        self.client_id = client_id
        self.n_samples = len(client_data)
        self.mortality_rate = client_data["mortality"].mean()
        self.X = preprocessor.transform(client_data)
        self.y = client_data["mortality"].values.astype(np.float32)
        self.ethnicity = client_data["ethnicity"].values
        self.gender = client_data["gender"].values
        self.subjects = client_data["subject_id"].values

    def local_split_masks(self, test_size: float = 0.2, seed: int = 42):
        """Patient-disjoint local split: 20% of SUBJECTS -> local validation."""
        rng = np.random.RandomState(seed)
        uniq = np.sort(np.unique(self.subjects))
        perm = rng.permutation(len(uniq))
        n_val = max(1, int(round(test_size * len(uniq))))
        val_subj = uniq[perm[:n_val]]
        val_mask = np.isin(self.subjects, val_subj)
        return ~val_mask, val_mask

    def get_train_test_split(self, test_size: float = 0.2, seed: int = 42):
        tr, va = self.local_split_masks(test_size, seed)
        return self.X[tr], self.X[va], self.y[tr], self.y[va]


# ---------------------------------------------------------------------------
# Federation construction (single source of truth for main run + harnesses)
# ---------------------------------------------------------------------------

META_COLS = ["stay_id", "subject_id", "hospital_id", "mortality",
             "ethnicity", "gender", "insurance"]


def impute_features(data: pd.DataFrame, logger: Optional[logging.Logger] = None
                    ) -> pd.DataFrame:
    """Fill missing feature values with cohort medians computed before the
    split. Used ONLY by the PREPROCESSING="cohort" ablation control: these
    medians see validation/test rows and are not covered by the DP
    guarantee. Returns a new frame."""
    na_count = int(data[Config.FEATURE_COLS].isnull().sum().sum())
    if na_count == 0:
        return data
    data = data.copy()
    for col in Config.FEATURE_COLS:
        data[col] = data[col].fillna(data[col].median())
    if logger is not None:
        logger.warning(f"Imputed {na_count} NaN values across feature columns.")
    return data


def make_preprocessor(logger: Optional[logging.Logger] = None):
    if Config.PREPROCESSING == "reference":
        return dpc.ReferencePreprocessor()
    if Config.PREPROCESSING == "cohort":
        if logger is not None:
            logger.warning("PREPROCESSING='cohort': fitted statistics are NOT "
                           "covered by the DP guarantee (ablation control only).")
        return dpc.CohortPreprocessor()
    raise ValueError(f"unknown PREPROCESSING {Config.PREPROCESSING!r}")


def prepare_federation(data: pd.DataFrame, split_seed: int = 42,
                       split_unit: str = "patient",
                       design_fraction: Optional[float] = None,
                       logger: Optional[logging.Logger] = None) -> Dict:
    """Split the cohort 70/10/20 and build the federated clients.

    split_unit="patient" (the protocol): partitions are drawn over
    subject_id and each training patient is assigned to exactly one client
    (modal care unit of their training stays). split_unit="stay" (leaky
    control used by the split ablation): partitions are drawn over stays
    and clients are the raw care units.

    Within every client, a fraction `design_fraction` (default
    Config.DESIGN_FRACTION) of the training units is set aside as the DESIGN
    cohort, used only by the budget searches; the remainder is the
    DEPLOYMENT cohort on which every reported model is trained. The two are
    disjoint, so search runs spend no privacy budget of deployment patients.

    Preprocessing follows Config.PREPROCESSING ("reference": public
    constants only). Returns a dict with clients (deployment),
    design_clients, X/y for the validation and test partitions, and per-row
    metadata frames (meta_val / meta_test) aligned with those arrays.
    """
    if design_fraction is None:
        design_fraction = Config.DESIGN_FRACTION
    preprocessor = make_preprocessor(logger)
    if Config.PREPROCESSING == "cohort":
        data = impute_features(data, logger)
    data = data.reset_index(drop=True)
    if "subject_id" not in data.columns:
        if logger is not None:
            logger.warning("subject_id missing: treating each stay as its "
                           "own subject.")
        data = data.assign(subject_id=np.arange(len(data)))

    if split_unit == "patient":
        unit_of_row = data["subject_id"].values
    elif split_unit == "stay":
        unit_of_row = data.index.values
    else:
        raise ValueError(f"unknown split_unit {split_unit!r}")
    units = np.sort(np.unique(unit_of_row))
    perm = np.random.RandomState(split_seed).permutation(len(units))
    n_test = int(round(Config.TEST_SIZE * len(units)))
    n_val = int(round(Config.VAL_SIZE * len(units)))
    test_units = units[perm[:n_test]]
    val_units = units[perm[n_test:n_test + n_val]]
    part = np.where(np.isin(unit_of_row, test_units), "test",
                    np.where(np.isin(unit_of_row, val_units), "val", "train"))
    data = data.assign(_partition=part, _unit=unit_of_row)
    train_rows = data[data["_partition"] == "train"].copy()

    if split_unit == "patient":
        # Patient -> single client: modal care unit over the patient's
        # training stays (ties broken by pandas value_counts order).
        modal = (train_rows.groupby("subject_id")["hospital_id"]
                 .agg(lambda s: s.value_counts().index[0]))
        train_rows["client_id"] = train_rows["subject_id"].map(modal).astype(int)
        if logger is not None:
            moved = int((train_rows["client_id"] != train_rows["hospital_id"]).sum())
            multi = int((train_rows.groupby("subject_id")["hospital_id"]
                         .nunique() > 1).sum())
            max_stays = int(train_rows.groupby(["client_id", "subject_id"])
                            .size().max())
            logger.info(f"Patient->client assignment: {multi:,} multi-unit "
                        f"patients; {moved:,} stays reassigned "
                        f"({100 * moved / len(train_rows):.1f}%); max "
                        f"stays/patient within a client: {max_stays}")
    else:
        train_rows["client_id"] = train_rows["hospital_id"].astype(int)

    # Design / deployment cohorts: disjoint sets of units within each client.
    rng_design = np.random.RandomState(split_seed + 1000)
    is_design = np.zeros(len(train_rows), dtype=bool)
    client_ids = sorted(train_rows["client_id"].unique())[:Config.N_CLIENTS]
    for i in client_ids:
        in_client = (train_rows["client_id"] == i).values
        c_units = np.sort(train_rows.loc[in_client, "_unit"].unique())
        n_design = int(round(design_fraction * len(c_units)))
        chosen = c_units[rng_design.permutation(len(c_units))[:n_design]]
        is_design |= in_client & np.isin(train_rows["_unit"].values, chosen)
    train_rows["_design"] = is_design

    preprocessor.fit(train_rows[~train_rows["_design"]])

    def _clients(rows: pd.DataFrame, label: str):
        out = []
        for i in client_ids:
            cdf = rows[rows["client_id"] == i]
            c = FederatedClient(int(i), cdf.reset_index(drop=True), preprocessor)
            out.append(c)
            if logger is not None:
                logger.info(f"  {label} client {i}: {c.n_samples:,} stays "
                            f"({cdf['subject_id'].nunique():,} patients), "
                            f"mortality={c.mortality_rate:.1%}")
        return out

    clients = _clients(train_rows[~train_rows["_design"]], "Deployment")
    design_clients = (_clients(train_rows[train_rows["_design"]], "Design")
                      if design_fraction > 0 else clients)

    def _extract(partition: str):
        pdf = data[data["_partition"] == partition]
        y = pdf["mortality"].values.astype(np.float32)
        meta = pdf[[c for c in META_COLS if c in pdf.columns]].reset_index(drop=True)
        return preprocessor.transform(pdf), y, meta

    X_val, y_val, meta_val = _extract("val")
    X_test, y_test, meta_test = _extract("test")
    if logger is not None:
        logger.info(f"{split_unit.capitalize()}-level split (seed {split_seed}) "
                    f"-- train: {len(train_rows):,} stays "
                    f"(design {int(is_design.sum()):,} / deployment "
                    f"{int((~is_design).sum()):,}), val: {len(y_val):,}, "
                    f"test: {len(y_test):,} ({len(units):,} {split_unit} units); "
                    f"{dpc.N_FEATURES} features, {preprocessor.name} preprocessing")
    return {
        "clients": clients, "design_clients": design_clients,
        "preprocessor": preprocessor,
        "X_val": X_val, "y_val": y_val, "meta_val": meta_val,
        "X_test": X_test, "y_test": y_test, "meta_test": meta_test,
        "split_seed": split_seed, "split_unit": split_unit,
        "design_fraction": design_fraction,
    }


# ---------------------------------------------------------------------------
# Worker-process context
# ---------------------------------------------------------------------------
# The federation (client data, validation/test arrays) is installed once per
# worker process by the pool initializer instead of being pickled into every
# task, and the parent's Config is mirrored so that overrides (--fast,
# harness settings) also hold under the "spawn" start method.

_CTX: dict = {}


def _config_snapshot() -> dict:
    return {k: getattr(Config, k) for k in dir(Config) if k.isupper()}


def _init_worker(ctx: dict):
    global _CTX
    _CTX = ctx
    for k, v in ctx.get("config", {}).items():
        setattr(Config, k, v)


def validation_folds(subject_id: np.ndarray, split_seed: int) -> np.ndarray:
    """Two folds of the validation split, patients kept whole, for the
    cross-fitted search constraints (dp_core_v5.crossfit_predictions)."""
    uniq, inv = np.unique(subject_id, return_inverse=True)
    rng = np.random.RandomState(split_seed + 2000)
    return (rng.permutation(len(uniq)) % 2)[inv]


def make_context(fed: dict) -> dict:
    """Context shared with worker processes, built from prepare_federation()."""
    return {
        "clients": fed["clients"],                  # deployment cohort
        "design_clients": fed["design_clients"],    # budget-search cohort
        "X_val": fed["X_val"], "y_val": fed["y_val"],
        "val_fold": validation_folds(fed["meta_val"]["subject_id"].values,
                                     fed["split_seed"]),
        "eth_val": fed["meta_val"]["ethnicity"].values,
        "gen_val": fed["meta_val"]["gender"].values,
        "X_test": fed["X_test"], "y_test": fed["y_test"],
        "eth_test": fed["meta_test"]["ethnicity"].values,
        "gen_test": fed["meta_test"]["gender"].values,
        "config": _config_snapshot(),
    }


def make_pool(ctx: dict, n_jobs: int) -> Optional[Pool]:
    """Install ctx in this process and in n_jobs workers (None if n_jobs<=1)."""
    _init_worker(ctx)
    if n_jobs <= 1:
        return None
    return Pool(n_jobs, initializer=_init_worker, initargs=(ctx,))


def _pool_map(pool: Optional[Pool], fn, tasks: list) -> list:
    if pool is not None and len(tasks) > 1:
        return pool.map(fn, tasks, chunksize=1)
    return [fn(t) for t in tasks]


def _make_cfg() -> dict:
    return {
        "epsilon_max":      Config.EPSILON_MAX,
        "auc_min":          Config.AUC_MIN,
        "sensitivity_min":  Config.SENSITIVITY_MIN,
        "fairness_gap_max": Config.FAIRNESS_GAP_MAX,
        "fairness_objective": Config.FAIRNESS_OBJECTIVE,
    }


# ---------------------------------------------------------------------------
# Search fitness workers (top-level so they pickle under "spawn")
# ---------------------------------------------------------------------------
# Candidates are trained on the DESIGN cohort only, so a search spends no
# privacy budget of the deployment patients. Fitness is computed on the
# validation split, which also serves as the threshold-selection split
# inside the worker's trainer. A failing evaluation raises: a silent
# penalty value would corrupt the search.

def _moop_eval_worker(args):
    """One candidate of the 3-objective problem (privacy, utility, fairness)."""
    xi, eval_rounds, seed, cfg = args
    clients = _CTX["design_clients"]
    X_s, y_s = _CTX["X_val"], _CTX["y_val"]

    trainer = HealthcareFLTrainer(clients, X_s, y_s, seed=seed)
    trainer.reset(seed)
    client_sizes = {c.client_id: c.n_samples for c in clients}
    eps_alloc = {j: float(xi[j]) for j in range(len(clients))}
    fairness_eval = FairnessEvaluator(pd.DataFrame(
        {"ethnicity": _CTX["eth_val"], "gender": _CTX["gen_val"]}))

    result = trainer.train_federated(num_rounds=eval_rounds,
                                     epsilon_allocations=eps_alloc)
    m = result["final_metrics"]
    y_true = y_s.flatten().astype(int)
    # Decisions with a threshold chosen on OTHER validation patients.
    y_pred = dpc.crossfit_predictions(m["prediction_probs"], y_true,
                                      _CTX["val_fold"], Config.SENS_TARGET)
    sens = float(y_pred[y_true == 1].mean())
    fnr_gap = fairness_eval.max_fnr_gap(y_true, y_pred)

    f = np.zeros(3)
    f[0] = float(np.mean(xi)) / cfg["epsilon_max"]
    f[1] = 1.0 - m["auc"]
    if cfg["fairness_objective"] == "fnr_gap":
        f[2] = fnr_gap
    else:
        lv = fairness_eval.evaluate_all(
            y_true, y_pred, m["prediction_probs"],
            trainer.X_test, result["client_aucs"], eps_alloc, client_sizes)
        f[2] = (three_level_unfairness(lv)
                if cfg["fairness_objective"] == "three_level"
                else lv["fairness_unified"])

    g = np.zeros(3)
    g[0] = cfg["sensitivity_min"] - sens
    g[1] = cfg["auc_min"] - m["auc"]
    g[2] = fnr_gap - cfg["fairness_gap_max"]
    return f, g


def _mo_privacy_eval_worker(args):
    """One candidate of the 2-objective ablation (privacy, utility)."""
    xi, eval_rounds, seed, cfg = args
    trainer = HealthcareFLTrainer(_CTX["design_clients"], _CTX["X_val"],
                                  _CTX["y_val"], seed=seed)
    trainer.reset(seed)
    eps_alloc = {j: float(xi[j]) for j in range(len(xi))}
    result = trainer.train_federated(num_rounds=eval_rounds,
                                     epsilon_allocations=eps_alloc)
    m = result["final_metrics"]
    y_true = _CTX["y_val"].flatten().astype(int)
    y_pred = dpc.crossfit_predictions(m["prediction_probs"], y_true,
                                      _CTX["val_fold"], Config.SENS_TARGET)
    f = np.array([float(np.mean(xi)) / cfg["epsilon_max"], 1.0 - m["auc"]])
    g = np.array([cfg["sensitivity_min"] - float(y_pred[y_true == 1].mean()),
                  cfg["auc_min"] - m["auc"]])
    return f, g


# ---------------------------------------------------------------------------
# Federated Trainer
# ---------------------------------------------------------------------------

class HealthcareFLTrainer:
    def __init__(self, clients: List[FederatedClient],
                 X_test: np.ndarray, y_test: np.ndarray, seed: int = 42,
                 X_val: Optional[np.ndarray] = None,
                 y_val: Optional[np.ndarray] = None):
        self.clients = clients
        self.X_test = X_test
        self.y_test = y_test.reshape(-1, 1)
        self.X_val = X_val
        self.y_val = y_val.reshape(-1, 1) if y_val is not None else None
        self.seed = seed
        self.global_model = None
        self.training_history: List[Dict] = []
        # Positive-class weight w_pos = n_neg / n_pos.
        if Config.CLASS_WEIGHT == "prior":
            # From a public prevalence prior: no label statistic is released.
            self.w_pos = float((1.0 - Config.MORTALITY_PRIOR) / Config.MORTALITY_PRIOR)
        elif Config.CLASS_WEIGHT == "pooled":
            # Original submission (ablation control): pooled training labels,
            # a data-dependent statistic outside the DP accountant.
            all_y = np.concatenate([c.y for c in clients])
            n_pos = max(float(all_y.sum()), 1.0)
            self.w_pos = float(max(len(all_y) - n_pos, 1.0) / n_pos)
        else:
            raise ValueError(f"unknown CLASS_WEIGHT {Config.CLASS_WEIGHT!r}")

    def reset(self, seed: Optional[int] = None):
        if seed is not None:
            self.seed = seed
        np.random.seed(self.seed)
        self.global_model = None
        self.training_history = []

    @staticmethod
    def dp_schedule(n_units: int) -> Tuple[float, int, int]:
        """(q, steps_per_epoch, total_steps) of the certified protocol for a
        client with n_units privacy units (patients, or stays at record
        level) in its local training set. Each step Poisson-samples units at
        rate q = B / n_units. total_steps always refers to the FULL
        GLOBAL_ROUNDS protocol, the horizon sigma is calibrated to."""
        bs = Config.BATCH_SIZE
        q = min(1.0, bs / n_units)
        steps_per_epoch = max(1, int(np.ceil(n_units / bs)))
        total_steps = Config.GLOBAL_ROUNDS * Config.LOCAL_EPOCHS * steps_per_epoch
        return q, steps_per_epoch, total_steps

    def _local_units(self, client: FederatedClient):
        """(train mask, unit index of each local training row or None,
        number of privacy units) for this trainer's seed."""
        tr_mask, _ = client.local_split_masks(seed=self.seed)
        if Config.PRIVACY_UNIT == "patient":
            uniq, unit = np.unique(client.subjects[tr_mask], return_inverse=True)
            return tr_mask, unit, int(len(uniq))
        if Config.PRIVACY_UNIT == "record":
            return tr_mask, None, int(tr_mask.sum())
        raise ValueError(f"unknown PRIVACY_UNIT {Config.PRIVACY_UNIT!r}")

    def privacy_report(self, epsilon_allocations: Optional[Dict]) -> Dict:
        """Per-client mechanism parameters actually used for an allocation:
        budget, calibrated sigma, sampling rate, certified step count, and
        the number of local training records, patients and privacy units."""
        report = {}
        for client in self.clients:
            eps = (epsilon_allocations or {}).get(client.client_id, 1e6)
            tr_mask, _, n_units = self._local_units(client)
            entry = {"n_local": int(tr_mask.sum()),
                     "n_patients_local": int(len(np.unique(client.subjects[tr_mask]))),
                     "n_units": n_units, "privacy_unit": Config.PRIVACY_UNIT}
            if eps >= NONPRIVATE_EPS:
                entry.update(epsilon=float("inf"), sigma=0.0, q=float("nan"),
                             total_steps=0)
            else:
                q, _, total_steps = self.dp_schedule(n_units)
                entry.update(epsilon=float(eps), q=float(q),
                             total_steps=int(total_steps),
                             sigma=float(dpc.calibrate_sigma_cached(
                                 eps, Config.DELTA, q, total_steps)))
            if Config.NOISE_SCALE != 1.0:   # ablation run: no certificate
                entry.update(certified=False, noise_scale=Config.NOISE_SCALE)
            report[client.client_id] = entry
        return report

    def composed_epsilon(self, runs: List[Tuple[Dict, int]]) -> Dict:
        """Per-client epsilon of the COMPOSITION of several training runs on
        this trainer's clients. `runs` is a list of (allocation, rounds):
        each run uses the sigma calibrated for its allocation over the full
        protocol but executes only `rounds` rounds. This is what a party
        observing all of the runs (e.g. a whole budget search) learns."""
        out = {}
        for client in self.clients:
            _, _, n_units = self._local_units(client)
            q, steps_per_epoch, total_steps = self.dp_schedule(n_units)
            triples = []
            for alloc, rounds in runs:
                eps = alloc.get(client.client_id, 1e6)
                if eps >= NONPRIVATE_EPS:
                    return {c.client_id: float("inf") for c in self.clients}
                sigma = dpc.calibrate_sigma_cached(eps, Config.DELTA, q, total_steps)
                triples.append((q, sigma, rounds * Config.LOCAL_EPOCHS * steps_per_epoch))
            out[client.client_id] = dpc.eps_from_rdp_runs(triples, Config.DELTA)
        return out

    def local_training_with_dp(self, client: FederatedClient,
                                global_weights, epsilon: float,
                                local_epochs: Optional[int] = None):
        """Certified DP-SGD at the configured privacy unit.

        Private path (epsilon < NONPRIVATE_EPS): each step Poisson-samples
        privacy units at rate q = B / n_units. With PRIVACY_UNIT="patient"
        a sampled patient contributes ALL of their local stays and the SUM
        of those stays' gradients is l2-clipped to C, so adding or removing
        one patient changes the summed update by at most C (patient-level
        adjacency). With "record" every stay is its own unit. Gaussian noise
        N(0, (sigma*C)^2) is added to the per-step sum. The noise multiplier
        is calibrated once per (epsilon, q, steps) by the RDP accountant so
        that the FULL protocol (GLOBAL_ROUNDS x LOCAL_EPOCHS x
        steps_per_epoch steps) satisfies the client's total
        (epsilon, delta)-DP budget.
        Non-private path: standard minibatch SGD, no clipping, no noise.
        """
        if local_epochs is None:
            local_epochs = Config.LOCAL_EPOCHS

        local_model = create_model(
            learning_rate=(Config.LEARNING_RATE if epsilon >= NONPRIVATE_EPS
                           else Config.DP_LEARNING_RATE))
        if global_weights is not None:
            local_model.set_weights(global_weights)

        tr_mask, unit, n_units = self._local_units(client)
        X_train, y_train = client.X[tr_mask], client.y[tr_mask].reshape(-1, 1)
        X_val, y_val = client.X[~tr_mask], client.y[~tr_mask].reshape(-1, 1)
        n_local = len(X_train)
        w_pos = self.w_pos
        rng = np.random.default_rng(np.random.randint(0, 2**31 - 1))
        bs = Config.BATCH_SIZE

        if epsilon >= NONPRIVATE_EPS:
            n_batches = max(1, n_local // bs)
            for _ in range(local_epochs):
                indices = np.random.permutation(n_local)
                for b in range(n_batches):
                    s, e = b * bs, min((b + 1) * bs, n_local)
                    Xb, yb = X_train[indices[s:e]], y_train[indices[s:e]]
                    _, cache = local_model._forward(Xb, training=True)
                    local_model._apply_adam(
                        dpc.batch_grads(local_model, yb, cache, w_pos))
        else:
            q, steps_per_epoch, total_steps = self.dp_schedule(n_units)
            sigma = dpc.calibrate_sigma_cached(epsilon, Config.DELTA, q,
                                               total_steps) * Config.NOISE_SCALE
            C = Config.CLIPPING_NORM
            for _ in range(local_epochs * steps_per_epoch):
                sampled = rng.random(n_units) < q
                if not sampled.any():
                    continue
                mask = sampled if unit is None else sampled[unit]
                Xb, yb = X_train[mask], y_train[mask]
                _, cache = local_model._forward(Xb, training=True)
                local_model._apply_adam(dpc.clipped_noisy_grads(
                    local_model, yb, cache, w_pos, C, sigma, bs, rng,
                    groups=None if unit is None else unit[mask]))

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
        """Uniform v5 evaluation: decision threshold selected on the
        validation split by the prespecified rule (dp_core_v5), then applied
        unchanged to the test split. Returns AUROC/AUPRC/threshold/
        sens/spec/PPV/NPV plus predictions at the chosen threshold."""
        if self.global_model is None:
            return {"accuracy": 0.0, "auc": 0.5, "auprc": 0.0, "loss": 1.0,
                    "sensitivity": 0.0, "specificity": 0.0, "ppv": 0.0,
                    "npv": 0.0, "threshold": 0.5,
                    "predictions": np.array([]),
                    "prediction_probs": np.array([])}
        X_thr = self.X_val if self.X_val is not None else self.X_test
        y_thr = self.y_val if self.y_val is not None else self.y_test
        val_probs = self.global_model.predict(X_thr, verbose=0).flatten()
        tau = dpc.select_threshold(val_probs, y_thr, Config.SENS_TARGET)
        probs = self.global_model.predict(self.X_test, verbose=0).flatten()
        m = dpc.threshold_metrics(self.y_test, probs, tau)
        m["val_probs"] = val_probs
        y_f = self.y_test.ravel()
        m["loss"] = float(-np.mean(y_f * np.log(probs + 1e-8)
                                   + (1 - y_f) * np.log(1 - probs + 1e-8)))
        return m

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

            # v5: per-round trajectory logged on the VALIDATION split only;
            # the test split is evaluated once, after training completes.
            if self.X_val is not None:
                from sklearn.metrics import roc_auc_score
                vp = self.global_model.predict(self.X_val, verbose=0).flatten()
                yv = self.y_val.ravel()
                try:
                    v_auc = float(roc_auc_score(yv, vp))
                except ValueError:
                    v_auc = 0.5
                v_loss = float(-np.mean(yv * np.log(vp + 1e-8)
                                        + (1 - yv) * np.log(1 - vp + 1e-8)))
                self.training_history.append({
                    "round": rnd + 1, "auc": v_auc, "loss": v_loss,
                    "accuracy": 0.0, "sensitivity": 0.0,
                })
                if verbose and (rnd + 1) % 10 == 0:
                    log.info(f"  Round {rnd+1}/{num_rounds}: val AUC={v_auc:.3f}")

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
                         k: int = 10, n_sub: int = 5000,
                         scale: str = "rank") -> float:
        """Local prediction consistency: 1 - mean absolute score difference
        among the k nearest neighbours in NON-PROTECTED feature space
        (continuous clinical features only). Seeded for determinism.

        scale="rank" (default) compares the scores' percentile ranks within
        the evaluated cohort, so the value is unchanged by any monotone
        rescaling of the model output. scale="raw" compares raw
        probabilities (the submitted definition): a model whose scores are
        all close to zero then looks consistent merely because its scores
        have a small range, which is why the raw value is reported only as
        a documented artefact."""
        from sklearn.neighbors import NearestNeighbors
        from scipy.stats import rankdata
        rng = np.random.default_rng(0)
        n = min(len(X_test), n_sub)
        idx = rng.choice(len(X_test), n, replace=False)
        X_sub = X_test[idx][:, :dpc.N_CONTINUOUS]
        scores = y_pred_prob.flatten()
        if scale == "rank":
            scores = rankdata(scores) / len(scores)
        elif scale != "raw":
            raise ValueError(f"unknown scale {scale!r}")
        p_sub = scores[idx]
        nn = NearestNeighbors(n_neighbors=k + 1, algorithm="ball_tree").fit(X_sub)
        _, indices = nn.kneighbors(X_sub)
        diffs = np.abs(p_sub[indices[:, 1:]] - p_sub[:, None])
        return max(0.0, float(1.0 - diffs.mean()))

    def max_fnr_gap(self, y_true, y_pred) -> float:
        """Maximum per-attribute false-negative-rate gap (v5 constraint g3):
        the largest max-min FNR difference over the subgroups of any single
        protected attribute (>5 positive cases per subgroup)."""
        gaps = [0.0]
        for attr in ["ethnicity", "gender"]:
            values = self.demographics[attr].values
            fnrs = []
            for g in np.unique(values):
                mask = (values == g) & (y_true == 1)
                if mask.sum() > 5:
                    fnrs.append(np.mean(y_pred[mask] == 0))
            if len(fnrs) >= 2:
                gaps.append(float(max(fnrs) - min(fnrs)))
        return float(max(gaps))

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
            "fairness_individual_raw": float(
                self.individual_level(X_test, y_pred_prob, scale="raw")),
            "max_fnr_gap": float(self.max_fnr_gap(y_true, y_pred)),
        }


def three_level_unfairness(levels: Dict) -> float:
    """Composite without the individual and system levels: outcome, group
    and client levels with their original weights renormalised to sum to 1.
    Used for the objective sensitivity analysis (reviewer comment R2.5)."""
    w = Config.W_OUTCOME + Config.W_GROUP + Config.W_CLIENT
    return float(1.0 - (Config.W_OUTCOME * levels["fairness_outcome"]
                        + Config.W_GROUP * levels["fairness_group"]
                        + Config.W_CLIENT * levels["fairness_client"]) / w)


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


def run_fairfed(trainer: HealthcareFLTrainer, seed: int = 42, lambda_fair: float = 1.0) -> Dict:
    trainer.reset(seed)

    def _agg(updates, rnd, t):
        if t.global_model is None:
            return t.federated_averaging(updates)
        gaps = []
        for client in t.clients:
            _, val_mask = client.local_split_masks(seed=t.seed)
            preds = (t.global_model.predict(client.X[val_mask], verbose=0)
                     > 0.5).astype(int).flatten()
            eth_val = client.ethnicity[val_mask]
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
    """FCFL-style baseline (v5): standard non-private training followed by
    global Platt recalibration of predicted probabilities (fit on the
    validation split). The decision threshold then follows the SAME
    prespecified validation rule as every other method."""
    from sklearn.linear_model import LogisticRegression

    trainer.reset(seed)
    eps = {c.client_id: 1e6 for c in trainer.clients}
    result = trainer.train_federated(num_rounds=Config.GLOBAL_ROUNDS,
                                     epsilon_allocations=eps)

    X_thr = trainer.X_val if trainer.X_val is not None else trainer.X_test
    y_thr = (trainer.y_val if trainer.y_val is not None
             else trainer.y_test).ravel()
    val_probs = trainer.global_model.predict(X_thr, verbose=0).flatten()
    lr = LogisticRegression(class_weight="balanced", max_iter=300)
    lr.fit(val_probs.reshape(-1, 1), y_thr.astype(int))
    cal_val = lr.predict_proba(val_probs.reshape(-1, 1))[:, 1]
    tau = dpc.select_threshold(cal_val, y_thr, Config.SENS_TARGET)

    raw_test = trainer.global_model.predict(trainer.X_test, verbose=0).flatten()
    cal_test = lr.predict_proba(raw_test.reshape(-1, 1))[:, 1]
    m = dpc.threshold_metrics(trainer.y_test, cal_test, tau)
    yf = trainer.y_test.ravel()
    m["loss"] = float(-np.mean(yf * np.log(cal_test + 1e-8)
                               + (1 - yf) * np.log(1 - cal_test + 1e-8)))
    m["val_probs"] = cal_val
    m["raw_val_probs"] = val_probs
    m["raw_test_probs"] = raw_test
    result["final_metrics"] = m
    result["method"] = "FCFL"
    result["avg_epsilon"] = float("inf")
    return result


# ---------------------------------------------------------------------------
# Budget-allocation searches (NSGA-II and the LHS control)
# ---------------------------------------------------------------------------
# All searches read the federation from the worker context (make_pool).
# Candidates are trained on the DESIGN cohort and scored on the validation
# split; the selected allocation is then trained once, for the full
# protocol, on the DEPLOYMENT cohort (the `trainer` argument). Each search
# returns that model, the complete archive (every evaluated allocation with
# its objectives and constraints) and the composed privacy cost of the
# search on the design cohort.

HV_REF_POINT = np.array([1.1, 0.5, 0.5])


def _best_compromise(F: np.ndarray) -> int:
    """Index of the solution minimizing the sum of max-normalized objectives."""
    F_norm = F / (F.max(axis=0) + 1e-8)
    return int(np.argmin(F_norm.sum(axis=1)))


def _run_nsga2(worker, n_obj: int, n_constr: int, seed: int,
               pool: Optional[Pool]):
    """NSGA-II over per-client budgets. Returns (pymoo result, archive,
    per-generation population objectives)."""
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.core.callback import Callback
    from pymoo.core.problem import Problem
    from pymoo.optimize import minimize
    from pymoo.operators.crossover.sbx import SBX
    from pymoo.operators.mutation.pm import PM
    from pymoo.operators.sampling.rnd import FloatRandomSampling

    cfg = _make_cfg()
    archive = {"X": [], "F": [], "G": []}

    class _Prob(Problem):
        def __init__(self):
            super().__init__(n_var=Config.N_CLIENTS, n_obj=n_obj,
                             n_constr=n_constr,
                             xl=Config.EPSILON_MIN, xu=Config.EPSILON_MAX)
            self.cache: Dict = {}

        def _evaluate(self, X, out, *args, **kwargs):
            F, G = np.zeros((len(X), n_obj)), np.zeros((len(X), n_constr))
            to_eval = []
            for i, xi in enumerate(X):
                key = tuple(np.round(xi, 2))
                if key in self.cache:
                    F[i], G[i] = self.cache[key]
                else:
                    to_eval.append((i, xi))
            raw = _pool_map(pool, worker,
                            [(xi, Config.EVAL_ROUNDS, seed, cfg)
                             for (_, xi) in to_eval])
            for (i, xi), (fi, gi) in zip(to_eval, raw):
                F[i], G[i] = fi, gi
                self.cache[tuple(np.round(xi, 2))] = (fi.copy(), gi.copy())
                archive["X"].append(np.array(xi, dtype=float))
                archive["F"].append(fi.copy())
                archive["G"].append(gi.copy())
            out["F"], out["G"] = F, G

    class _GenLog(Callback):
        def __init__(self):
            super().__init__()
            self.gen_F = []

        def notify(self, algorithm):
            self.gen_F.append(np.array(algorithm.pop.get("F")))

    genlog = _GenLog()
    algo = NSGA2(pop_size=Config.POP_SIZE, n_offsprings=Config.N_OFFSPRINGS,
                 sampling=FloatRandomSampling(), crossover=SBX(prob=0.9, eta=15),
                 mutation=PM(eta=20), eliminate_duplicates=True)
    res = minimize(_Prob(), algo, termination=("n_gen", Config.N_GENERATIONS),
                   seed=seed, verbose=False, callback=genlog)
    return res, {k: np.array(v) for k, v in archive.items()}, genlog.gen_F


def select_front(X: np.ndarray, F: np.ndarray, G: np.ndarray):
    """The one selection rule shared by every search, applied to the
    complete archive of evaluated candidates:
      * if any candidate satisfies all constraints, the non-dominated set of
        the feasible candidates;
      * otherwise the non-dominated set of the POP_SIZE candidates with the
        smallest total constraint violation (feasibility-first ranking).
    Returns (indices into the archive, number of feasible candidates)."""
    from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
    cv = np.clip(np.asarray(G), 0, None).sum(axis=1)
    feasible = cv <= 1e-12
    pool_idx = (np.where(feasible)[0] if feasible.any()
                else np.argsort(cv, kind="stable")[:Config.POP_SIZE])
    nd = NonDominatedSorting().do(np.asarray(F)[pool_idx],
                                  only_non_dominated_front=True)
    return pool_idx[nd], int(feasible.sum())


def _train_selected(trainer: HealthcareFLTrainer, archive: Dict, seed: int,
                    logger: Optional[logging.Logger] = None) -> Tuple[Dict, Dict]:
    """Select a front from the archive (select_front), fully train its
    best-compromise allocation. Returns (result, alloc)."""
    X, F, G = archive["X"], archive["F"], archive["G"]
    front_idx, n_feasible = select_front(X, F, G)
    front_X, front_F = X[front_idx], F[front_idx]
    if n_feasible == 0:
        (logger or log).warning(
            "  no candidate satisfies all constraints; selecting among the "
            f"{Config.POP_SIZE} least-violating candidates")
    best = _best_compromise(front_F)
    best_eps = {j: float(front_X[best][j]) for j in range(Config.N_CLIENTS)}
    trainer.reset(seed)
    result = trainer.train_federated(num_rounds=Config.GLOBAL_ROUNDS,
                                     epsilon_allocations=best_eps)
    result["pareto_X"], result["pareto_F"] = front_X, front_F
    result["n_pareto"] = int(len(front_X))
    result["n_feasible"] = n_feasible
    result["selection"] = "feasible" if n_feasible else "least_violation"
    result["best_allocation"] = best_eps
    result["avg_epsilon"] = float(np.mean(list(best_eps.values())))
    result["search_X"], result["search_F"], result["search_G"] = X, F, G
    result["n_evaluations"] = int(len(X))
    result["search_composition"] = search_composition(X, best_eps, seed)
    return result, best_eps


def search_composition(search_X, best_eps: Dict, seed: int) -> Dict:
    """Privacy cost of the SEARCH itself, per client of the design cohort.

    Every candidate evaluation is a DP training run on the same design
    patients, so the runs compose. Reports
      * design_cohort: epsilon after all EVAL_ROUNDS-round candidate runs --
        the guarantee the design patients actually receive from this search;
      * naive_workflow: the same plus one full training run of the selected
        allocation -- what every training patient would receive if the
        search and the final model shared one cohort (no design split).
    """
    trainer = HealthcareFLTrainer(_CTX["design_clients"], _CTX["X_val"],
                                  _CTX["y_val"], seed=seed)
    runs = [({j: float(x[j]) for j in range(Config.N_CLIENTS)}, Config.EVAL_ROUNDS)
            for x in np.asarray(search_X)]
    return {
        "n_candidate_runs": len(runs),
        "eval_rounds": Config.EVAL_ROUNDS,
        "design_cohort": trainer.composed_epsilon(runs),
        "naive_workflow": trainer.composed_epsilon(
            runs + [(best_eps, Config.GLOBAL_ROUNDS)]),
    }


def run_mo_privacy(trainer: HealthcareFLTrainer, seed: int = 42,
                   pool: Optional[Pool] = None) -> Dict:
    """Two-objective ablation: NSGA-II over (privacy, utility) only."""
    res, archive, _ = _run_nsga2(_mo_privacy_eval_worker, 2, 2, seed, pool)
    result, _ = _train_selected(trainer, archive, seed)
    result["method"] = "MO-Privacy"
    return result


def run_moop_healthfl(trainer: HealthcareFLTrainer, seed: int = 42,
                      pool: Optional[Pool] = None,
                      logger: Optional[logging.Logger] = None) -> Dict:
    """Full three-objective NSGA-II search (privacy, utility, fairness)."""
    logger = logger or log
    logger.info(f"  [MOOP] NSGA-II pop={Config.POP_SIZE}, "
                f"gen={Config.N_GENERATIONS}, eval_rounds={Config.EVAL_ROUNDS}, "
                f"workers={pool._processes if pool is not None else 1}")
    res, archive, gen_F = _run_nsga2(_moop_eval_worker, 3, 3, seed, pool)
    logger.info("  [MOOP] Running full training with best allocation...")
    result, _ = _train_selected(trainer, archive, seed, logger)
    from pymoo.indicators.hv import HV
    hv_ind = HV(ref_point=HV_REF_POINT)
    result["hv_per_gen"] = [float(hv_ind(F[np.all(np.isfinite(F), axis=1)]))
                            for F in gen_F]
    result["method"] = "MOOP-HealthFL"
    return result


def lhs_candidates(n_cand: int, seed: int) -> np.ndarray:
    """Latin-hypercube sample of n_cand allocations over the budget box."""
    K = Config.N_CLIENTS
    rng = np.random.default_rng(seed)
    strata = np.tile(np.arange(n_cand, dtype=float), (K, 1)).T
    for j in range(K):
        rng.shuffle(strata[:, j])
    u = (strata + rng.random((n_cand, K))) / n_cand
    return Config.EPSILON_MIN + u * (Config.EPSILON_MAX - Config.EPSILON_MIN)


def run_lhs_search(trainer: HealthcareFLTrainer, seed: int = 42,
                   pool: Optional[Pool] = None,
                   logger: Optional[logging.Logger] = None) -> Dict:
    """Latin-hypercube random search with the identical evaluation budget
    (POP + (G-1)*OFFSPRING candidates), worker, constraints, and selection
    rule -- the search-strategy control for NSGA-II."""
    logger = logger or log
    cfg = _make_cfg()
    n_cand = Config.POP_SIZE + (Config.N_GENERATIONS - 1) * Config.N_OFFSPRINGS
    Xc = lhs_candidates(n_cand, seed)

    logger.info(f"  [LHS] evaluating {n_cand} candidates "
                f"(eval_rounds={Config.EVAL_ROUNDS})")
    raw = _pool_map(pool, _moop_eval_worker,
                    [(Xc[i], Config.EVAL_ROUNDS, seed, cfg)
                     for i in range(n_cand)])
    F = np.array([r[0] for r in raw])
    G = np.array([r[1] for r in raw])

    result, _ = _train_selected(trainer, {"X": Xc, "F": F, "G": G}, seed, logger)
    from pymoo.indicators.hv import HV
    result["hv_final"] = float(HV(ref_point=HV_REF_POINT)(
        np.asarray(result["pareto_F"])))
    result["method"] = "LHS-Search"
    return result


# ---------------------------------------------------------------------------
# Method registry, result finalisation, saved predictions
# ---------------------------------------------------------------------------

BASELINES = {
    "fedavg":         ("FedAvg",          lambda t, s: run_fedavg(t, s)),
    "fixed_eps_1.0":  ("Fixed-eps(1.0)",  lambda t, s: run_fixed_eps(t, 1.0, s)),
    "fixed_eps_2.0":  ("Fixed-eps(2.0)",  lambda t, s: run_fixed_eps(t, 2.0, s)),
    "fixed_eps_5.0":  ("Fixed-eps(5.0)",  lambda t, s: run_fixed_eps(t, 5.0, s)),
    "fixed_eps_10.0": ("Fixed-eps(10.0)", lambda t, s: run_fixed_eps(t, 10.0, s)),
    "fairfed":        ("FairFed",         lambda t, s: run_fairfed(t, s)),
    "fcfl":           ("FCFL",            lambda t, s: run_fcfl(t, s)),
}
SEARCHES = {
    "mo_privacy":    ("MO-Privacy",    lambda t, s, pool, lg: run_mo_privacy(t, s, pool)),
    "lhs_search":    ("LHS-Search",    lambda t, s, pool, lg: run_lhs_search(t, s, pool, lg)),
    "moop_healthfl": ("MOOP-HealthFL", lambda t, s, pool, lg: run_moop_healthfl(t, s, pool, lg)),
}
METHOD_ORDER = list(BASELINES) + list(SEARCHES)


def _context_trainer(seed: int) -> "HealthcareFLTrainer":
    c = _CTX
    return HealthcareFLTrainer(c["clients"], c["X_test"], c["y_test"],
                               seed=seed, X_val=c["X_val"], y_val=c["y_val"])


def finalize_result(result: Dict, trainer: "HealthcareFLTrainer") -> Dict:
    """Attach the five-level fairness evaluation (test split) and the
    per-client privacy report to a trained method's result."""
    m = result["final_metrics"]
    clients = trainer.clients
    if result.get("best_allocation"):
        eps_alloc = result["best_allocation"]
    elif result.get("avg_epsilon", float("inf")) != float("inf"):
        eps_alloc = {c.client_id: result["avg_epsilon"] for c in clients}
    else:
        eps_alloc = None
    fairness_eval = FairnessEvaluator(pd.DataFrame(
        {"ethnicity": _CTX["eth_test"], "gender": _CTX["gen_test"]}))
    result["fairness"] = fairness_eval.evaluate_all(
        trainer.y_test.flatten().astype(int), m["predictions"],
        m["prediction_probs"], trainer.X_test,
        result.get("client_aucs", [0.5] * Config.N_CLIENTS),
        eps_alloc, {c.client_id: c.n_samples for c in clients})
    result["privacy"] = trainer.privacy_report(eps_alloc)
    return result


def _baseline_worker(args):
    """Train one (method, seed) baseline end to end in a worker process."""
    method_key, seed = args
    trainer = _context_trainer(seed)
    t0 = time.time()
    result = BASELINES[method_key][1](trainer, seed)
    result = finalize_result(result, trainer)
    result["elapsed_s"] = time.time() - t0
    return method_key, seed, result


_ARRAY_KEYS = ("predictions", "prediction_probs", "val_probs",
               "raw_val_probs", "raw_test_probs")


def predictions_path(pred_dir: str, seed, method_key: str) -> str:
    return os.path.join(pred_dir, f"seed{seed}_{method_key}.npz")


def save_predictions(pred_dir: str, seed, method_key: str, result: Dict):
    """Write a trained model's validation/test probabilities, threshold and
    per-client mechanism parameters to <pred_dir>/seed<seed>_<method>.npz,
    then drop the large arrays from the result (they do not belong in the
    checkpoint). Rows align with eval_split.npz."""
    m = result["final_metrics"]
    rep = result["privacy"]
    ids = sorted(rep)
    arrays = {
        "test_probs": np.asarray(m["prediction_probs"]).ravel(),
        "val_probs": np.asarray(m["val_probs"]).ravel(),
        "threshold": np.float64(m["threshold"]),
        "client_ids": np.array(ids),
    }
    arrays["privacy_unit"] = np.array(rep[ids[0]]["privacy_unit"])
    for field in ("epsilon", "sigma", "q", "total_steps", "n_local",
                  "n_patients_local", "n_units"):
        arrays[field] = np.array([rep[i][field] for i in ids], dtype=float)
    for k in ("raw_val_probs", "raw_test_probs"):
        if k in m:
            arrays[k] = np.asarray(m[k]).ravel()
    os.makedirs(pred_dir, exist_ok=True)
    np.savez_compressed(predictions_path(pred_dir, seed, method_key), **arrays)
    for k in _ARRAY_KEYS:
        m.pop(k, None)


def save_eval_split(pred_dir: str, fed: Dict):
    """Labels and per-row metadata of the validation and test partitions
    (aligned with every saved probability vector)."""
    arrays = {"split_seed": np.int64(fed["split_seed"])}
    for part in ("val", "test"):
        arrays[f"{part}_y"] = fed[f"y_{part}"].astype(np.int8)
        for col in fed[f"meta_{part}"].columns:
            arrays[f"{part}_{col}"] = fed[f"meta_{part}"][col].values
    os.makedirs(pred_dir, exist_ok=True)
    np.savez_compressed(os.path.join(pred_dir, "eval_split.npz"), **arrays)


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
        for metric in ["auc", "auprc", "accuracy", "sensitivity",
                       "specificity", "ppv", "npv", "threshold"]:
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
        "fairfed": "#d62728", "fcfl": "#9467bd",
        "mo_privacy": "#8c564b",
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
    eps_vals = [1.0, 2.0, 5.0, 10.0]
    keys = ["fixed_eps_1.0", "fixed_eps_2.0", "fixed_eps_5.0", "fixed_eps_10.0"]
    aucs_m = [_m(k, "auc")[0] for k in keys]
    aucs_s = [_m(k, "auc")[1] for k in keys]
    ax.errorbar(eps_vals, aucs_m, yerr=aucs_s, marker="s", color=COLORS["fixed_eps"],
                label="Fixed-$\\varepsilon$ DP", capsize=3, linewidth=1.5)
    for mk, label, marker in [
        ("fedavg", "FedAvg (no DP)", "o"),
        ("fairfed", "FairFed", "D"), ("fcfl", "FCFL", "v"),
        ("mo_privacy", "MO-Privacy", "p"), ("lhs_search", "LHS-Search", "h"),
        ("moop_healthfl", "MOOP-HealthFL", "*"),
    ]:
        if mk not in agg:
            continue
        eps = agg[mk].get("avg_epsilon", float("inf"))
        eps_plot = min(eps, 12.0) if eps != float("inf") else 12.0
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
    ax.set_xlim(0, 12.5)
    _save("fig_privacy_utility.pdf")

    # Figure 2: Convergence
    first_seed = str(Config.SEEDS[0])
    fig, ax = plt.subplots(figsize=(7, 5))
    for mk, label, color in [
        ("fedavg", "FedAvg", COLORS["fedavg"]),
        ("fixed_eps_1.0", "Fixed-$\\varepsilon$=1.0", COLORS["fixed_eps"]),
        ("fcfl", "FCFL", COLORS["fcfl"]),
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
    ax.set_ylabel("Validation AUROC")
    ax.set_title("Training Convergence (validation split)")
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

    order = ["fedavg", "fixed_eps_1.0", "fixed_eps_2.0", "fixed_eps_5.0",
             "fixed_eps_10.0", "fairfed", "fcfl",
             "mo_privacy", "lhs_search", "moop_healthfl"]
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

CACHE_NAME = "mimic_iv_features_v2.parquet"   # retains subject_id


def load_data(mimic_path: Optional[str], cache_file: str,
              logger: logging.Logger) -> pd.DataFrame:
    """Feature cache if present, else build it from raw MIMIC-IV, else fall
    back to synthetic data (pipeline smoke tests only)."""
    if os.path.exists(cache_file) or (mimic_path and os.path.exists(mimic_path)):
        return load_mimic_data(mimic_path or "", cache_file, logger)
    logger.warning("No feature cache and no MIMIC-IV path -- using SYNTHETIC "
                   "data (smoke test only; results are meaningless).")
    return generate_synthetic_data()


def run_experiments(args, logger: logging.Logger):
    output_dir = args.output_dir
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    figures_dir = os.path.join(output_dir, "figures")
    results_dir = os.path.join(output_dir, "results")
    pred_dir = os.path.join(output_dir, "predictions")
    for d in [output_dir, checkpoint_dir, figures_dir, results_dir, pred_dir]:
        os.makedirs(d, exist_ok=True)

    if args.fast:
        Config.apply_fast_mode()
        logger.info("FAST MODE enabled.")
    Config.N_JOBS = args.n_jobs
    if args.seed is not None:
        Config.SEEDS = [args.seed]
    for arg, attr in [("privacy_unit", "PRIVACY_UNIT"),
                      ("preprocessing", "PREPROCESSING"),
                      ("class_weight", "CLASS_WEIGHT"),
                      ("design_fraction", "DESIGN_FRACTION"),
                      ("fairness_objective", "FAIRNESS_OBJECTIVE"),
                      ("clip_norm", "CLIPPING_NORM"),
                      ("dp_lr", "DP_LEARNING_RATE"),
                      ("eval_rounds", "EVAL_ROUNDS")]:
        if getattr(args, arg, None) is not None:
            setattr(Config, attr, getattr(args, arg))
    if getattr(args, "drop_features", None):
        dpc.DROP_GROUPS = tuple(args.drop_features)
        logger.info(f"ABLATION: model inputs without {dpc.DROP_GROUPS}")
    methods = args.methods or METHOD_ORDER
    unknown = [k for k in methods if k not in METHOD_ORDER]
    if unknown:
        raise SystemExit(f"Unknown method(s) {unknown}; choose from {METHOD_ORDER}")

    logger.info("=" * 70)
    logger.info("MOOP-HealthFL -- certified DP-SGD (RDP accountant)")
    logger.info(f"Seeds:          {Config.SEEDS} (data split seed {args.split_seed})")
    logger.info(f"Privacy:        unit={Config.PRIVACY_UNIT}, "
                f"preprocessing={Config.PREPROCESSING}, "
                f"class weight={Config.CLASS_WEIGHT}, "
                f"design fraction={Config.DESIGN_FRACTION}")
    logger.info(f"DP training:    C={Config.CLIPPING_NORM}, lot={Config.BATCH_SIZE}, "
                f"lr={Config.DP_LEARNING_RATE} (non-private lr={Config.LEARNING_RATE})")
    logger.info(f"Search:         fairness objective={Config.FAIRNESS_OBJECTIVE}, "
                f"constraints sens>={Config.SENSITIVITY_MIN}, "
                f"AUROC>={Config.AUC_MIN}, FNR gap<={Config.FAIRNESS_GAP_MAX} "
                f"(cross-fitted on validation)")
    logger.info(f"Global rounds:  {Config.GLOBAL_ROUNDS}")
    logger.info(f"NSGA-II:        pop={Config.POP_SIZE}, gen={Config.N_GENERATIONS}, "
                f"eval_rounds={Config.EVAL_ROUNDS}")
    logger.info(f"Workers:        {Config.N_JOBS}")
    logger.info("=" * 70)

    cache_file = args.cache_file or os.path.join(output_dir, CACHE_NAME)
    data = load_data(args.mimic_path, cache_file, logger)
    logger.info(f"Dataset: {len(data):,} stays, mortality={data['mortality'].mean():.1%}")

    fed = prepare_federation(data, split_seed=args.split_seed, logger=logger)
    save_eval_split(pred_dir, fed)
    with open(os.path.join(results_dir, "run_config.json"), "w") as f:
        json.dump({**_config_snapshot(), "split_seed": args.split_seed,
                   "methods": methods}, f, indent=2, default=str)

    ckpt = CheckpointManager(checkpoint_dir)
    all_results = ckpt.load() or {}
    for seed in Config.SEEDS:
        all_results.setdefault(str(seed), {})

    def _done(seed, key) -> bool:
        r = all_results[str(seed)].get(key)
        return (r is not None and "error" not in r
                and os.path.exists(predictions_path(pred_dir, seed, key)))

    def _store(seed, key, name, result):
        m = result["final_metrics"]
        logger.info(f"  [seed {seed}] [{name}] {result.get('elapsed_s', 0):.0f}s -- "
                    f"AUC={m['auc']:.4f}, Sens={m['sensitivity']:.3f}, "
                    f"Fair={result['fairness']['fairness_unified']:.3f}")
        save_predictions(pred_dir, seed, key, result)
        all_results[str(seed)][key] = result
        ckpt.save(all_results)

    pool = make_pool(make_context(fed), Config.N_JOBS)
    logger.info(f"Pool: {'%d workers' % Config.N_JOBS if pool else 'sequential'}")
    t_start = time.time()
    try:
        # Baselines: one worker per (method, seed), all seeds at once.
        tasks = [(k, s) for s in Config.SEEDS for k in BASELINES
                 if k in methods and not _done(s, k)]
        if tasks:
            logger.info(f"\nRunning {len(tasks)} baseline (method, seed) runs...")
            done_iter = (pool.imap_unordered(_baseline_worker, tasks, chunksize=1)
                         if pool is not None else map(_baseline_worker, tasks))
            for key, seed, result in done_iter:
                _store(seed, key, BASELINES[key][0], result)

        # Searches: one at a time, each using the whole pool for candidates.
        for seed in Config.SEEDS:
            for key, (name, fn) in SEARCHES.items():
                if key not in methods:
                    continue
                if _done(seed, key):
                    logger.info(f"  [seed {seed}] [{name}] already done -- skipping.")
                    continue
                logger.info(f"\n  [seed {seed}] Running [{name}]...")
                trainer = _context_trainer(seed)
                t0 = time.time()
                result = finalize_result(fn(trainer, seed, pool, logger), trainer)
                result["elapsed_s"] = time.time() - t0
                _store(seed, key, name, result)
    finally:
        if pool is not None:
            pool.close()
            pool.join()
    logger.info(f"All methods finished in {(time.time() - t_start) / 60:.1f} min.")

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

    print_latex_tables(agg)

    logger.info("\nGenerating figures...")
    try:
        generate_figures(agg, all_results, figures_dir, results_dir, logger)
    except Exception as exc:
        logger.warning(f"Figure generation failed: {exc}")

    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)
    if "moop_healthfl" in agg:
        m = agg["moop_healthfl"]
        logger.info(f"MOOP-HealthFL: AUC={m['auc'][0]:.3f}±{m['auc'][1]:.3f}, "
                    f"Sens={m['sensitivity'][0]:.3f}±{m['sensitivity'][1]:.3f}, "
                    f"Fair={m['fairness_unified'][0]:.3f}±{m['fairness_unified'][1]:.3f}, "
                    f"ε={m['avg_epsilon']:.2f}")
    logger.info(f"Predictions: {pred_dir}")
    logger.info(f"Results:     {results_dir}")
    logger.info("\nAll done.")
    return agg


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="MOOP-HealthFL experiment runner")
    parser.add_argument("--fast", action="store_true",
                        help="Fast mode (pop=20, gen=30, eval_rounds=10, seed=42 only)")
    parser.add_argument("--n-jobs", type=int,
                        default=max(1, (os.cpu_count() or 2) - 1),
                        help="Worker processes (default: cpu_count - 1)")
    parser.add_argument("--mimic-path", type=str, default=None,
                        help="MIMIC-IV directory (needed only to build the feature cache)")
    parser.add_argument("--cache-file", type=str, default=None,
                        help=f"Feature cache (default: <output-dir>/{CACHE_NAME})")
    parser.add_argument("--output-dir", type=str, default="output",
                        help="Output directory")
    parser.add_argument("--seed", type=int, default=None,
                        help="Run a single training seed (default: all seeds)")
    parser.add_argument("--split-seed", type=int, default=42,
                        help="Seed of the patient-level train/val/test split")
    parser.add_argument("--privacy-unit", choices=["patient", "record"], default=None,
                        help="DP adjacency (default: patient)")
    parser.add_argument("--preprocessing", choices=["reference", "cohort"], default=None,
                        help="reference = public constants (default); cohort = "
                             "fitted statistics of the original submission (ablation)")
    parser.add_argument("--class-weight", choices=["prior", "pooled"], default=None,
                        help="prior = public mortality prior (default); pooled = "
                             "training labels (ablation)")
    parser.add_argument("--design-fraction", type=float, default=None,
                        help="Share of training patients reserved for budget "
                             "search (default 0.25; 0 = search on the training cohort)")
    parser.add_argument("--fairness-objective", choices=["five_level", "three_level", "fnr_gap"],
                        default=None,
                        help="Fairness objective of the search (default five_level)")
    parser.add_argument("--clip-norm", type=float, default=None,
                        help="DP clipping norm C (default: Config.CLIPPING_NORM)")
    parser.add_argument("--dp-lr", type=float, default=None,
                        help="Learning rate of certified training "
                             "(default: Config.DP_LEARNING_RATE)")
    parser.add_argument("--eval-rounds", type=int, default=None,
                        help="Federated rounds per search candidate (default: "
                             "Config.EVAL_ROUNDS; set to the full round count "
                             "for a full-fidelity search)")
    parser.add_argument("--drop-features", nargs="+", default=None,
                        choices=["ethnicity", "gender", "insurance"],
                        help="Ablation: remove these demographic indicator "
                             "groups from the model inputs")
    parser.add_argument("--methods", nargs="+", default=None,
                        help=f"Subset of methods to run (default: all of {METHOD_ORDER})")
    args = parser.parse_args()

    log_dir = os.path.join(args.output_dir, "logs")
    logger = setup_logging(log_dir)
    run_experiments(args, logger)


if __name__ == "__main__":
    main()
