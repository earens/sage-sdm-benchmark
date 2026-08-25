#!/usr/bin/env python
"""
Train and evaluate sklearn-compatible SDMs (RandomForest, LightGBM, …)
per-species in a single pass.

Mirrors the maxent_train.py pipeline but stays entirely in Python —
no R subprocess needed.  Inline evaluation eliminates the need for a
separate eval SLURM job.

Usage
-----
    python scripts/traditional_sdms/sklearn_train.py <start_idx> <end_idx> [--config path]

The optional ``--config`` flag lets you point to a different YAML
(default: configs/model/sklearn.yaml).
"""

import os
import time
import yaml
import argparse

import rootutils
import torch
import numpy as np
import h5py
import hdf5plugin  # noqa: F401  — registers the Blosc filters the data is written with
import joblib
import pandas as pd
from tqdm import tqdm
from omegaconf import OmegaConf

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from utils import set_global_seed, get_binary_metric, _compute_metric, create_metrics_h5
from src.data_modules.datamodule import DataModule
from src.models.resnet import encode_locations
from scripts.traditional_sdms.maxent_train import (
    get_data,
    prepare_train_data,
    WEIGHTED_MODELS,
)

# Maps each model to the hyperparameter that is selected per species from the
# 3-value grid (see per_species_param_file handling in main()).
SELECT_PARAM = {
    "glm": "C",
    "gam": "lam",
    "randomforest": "max_features",
    "brt": "tc",
}

# ── paths ────────────────────────────────────────────────────────────────────
root = os.getenv("PROJECT_ROOT")
local_default_cfg = OmegaConf.load(os.path.join(root, "configs/local_default.yaml"))
data_dir = local_default_cfg["data_dir"]
log_dir  = local_default_cfg["log_dir"]

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"


# ── model registry ───────────────────────────────────────────────────────────
MODEL_REGISTRY = {}


def register_model(name):
    """Decorator that adds a builder function to the registry."""
    def _decorator(fn):
        MODEL_REGISTRY[name] = fn
        return fn
    return _decorator


@register_model("randomforest")
def _build_random_forest(seed, num_workers, **kw):
    """Down-sampled Random Forest (Valavi et al. 2021).

    Each tree draws a class-balanced bootstrap from the 10k background pool
    (all presences + an equal-sized random background subsample); the ensemble
    averages over the full background. ``max_features`` (mtry) is the tuned knob.
    """
    from imblearn.ensemble import BalancedRandomForestClassifier
    defaults = dict(
        n_estimators=1000,
        max_features="sqrt",
        sampling_strategy="all",
        replacement=False,
        bootstrap=True,
        n_jobs=num_workers,
        random_state=seed,
    )
    defaults.update(kw)
    return BalancedRandomForestClassifier(**defaults)


@register_model("gradientboosting")
def _build_gradient_boosting(seed, num_workers, **kw):
    from sklearn.ensemble import GradientBoostingClassifier
    defaults = dict(
        n_estimators=500,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        random_state=seed,
    )
    defaults.update(kw)
    return GradientBoostingClassifier(**defaults)


@register_model("lightgbm")
def _build_lightgbm(seed, num_workers, **kw):
    import lightgbm as lgb
    defaults = dict(
        objective="binary",
        metric="auc",
        learning_rate=0.05,
        num_leaves=31,
        n_estimators=500,
        n_jobs=num_workers,
        random_state=seed,
        verbose=-1,
    )
    defaults.update(kw)
    return lgb.LGBMClassifier(**defaults)


@register_model("xgboost")
def _build_xgboost(seed, num_workers, **kw):
    from xgboost import XGBClassifier
    defaults = dict(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        n_jobs=num_workers,
        random_state=seed,
        eval_metric="logloss",
    )
    defaults.update(kw)
    return XGBClassifier(**defaults)


@register_model("logistic")
@register_model("linear")
def _build_logistic(seed, num_workers, **kw):
    from sklearn.linear_model import LogisticRegression
    defaults = dict(
        max_iter=1000,
        solver="lbfgs",
        n_jobs=num_workers,
        random_state=seed,
    )
    defaults.update(kw)
    return LogisticRegression(**defaults)


def _linear_plus_squares(X):
    """Append per-variable squares: [x] -> [x, x**2]. No cross terms (à la Valavi).

    Module-level (not a lambda) so the fitted Pipeline stays picklable.
    """
    return np.hstack([X, X ** 2])


@register_model("glm")
def _build_glm(seed, num_workers, **kw):
    """GLM à la Valavi: lasso logistic regression on linear + per-variable
    quadratic features (squares only, no cross terms), standardized."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import FunctionTransformer, StandardScaler
    from sklearn.pipeline import Pipeline

    lr_defaults = dict(
        max_iter=1000,
        solver="saga",     # required for L1
        penalty="l1",
        C=1.0,
        random_state=seed,
    )
    lr_defaults.update(kw)

    return Pipeline([
        ("squares", FunctionTransformer(_linear_plus_squares)),
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(**lr_defaults)),
    ])


class _BRTClassifier:
    """LightGBM configured to emulate classic BRT (Elith 2008 / R gbm).

    ``num_leaves = tc + 1`` matches gbm's ``interaction.depth`` (tc splits per
    tree). The number of trees is chosen by early stopping on an internal
    stratified validation split (≈ gbm.step's CV-based n.trees selection).
    Sample weights (IPP down-weighting) flow through to the booster.
    """

    def __init__(self, tc=3, learning_rate=0.01, subsample=0.75,
                 n_estimators=10_000, validation_fraction=0.2,
                 early_stopping_rounds=50, num_workers=4, seed=0):
        self.tc = int(tc)
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.n_estimators = n_estimators
        self.validation_fraction = validation_fraction
        self.early_stopping_rounds = early_stopping_rounds
        self.num_workers = num_workers
        self.seed = seed
        self._model = None

    def _make_model(self):
        import lightgbm as lgb
        return lgb.LGBMClassifier(
            objective="binary",
            num_leaves=self.tc + 1,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            subsample_freq=1,
            n_estimators=self.n_estimators,
            n_jobs=self.num_workers,
            random_state=self.seed,
            verbose=-1,
        )

    def fit(self, X, y, sample_weight=None):
        import lightgbm as lgb
        from sklearn.model_selection import train_test_split

        self._model = self._make_model()
        y = np.asarray(y)

        can_split = (
            len(np.unique(y)) > 1 and np.min(np.bincount(y.astype(int))) >= 2
        )
        if can_split:
            if sample_weight is None:
                X_tr, X_val, y_tr, y_val = train_test_split(
                    X, y, test_size=self.validation_fraction,
                    stratify=y, random_state=self.seed,
                )
                w_tr = w_val = None
            else:
                X_tr, X_val, y_tr, y_val, w_tr, w_val = train_test_split(
                    X, y, sample_weight, test_size=self.validation_fraction,
                    stratify=y, random_state=self.seed,
                )
            self._model.fit(
                X_tr, y_tr, sample_weight=w_tr,
                eval_set=[(X_val, y_val)],
                eval_sample_weight=None if w_val is None else [w_val],
                callbacks=[lgb.early_stopping(self.early_stopping_rounds),
                           lgb.log_evaluation(0)],
            )
        else:
            # Too few samples to early-stop; fit a fixed, smaller ensemble.
            self._model.set_params(n_estimators=1000)
            self._model.fit(X, y, sample_weight=sample_weight)
        return self

    @property
    def best_iteration_(self):
        return getattr(self._model, "best_iteration_", None)

    def predict_proba(self, X):
        return self._model.predict_proba(X)

    def predict(self, X):
        return self._model.predict(X)


@register_model("brt")
def _build_brt(seed, num_workers, **kw):
    """Gradient-boosted trees (LightGBM) emulating Elith-2008 BRT."""
    return _BRTClassifier(
        tc=kw.pop("tc", 3),
        learning_rate=kw.pop("learning_rate", 0.01),
        subsample=kw.pop("subsample", 0.75),
        n_estimators=kw.pop("n_estimators", 10_000),
        num_workers=num_workers,
        seed=seed,
    )


class _GAMClassifier:
    """Thin wrapper around pyGAM LogisticGAM with an sklearn-compatible interface.

    One additive cubic P-spline per feature; ``lam`` (smoothing) is the tuned
    knob. ``fit`` accepts sample weights, mapped to pyGAM's ``weights=``.
    """

    def __init__(self, n_splines=10, lam=0.6, max_iter=100, **kw):
        self.n_splines = n_splines
        self.lam = lam
        self.max_iter = max_iter
        self.kw = kw
        self._model = None

    def fit(self, X, y, sample_weight=None):
        from pygam import LogisticGAM, s
        n_features = X.shape[1]
        terms = s(0, n_splines=self.n_splines)
        for i in range(1, n_features):
            terms += s(i, n_splines=self.n_splines)
        self._model = LogisticGAM(terms, lam=self.lam, max_iter=self.max_iter, **self.kw)
        self._model.fit(X, y, weights=sample_weight)
        return self

    def predict_proba(self, X):
        p = self._model.predict_proba(X)
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return self._model.predict(X)


@register_model("gam")
def _build_gam(seed, num_workers, **kw):
    """Additive GAM with per-feature cubic P-splines via pyGAM."""
    defaults = dict(n_splines=10, lam=0.6, max_iter=100)
    defaults.update(kw)
    return _GAMClassifier(**defaults)


def get_model(name, seed, num_workers, **extra_params):
    name = name.lower()
    if name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{name}'. Available: {list(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[name](seed=seed, num_workers=num_workers, **extra_params)


# ── prediction helper ────────────────────────────────────────────────────────
def predict_proba(model, X):
    """Return positive-class probability from any sklearn-compatible estimator."""
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)
        return probs[:, 1] if probs.ndim == 2 and probs.shape[1] == 2 else probs.ravel()
    elif hasattr(model, "decision_function"):
        from scipy.special import expit
        return expit(model.decision_function(X))
    else:
        return model.predict(X).astype(np.float32)


# ── fit helper ────────────────────────────────────────────────────────────────
def fit_with_weights(model, X, y, sample_weight):
    """Fit, routing optional sample weights to the right place.

    For a ``Pipeline`` the weight must be addressed to the final step
    (e.g. ``lr__sample_weight``); bare estimators / wrappers take ``sample_weight``
    directly. ``_GAMClassifier``/``_BRTClassifier`` accept ``sample_weight`` and
    forward it to their underlying library.
    """
    from sklearn.pipeline import Pipeline
    if sample_weight is None:
        model.fit(X, y)
        return
    if isinstance(model, Pipeline):
        final = model.steps[-1][0]
        model.fit(X, y, **{f"{final}__sample_weight": sample_weight})
    else:
        model.fit(X, y, sample_weight=sample_weight)


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Train & evaluate sklearn SDMs per species")
    parser.add_argument("start_idx", type=int)
    parser.add_argument("end_idx", type=int)
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join(root, "configs/model/sklearn.yaml"),
        help="Path to the sklearn YAML config file",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Override the data directory (default: the data_dir from "
             "configs/local_default.yaml). Must contain targets/ and predictors/. "
             "Used e.g. to point at the bundled examples/demo_data/.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Retrain species even if their model file already exists "
             "(overrides skip_existing from the config).",
    )
    args = parser.parse_args()

    start_idx = args.start_idx
    end_idx = args.end_idx

    # Allow overriding the module-level data_dir (read from local_default.yaml)
    # with a CLI value, e.g. the shippable demo subset.
    global data_dir
    data_tag = None
    if args.data_dir is not None:
        data_dir = os.path.abspath(args.data_dir)
        if not data_dir.endswith(os.sep):
            data_dir += os.sep
        # Tag the run with the data-source folder so a demo/subset run can't
        # overwrite full-data outputs that share the same model / bg / seed.
        data_tag = os.path.basename(data_dir.rstrip(os.sep))

    # ── Load config ──────────────────────────────────────────────────────
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    model_choice = cfg["model"].lower()
    model_params = cfg.get("model_params", {}) or {}
    species_ids = cfg["species_ids"]
    bg_type = cfg.get("bg_type", "random")
    seed = cfg.get("seed", 42)
    train_cap = cfg.get("train_cap", 10_000)
    skip_existing = cfg.get("skip_existing", False)
    if args.force:
        skip_existing = False
    normalize_predictors = cfg.get("normalize_predictors", True)
    num_workers = cfg.get("num_workers", 4)
    # Optional tag distinguishing reference runs that share model/bg/seed but vary
    # the swept hyperparameter (so the 3 grid runs don't overwrite each other).
    config_tag = cfg.get("config_tag", None)
    # Optional CSV (species_id, param) freezing the per-species selected
    # hyperparameter for the final 5-seed runs.
    per_species_param_file = cfg.get("per_species_param_file", None)

    run_name = f"{model_choice}_{bg_type}_s{seed}"
    if data_tag:
        run_name = f"{run_name}_{data_tag}"
    if config_tag:
        run_name = f"{run_name}_{config_tag}"
    out_dir   = os.path.join(log_dir, "traditional_sdms", run_name, "predictions")
    model_dir = os.path.join(log_dir, "traditional_sdms", run_name, "models")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    set_global_seed(seed)

    # ── Data-module config ───────────────────────────────────────────────
    with open(f"{root}/configs/data/datamodule.yaml") as f:
        dm_config = yaml.safe_load(f)["data"]["init_args"]
        SUFFIX = "_1km" if dm_config.get("aggregated", True) else ""

    # Auto-disable range masking / occurrence subsampling when the corresponding
    # files are absent from data_dir (e.g. the bundled demo subset ships neither).
    _sub_file = dm_config.get("subsample_indices_file")
    if _sub_file and not os.path.exists(os.path.join(data_dir, "targets", _sub_file)):
        dm_config["subsample_indices_file"] = None
    _po_mask_file = os.path.join(data_dir, f"range_masks{SUFFIX}", "train_range_mask.h5")
    _pa_mask_file = os.path.join(data_dir, "range_masks_1km", "eval_range_mask.h5")
    if not os.path.exists(_po_mask_file):
        dm_config["po_range_mask"] = False
    if not os.path.exists(_pa_mask_file):
        dm_config["pa_range_mask"] = False

    # resnet.yaml is still read for the metric definitions below, but
    # location encoding for the standard SDMs is controlled by the traditional
    # config (environment-only by default), decoupled from the DL model config.
    with open(f"{root}/configs/model/resnet.yaml") as f:
        model_config = yaml.safe_load(f)
    use_location = cfg.get("use_location", False)

    species_df = pd.read_csv(
        os.path.join(data_dir, "targets", "species_names.csv")
    )
    num_species = len(species_df)

    if species_ids == "all":
        species_ids = list(range(num_species))
    else:
        species_ids = [int(sid) for sid in species_ids]

    species_ids = species_ids[start_idx : end_idx + 1]

    # ── Skip existing ────────────────────────────────────────────────────
    if skip_existing:
        original_count = len(species_ids)
        species_ids = [
            sid for sid in species_ids
            if not os.path.exists(os.path.join(model_dir, f"species_{sid}.pkl"))
        ]
        skipped = original_count - len(species_ids)
        if skipped:
            print(f"Skipping {skipped} species with existing models")
        if not species_ids:
            print("All species in this range already trained. Exiting.")
            return

    # ── Initialize DataModule ────────────────────────────────────────────
    dm_config["data_dir"] = data_dir
    print("Initializing data module …")
    dm = DataModule(**dm_config)
    dm.setup(stage="fit")

    transforms = dm._make_transforms(normalize=normalize_predictors)
    dm.data_train.transforms = transforms
    dm.data_val.dataset.transforms = transforms
    dm.data_test.dataset.transforms = transforms

    # ── Load val / test once (with range masks for inline eval) ──────────
    print("Loading val/test data")
    _val_loader  = torch.utils.data.DataLoader(
        dm.data_val, batch_size=len(dm.data_val), shuffle=False, num_workers=0,
    )
    _test_loader = torch.utils.data.DataLoader(
        dm.data_test, batch_size=len(dm.data_test), shuffle=False, num_workers=0,
    )
    val_data_full  = next(iter(_val_loader))
    test_data_full = next(iter(_test_loader))

    val_range_mask   = val_data_full.get("range_mask", None)
    test_range_mask  = test_data_full.get("range_mask", None)
    val_targets_all  = val_data_full["observation_targets"].float()
    test_targets_all = test_data_full["observation_targets"].float()

    num_val  = val_targets_all.shape[0]
    num_test = test_targets_all.shape[0]

    # Pre-compute feature matrices (torch → numpy, done once)
    if use_location:
        val_loc_enc  = encode_locations(val_data_full["observation_locations"])
        test_loc_enc = encode_locations(test_data_full["observation_locations"])
        val_X_np = torch.cat(
            (val_loc_enc, val_data_full["observation_predictors"]), dim=-1,
        ).cpu().numpy()
        test_X_np = torch.cat(
            (test_loc_enc, test_data_full["observation_predictors"]), dim=-1,
        ).cpu().numpy()
    else:
        val_X_np  = val_data_full["observation_predictors"].cpu().numpy()
        test_X_np = test_data_full["observation_predictors"].cpu().numpy()

    # Free intermediate tensors — keep only numpy features, range masks, targets
    del val_data_full, test_data_full, _val_loader, _test_loader

    # ── Per-job HDF5 for predictions ─────────────────────────────────────
    h5_path = os.path.join(
        out_dir, f"predictions_by_species_{start_idx}_{end_idx}.h5"
    )
    num_species_in_job = end_idx - start_idx + 1

    if not os.path.exists(h5_path):
        with h5py.File(h5_path, "w") as hf:
            hf.create_dataset(
                "val",  shape=(num_species_in_job, num_val),
                dtype="float32", fillvalue=np.nan,
            )
            hf.create_dataset(
                "test", shape=(num_species_in_job, num_test),
                dtype="float32", fillvalue=np.nan,
            )
            hf.attrs["start_idx"] = start_idx
            hf.attrs["end_idx"]   = end_idx

    # ── Metrics setup for inline evaluation ──────────────────────────────
    metrics_conf      = model_config["model"]["init_args"]["metrics"]
    val_metrics_conf  = metrics_conf["val"]
    test_metrics_conf = metrics_conf["test"]
    val_metric_names  = [f"val_{n}" for n in val_metrics_conf]
    test_metric_names = [f"test_{n}" for n in test_metrics_conf]
    all_metric_names  = val_metric_names + test_metric_names

    val_metrics  = {n: get_binary_metric(c) for n, c in val_metrics_conf.items()}
    test_metrics = {n: get_binary_metric(c) for n, c in test_metrics_conf.items()}

    metrics_h5_path = os.path.join(
        log_dir, "traditional_sdms", run_name,
        f"metrics_by_species_{start_idx}_{end_idx}.h5",
    )
    create_metrics_h5(metrics_h5_path, num_species, all_metric_names)

    # ── Training-time CSV ────────────────────────────────────────────────
    time_csv = os.path.join(
        log_dir, "traditional_sdms", run_name,
        f"training_times_{start_idx}_{end_idx}.csv",
    )
    if not os.path.exists(time_csv):
        pd.DataFrame({
            "species_id":         species_ids,
            "train_time_seconds": [np.nan] * len(species_ids),
            "eval_time_seconds":  [np.nan] * len(species_ids),
            "n_samples":          [-1]     * len(species_ids),
            "n_presences":        [-1]     * len(species_ids),
            "n_features":         [-1]     * len(species_ids),
        }).to_csv(time_csv, index=False)

    # ── Per-species frozen hyperparameter (final 5-seed runs) ────────────
    species_param_map = {}
    if per_species_param_file:
        pmap_df = pd.read_csv(per_species_param_file)
        species_param_map = dict(zip(pmap_df["species_id"], pmap_df["param"]))
        print(
            f"Loaded per-species '{SELECT_PARAM[model_choice]}' for "
            f"{len(species_param_map)} species from {per_species_param_file}"
        )

    # max_features (RF mtry) and tc (BRT) are integer-valued; others are floats.
    _int_param = SELECT_PARAM.get(model_choice) in {"max_features", "tc"}

    # ── Train + eval loop ────────────────────────────────────────────────
    with h5py.File(h5_path, "a") as hf, \
         h5py.File(metrics_h5_path, "a") as mhf:
        for idx, species_id in enumerate(tqdm(species_ids)):
            model_path = os.path.join(model_dir, f"species_{species_id}.pkl")
            if skip_existing and os.path.exists(model_path):
                print(f"Model exists for species {species_id}, skipping.")
                continue

            # Load training batch
            species_start_time = time.perf_counter()
            try:
                train_batch = get_data(
                    dm, species_id, num_workers=0, cap=train_cap,
                    split="train", model_choice=model_choice,
                    bg_type=bg_type, suffix=SUFFIX, seed=seed,
                )
            except ValueError as e:
                print(f"Skipping species {species_id}: {e}")
                continue

            X_train, y_train = prepare_train_data(train_batch, use_location)

            # Deterministic shuffle
            perm = torch.randperm(
                X_train.size(0),
                generator=torch.Generator().manual_seed(seed),
            )
            X_train = X_train[perm].cpu().numpy()
            y_train = y_train[perm].cpu().numpy().astype(int)

            n_pos = int(y_train.sum())
            n_bg = len(y_train) - n_pos
            print(
                f"Species {species_id}: N={len(X_train)}, pos={n_pos}, "
                f"features={X_train.shape[1]}"
            )

            # Per-species frozen hyperparameter (final runs); else global grid value.
            species_params = dict(model_params)
            if species_id in species_param_map:
                val = species_param_map[species_id]
                val = int(val) if _int_param else float(val)
                species_params[SELECT_PARAM[model_choice]] = val

            # IPP down-weighting for GLM/GAM/BRT: equalise summed presence/bg weight.
            if model_choice in WEIGHTED_MODELS and n_pos > 0 and n_bg > 0:
                sample_weight = np.ones(len(y_train), dtype=np.float64)
                sample_weight[y_train == 0] = n_pos / n_bg
            else:
                sample_weight = None

            # Build & fit
            model = get_model(
                model_choice, seed=seed, num_workers=num_workers, **species_params
            )
            t0 = time.time()
            try:
                fit_with_weights(model, X_train, y_train, sample_weight)
            except Exception as e:
                print(f"Training failed for species {species_id}: {e}")
                continue
            elapsed = time.time() - t0
            print(f"  trained in {elapsed:.1f}s")

            joblib.dump(model, model_path)

            # Predict on val / test (full, unmasked)
            val_preds  = predict_proba(model, val_X_np)
            test_preds = predict_proba(model, test_X_np)

            hf["val"][idx, :]  = val_preds
            hf["test"][idx, :] = test_preds

            # ── Inline evaluation (range-masked) ─────────────────────
            eval_start = time.perf_counter()

            if val_range_mask is not None:
                vmask = val_range_mask[:, species_id]
                tmask = test_range_mask[:, species_id]
                if vmask.dtype != torch.bool:
                    vmask = vmask != 0
                if tmask.dtype != torch.bool:
                    tmask = tmask != 0
            else:
                vmask = tmask = None

            val_tgt  = val_targets_all[:, species_id]
            test_tgt = test_targets_all[:, species_id]

            val_preds_t  = torch.from_numpy(val_preds).float()
            test_preds_t = torch.from_numpy(test_preds).float()

            if vmask is not None:
                val_preds_masked  = val_preds_t[vmask]
                val_tgt_masked    = val_tgt[vmask].int()
                test_preds_masked = test_preds_t[tmask]
                test_tgt_masked   = test_tgt[tmask].int()
            else:
                val_preds_masked  = val_preds_t
                val_tgt_masked    = val_tgt.int()
                test_preds_masked = test_preds_t
                test_tgt_masked   = test_tgt.int()

            if val_preds_masked.shape[0] > 0 and test_preds_masked.shape[0] > 0:
                for name, metric in val_metrics.items():
                    mhf[f"val_{name}"][species_id] = _compute_metric(
                        metric, val_preds_masked, val_tgt_masked,
                    )
                for name, metric in test_metrics.items():
                    mhf[f"test_{name}"][species_id] = _compute_metric(
                        metric, test_preds_masked, test_tgt_masked,
                    )
                mhf.flush()

            eval_elapsed = time.perf_counter() - eval_start

            # Update time log
            total_species_time = time.perf_counter() - species_start_time
            try:
                df = pd.read_csv(time_csv)
                row = df.index[df["species_id"] == species_id]
                if len(row):
                    df.loc[row[0], "train_time_seconds"] = total_species_time
                    df.loc[row[0], "eval_time_seconds"]  = eval_elapsed
                    df.loc[row[0], "n_samples"]          = len(X_train)
                    df.loc[row[0], "n_presences"]        = n_pos
                    df.loc[row[0], "n_features"]         = X_train.shape[1]
                    df.to_csv(time_csv, index=False)
            except Exception:
                pass  # non-critical

    print("Done.")


if __name__ == "__main__":
    main()
