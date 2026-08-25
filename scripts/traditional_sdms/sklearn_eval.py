#!/usr/bin/env python
"""
Evaluate trained sklearn SDMs — compute per-species metrics and
optionally generate prediction maps.

Mirrors maxent_eval.py but loads .pkl models with joblib instead
of calling R.

Usage
-----
    # Single-machine (evaluate all species):
    python scripts/traditional_sdms/sklearn_eval.py [--config path]

    # Parallel on SLURM (evaluate a slice):
    python scripts/traditional_sdms/sklearn_eval.py 0 499 [--config path]
    python scripts/traditional_sdms/sklearn_eval.py 500 999 [--config path]

    # After all jobs finish, merge per-job metrics:
    python scripts/traditional_sdms/merge_sklearn_results.py
"""

import os
import re
import time
import glob
import yaml
import argparse

import rootutils
import torch
import numpy as np
import pandas as pd
import h5py
import hdf5plugin  # noqa: F401  — registers the Blosc filters the data is written with
import joblib
from tqdm import tqdm

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.data_modules.datamodule import DataModule
from src.models.resnet import encode_locations
from scripts.traditional_sdms.visualize_predictions import generate_maps_for_species
from utils import get_binary_metric, _compute_metric, create_metrics_h5

root = os.getenv("PROJECT_ROOT")
from omegaconf import OmegaConf

local_default_cfg = OmegaConf.load(os.path.join(root, "configs/local_default.yaml"))
data_dir = local_default_cfg["data_dir"]
log_dir  = local_default_cfg["log_dir"]

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"


# ── helpers ──────────────────────────────────────────────────────────────────

def sklearn_predict(model, X):
    """Positive-class probability from any sklearn-like estimator.

    Accepts torch.Tensor or numpy array; always returns numpy 1-D array.
    """
    X_np = X.cpu().numpy() if isinstance(X, torch.Tensor) else np.asarray(X)
    if hasattr(model, "predict_proba"):
        p = model.predict_proba(X_np)
        return p[:, 1] if p.ndim == 2 and p.shape[1] == 2 else p.ravel()
    elif hasattr(model, "decision_function"):
        from scipy.special import expit
        return expit(model.decision_function(X_np))
    return model.predict(X_np).astype(np.float32)


def get_species_ids_from_models(model_dir, extension=".pkl"):
    """Extract species IDs from all saved model files."""
    pattern = os.path.join(model_dir, f"species_*{extension}")
    species_ids = []
    for fp in glob.glob(pattern):
        m = re.search(r"species_(\d+)" + re.escape(extension), os.path.basename(fp))
        if m:
            species_ids.append(int(m.group(1)))
    return sorted(species_ids)


def get_evaluated_species(h5_path, metric_names):
    """Return set of species IDs whose first metric is non-NaN."""
    if not os.path.exists(h5_path):
        return set()
    evaluated = set()
    with h5py.File(h5_path, "r") as hf:
        if not metric_names:
            return set()
        first = metric_names[0]
        if first not in hf:
            return set()
        data = hf[first][:]
        for sid, val in enumerate(data):
            if not np.isnan(val):
                evaluated.add(sid)
    return evaluated


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate sklearn SDMs")
    parser.add_argument(
        "start_idx", nargs="?", type=int, default=None,
        help="First species index (inclusive). Omit for all species.",
    )
    parser.add_argument(
        "end_idx", nargs="?", type=int, default=None,
        help="Last species index (inclusive). Omit for all species.",
    )
    parser.add_argument(
        "--config",
        default=os.path.join(root, "configs/model/sklearn.yaml"),
        help="Path to the sklearn YAML config file",
    )
    args = parser.parse_args()

    parallel_mode = args.start_idx is not None and args.end_idx is not None

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    model_choice   = cfg["model"].lower()
    bg_type        = cfg.get("bg_type", "random")
    skip_existing  = cfg.get("skip_existing", True)
    normalize_predictors = cfg.get("normalize_predictors", True)

    seed       = cfg.get("seed", 42)
    config_tag = cfg.get("config_tag", None)
    run_name  = f"{model_choice}_{bg_type}_s{seed}"
    if config_tag:
        run_name = f"{run_name}_{config_tag}"
    model_dir = os.path.join(log_dir, "traditional_sdms", run_name, "models")
    map_dir   = os.path.join(log_dir, "traditional_sdms", run_name, "maps")
    os.makedirs(map_dir, exist_ok=True)

    # Discover trained models
    print(f"Scanning for models in: {model_dir}")
    species_ids = get_species_ids_from_models(model_dir, extension=".pkl")
    if not species_ids:
        print(f"No .pkl model files found in {model_dir}")
        return
    print(f"Found {len(species_ids)} trained models "
          f"(range {min(species_ids)}–{max(species_ids)})")

    # Slice to the requested range when running in parallel mode
    if parallel_mode:
        species_ids = [s for s in species_ids
                       if args.start_idx <= s <= args.end_idx]
        if not species_ids:
            print(f"No models in range {args.start_idx}–{args.end_idx}. Exiting.")
            return
        print(f"Parallel mode: evaluating {len(species_ids)} species "
              f"(range {args.start_idx}–{args.end_idx})")

    # ── Data & metric config ─────────────────────────────────────────────
    with open(f"{root}/configs/data/datamodule.yaml") as f:
        dm_config = yaml.safe_load(f)["data"]["init_args"]

    with open(f"{root}/configs/model/resnet.yaml") as f:
        model_config = yaml.safe_load(f)
        use_location = (
            model_config["model"]["init_args"]["net"]["init_args"]
            .get("use_location", True)
        )

    species_df = pd.read_csv(
        os.path.join(data_dir, "targets", "species_names.csv")
    )
    num_species = len(species_df)

    metrics_conf       = model_config["model"]["init_args"]["metrics"]
    val_metrics_conf   = metrics_conf["val"]
    test_metrics_conf  = metrics_conf["test"]
    torch_val_names    = [f"val_{n}" for n in val_metrics_conf]
    torch_test_names   = [f"test_{n}" for n in test_metrics_conf]
    all_metric_names   = torch_val_names + torch_test_names

    # In parallel mode each job writes its own per-job HDF5 to avoid
    # locking issues; in single mode we write one global file.
    if parallel_mode:
        metrics_h5_path = os.path.join(
            log_dir, "traditional_sdms", run_name,
            f"metrics_by_species_{args.start_idx}_{args.end_idx}.h5",
        )
    else:
        metrics_h5_path = os.path.join(
            log_dir, "traditional_sdms", run_name, "metrics_by_species.h5"
        )

    # ── Skip already-evaluated species ───────────────────────────────────
    if skip_existing and not parallel_mode:
        already = get_evaluated_species(metrics_h5_path, all_metric_names)
        before  = len(species_ids)
        species_ids = [s for s in species_ids if s not in already]
        if before - len(species_ids):
            print(f"Skipping {before - len(species_ids)} already-evaluated species")
        if not species_ids:
            print("All species already evaluated. Exiting.")
            return
    print(f"Will evaluate {len(species_ids)} species")

    # ── DataModule ───────────────────────────────────────────────────────
    dm_config["data_dir"] = data_dir
    dm = DataModule(**dm_config)
    dm.setup(stage="fit")

    transforms = dm._make_transforms(normalize=normalize_predictors)
    dm.data_train.transforms = transforms
    dm.data_val.dataset.transforms = transforms
    dm.data_test.dataset.transforms = transforms

    # Full val/test load
    print("Loading validation and test data …")
    val_loader  = torch.utils.data.DataLoader(
        dm.data_val,  batch_size=len(dm.data_val),  shuffle=False
    )
    test_loader = torch.utils.data.DataLoader(
        dm.data_test, batch_size=len(dm.data_test), shuffle=False
    )
    val_data  = next(iter(val_loader))
    test_data = next(iter(test_loader))

    val_range_mask   = val_data.get("range_mask",  None)
    test_range_mask  = test_data.get("range_mask", None)

    val_locations    = val_data["observation_locations"]
    val_predictors   = val_data["observation_predictors"]
    val_targets_all  = val_data["observation_targets"].float()

    test_locations   = test_data["observation_locations"]
    test_predictors  = test_data["observation_predictors"]
    test_targets_all = test_data["observation_targets"].float()

    if use_location:
        val_loc_enc  = encode_locations(val_locations)
        test_loc_enc = encode_locations(test_locations)
        val_features_all  = torch.cat((val_loc_enc,  val_predictors),  dim=-1)
        test_features_all = torch.cat((test_loc_enc, test_predictors), dim=-1)
    else:
        val_features_all  = val_predictors
        test_features_all = test_predictors

    # Metric instances
    val_metrics  = {n: get_binary_metric(c) for n, c in val_metrics_conf.items()}
    test_metrics = {n: get_binary_metric(c) for n, c in test_metrics_conf.items()}

    create_metrics_h5(metrics_h5_path, num_species, all_metric_names)
    metrics_h5 = h5py.File(metrics_h5_path, "a")

    # Map configs
    continent_map_configs = (
        model_config["model"]["init_args"].get("continent_map_configs", []) or []
    )
    map_species_ids = {
        sid
        for cfg in continent_map_configs
        for sid in cfg.get("species_ids", [])
    }
    if map_species_ids:
        print(f"Will generate maps for species: {sorted(map_species_ids)}")

    # Eval timing CSV
    if parallel_mode:
        eval_time_csv = os.path.join(
            log_dir, "traditional_sdms", run_name,
            f"eval_times_{args.start_idx}_{args.end_idx}.csv",
        )
    else:
        eval_time_csv = os.path.join(
            log_dir, "traditional_sdms", run_name, "eval_times.csv",
        )
    eval_time_records = []

    # ── Evaluate loop ────────────────────────────────────────────────────
    evaluated = skipped = 0

    for species_id in tqdm(species_ids):
        species_eval_start = time.perf_counter()
        # Range masks
        if val_range_mask is not None:
            vmask = val_range_mask[:, species_id]
            tmask = test_range_mask[:, species_id]
            if vmask.dtype != torch.bool:
                vmask = vmask != 0
            if tmask.dtype != torch.bool:
                tmask = tmask != 0
        else:
            vmask = tmask = None

        val_targets  = val_targets_all[:, species_id]
        test_targets = test_targets_all[:, species_id]

        if vmask is not None:
            val_feats    = val_features_all[vmask]
            val_targets  = val_targets[vmask]
            test_feats   = test_features_all[tmask]
            test_targets = test_targets[tmask]
            test_locs    = test_locations[tmask]
        else:
            val_feats  = val_features_all
            test_feats = test_features_all
            test_locs  = test_locations

        if val_feats.shape[0] == 0 or test_feats.shape[0] == 0:
            print(f"  Species {species_id}: no in-range samples, skipping.")
            skipped += 1
            continue

        model_path = os.path.join(model_dir, f"species_{species_id}.pkl")
        if not os.path.exists(model_path):
            print(f"  Model not found: {model_path}")
            skipped += 1
            continue

        try:
            model = joblib.load(model_path)
            val_probs  = sklearn_predict(model, val_feats)
            test_probs = sklearn_predict(model, test_feats)

            val_probs_t  = torch.from_numpy(val_probs).squeeze()
            test_probs_t = torch.from_numpy(test_probs).squeeze()
            val_tgt_int  = val_targets.int()
            test_tgt_int = test_targets.int()

            for name, metric in val_metrics.items():
                metrics_h5[f"val_{name}"][species_id] = _compute_metric(
                    metric, val_probs_t, val_tgt_int
                )
            for name, metric in test_metrics.items():
                metrics_h5[f"test_{name}"][species_id] = _compute_metric(
                    metric, test_probs_t, test_tgt_int
                )
            metrics_h5.flush()
            evaluated += 1
            eval_time_records.append({
                "species_id": species_id,
                "eval_time_seconds": time.perf_counter() - species_eval_start,
            })

            # Optional map generation
            if species_id in map_species_ids:
                for map_cfg in continent_map_configs:
                    if species_id not in map_cfg.get("species_ids", []):
                        continue
                    continent = map_cfg.get("continent", "europe")
                    species_name = species_df.loc[
                        species_df["Index"] == species_id, "Species Name"
                    ].values[0]
                    print(
                        f"  Generating map for species {species_id} "
                        f"({species_name}) in {continent} …"
                    )
                    def sk_predict(feats, _m=model):
                        return sklearn_predict(_m, feats)
                    generate_maps_for_species(
                        model=None,
                        species_id=species_id,
                        species_name=species_name,
                        data_dir=data_dir,
                        output_dir=map_dir,
                        continent=continent,
                        predictor_subset=dm_config.get("predictors_subset"),
                        transforms=transforms,
                        use_location=use_location,
                        obs_locations=test_locs.numpy(),
                        obs_targets=test_tgt_int.numpy(),
                        obs_probs=test_probs_t.numpy(),
                        threshold=0.5,
                        tn_subsample=10_000,
                        mode="eval",
                        predict_func=sk_predict,
                    )

        except Exception as e:
            print(f"  Species {species_id}: evaluation failed — {e}")
            skipped += 1
            continue

    metrics_h5.close()

    # Save eval timing
    if eval_time_records:
        pd.DataFrame(eval_time_records).to_csv(eval_time_csv, index=False)
        print(f"Saved eval times to {eval_time_csv}")

    print(f"Evaluation complete: {evaluated} evaluated, {skipped} skipped")


if __name__ == "__main__":
    main()
