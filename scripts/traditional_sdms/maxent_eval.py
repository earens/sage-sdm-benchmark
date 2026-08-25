import os
import time
import argparse
import yaml
import rootutils
import torch
import numpy as np
import pandas as pd
from omegaconf import OmegaConf
import h5py
import hdf5plugin  # noqa: F401  — registers the Blosc filters the data is written with
from tqdm import tqdm
import tempfile
import subprocess
import shutil
import glob
import re

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from utils import get_binary_metric, _compute_metric, create_metrics_h5
from scripts.traditional_sdms.visualize_predictions import generate_maps_for_species
from src.models.resnet import encode_locations
from src.data_modules.datamodule import DataModule

root = os.getenv("PROJECT_ROOT")
local_default_cfg = OmegaConf.load(os.path.join(root, "configs/local_default.yaml"))
data_dir = local_default_cfg["data_dir"]
log_dir = local_default_cfg["log_dir"]

def get_species_ids_from_models(model_dir):
    """Extract species IDs from all .rds files in model directory."""
    pattern = os.path.join(model_dir, "species_*.rds")
    model_files = glob.glob(pattern)

    species_ids = []
    for filepath in model_files:
        filename = os.path.basename(filepath)
        # Extract number from "species_123.rds"
        match = re.search(r'species_(\d+)\.rds', filename)
        if match:
            species_id = int(match.group(1))
            species_ids.append(species_id)

    return sorted(species_ids)

def get_evaluated_species(h5_path, metric_names):
    """Get species IDs that already have metrics computed (non-NaN)."""
    if not os.path.exists(h5_path):
        return set()

    evaluated = set()
    with h5py.File(h5_path, "r") as h5f:
        if len(metric_names) == 0:
            return set()

        # Check the first metric to determine which species are evaluated
        first_metric = metric_names[0]
        if first_metric not in h5f:
            return set()

        data = h5f[first_metric][:]
        for species_id, value in enumerate(data):
            if not np.isnan(value):
                evaluated.add(species_id)

    return evaluated

def predict_with_maxent_r(model_path, X_val, X_test):
    """Call R to make predictions using saved MaxNet model"""

    # Create temp directory but don't auto-delete
    tmp_dir = tempfile.mkdtemp()

    try:
        input_h5 = os.path.join(tmp_dir, "predict_input.h5")
        output_h5 = os.path.join(tmp_dir, "predict_output.h5")

        # Write data
        with h5py.File(input_h5, "w") as f:
            f.create_dataset("X_val", data=X_val.cpu().numpy().astype(np.float32))
            f.create_dataset("X_test", data=X_test.cpu().numpy().astype(np.float32))

        # Call R script for prediction
        r_script = os.path.join(os.path.dirname(__file__), "maxent_eval.R")
        cmd = ["Rscript", r_script, input_h5, output_h5, model_path]
        subprocess.run(cmd, check=True, capture_output=True, text=True)

        # Read predictions
        with h5py.File(output_h5, "r") as f:
            val_preds = f["val"][:]
            test_preds = f["test"][:]

    except subprocess.CalledProcessError as e:
        # Re-raise with R's stderr for better debugging
        error_msg = f"R script failed:\nstdout: {e.stdout}\nstderr: {e.stderr}"
        raise RuntimeError(error_msg) from e

    finally:
        # Clean up temp directory
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return val_preds, test_preds


def predict_grid_with_maxent_r(model_path, grid_features):
    """Call R to make predictions for a single feature matrix (e.g., continental grid).

    Re-uses the existing maxent_eval.R script by passing grid_features as
    ``X_val`` and a tiny 1-row dummy as ``X_test``.

    Parameters
    ----------
    model_path : str
        Path to the saved .rds MaxNet model.
    grid_features : numpy array or torch Tensor
        Feature matrix of shape (N, D).

    Returns
    -------
    predictions : numpy array of shape (N,)
    """
    if isinstance(grid_features, torch.Tensor):
        grid_features = grid_features.cpu().numpy()

    tmp_dir = tempfile.mkdtemp()
    try:
        input_h5 = os.path.join(tmp_dir, "grid_input.h5")
        output_h5 = os.path.join(tmp_dir, "grid_output.h5")

        with h5py.File(input_h5, "w") as f:
            f.create_dataset("X_val", data=grid_features.astype(np.float32))
            # Minimal dummy so R script doesn't fail on missing X_test
            f.create_dataset("X_test", data=grid_features[:1].astype(np.float32))

        r_script = os.path.join(os.path.dirname(__file__), "maxent_eval.R")
        cmd = ["Rscript", r_script, input_h5, output_h5, model_path]
        subprocess.run(cmd, check=True, capture_output=True, text=True)

        with h5py.File(output_h5, "r") as f:
            predictions = f["val"][:]

    except subprocess.CalledProcessError as e:
        error_msg = f"R grid prediction failed:\nstdout: {e.stdout}\nstderr: {e.stderr}"
        raise RuntimeError(error_msg) from e

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return predictions


def main():
    parser = argparse.ArgumentParser(description="Evaluate MaxEnt SDMs")
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
        default=os.path.join(root, "configs/model/maxent.yaml"),
        help="Path to the MaxEnt YAML config file",
    )
    args = parser.parse_args()

    parallel_mode = args.start_idx is not None and args.end_idx is not None

    with open(args.config) as stream:
        eval_config = yaml.safe_load(stream)

    model_choice = eval_config["model"].lower()
    bg_type = eval_config.get("bg_type", "random")
    skip_existing = eval_config.get("skip_existing", True)  # Use same flag as training
    normalize_predictors = eval_config.get("normalize_predictors", False)

    seed = eval_config.get("seed", 42)
    config_tag = eval_config.get("config_tag", None)
    run_name = f"{model_choice}_{bg_type}_s{seed}"
    if config_tag:
        run_name = f"{run_name}_{config_tag}"
    model_dir = os.path.join(log_dir, "traditional_sdms", run_name, "models")
    map_dir = os.path.join(log_dir, "traditional_sdms", run_name, "maps")
    os.makedirs(map_dir, exist_ok=True)

    # Get species IDs from model files instead of config
    print(f"Scanning for models in: {model_dir}")
    species_ids = get_species_ids_from_models(model_dir)

    if not species_ids:
        print(f"No model files found in {model_dir}")
        return

    print(f"Found {len(species_ids)} trained models")
    print(f"Species ID range: {min(species_ids)} to {max(species_ids)}")

    # Slice to the requested range when running in parallel mode
    if parallel_mode:
        species_ids = [s for s in species_ids
                       if args.start_idx <= s <= args.end_idx]
        if not species_ids:
            print(f"No models in range {args.start_idx}\u2013{args.end_idx}. Exiting.")
            return
        print(f"Parallel mode: evaluating {len(species_ids)} species "
              f"(range {args.start_idx}\u2013{args.end_idx})")

    with open(f"{root}/configs/data/datamodule.yaml") as stream:
        dm_config = yaml.safe_load(stream)["data"]["init_args"]

    with open(f"{root}/configs/model/resnet.yaml") as stream:
        model_config = yaml.safe_load(stream)
        use_location = model_config["model"]["init_args"]["net"]["init_args"].get("use_location", True)

    # Read num_species from canonical species list
    with open(f"{root}/configs/local_default.yaml") as stream:
        _data_dir = yaml.safe_load(stream)["data_dir"]
    species_df = pd.read_csv(os.path.join(_data_dir, "targets", "species_names.csv"))
    num_species = len(species_df)

    # Get metric names early for skip_existing check
    metrics_conf = model_config["model"]["init_args"]["metrics"]
    val_metrics_conf = metrics_conf["val"]
    test_metrics_conf = metrics_conf["test"]
    torch_val_metric_names = [f"val_{name}" for name in val_metrics_conf.keys()]
    torch_test_metric_names = [f"test_{name}" for name in test_metrics_conf.keys()]
    all_metric_names = torch_val_metric_names + torch_test_metric_names

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

    # Filter out already evaluated species if skip_existing is True
    if skip_existing and not parallel_mode:
        already_evaluated = get_evaluated_species(metrics_h5_path, all_metric_names)
        original_count = len(species_ids)
        species_ids = [sid for sid in species_ids if sid not in already_evaluated]
        skipped_count = original_count - len(species_ids)
        if skipped_count > 0:
            print(f"Skipping {skipped_count} species with existing metrics")

        if len(species_ids) == 0:
            print("All species already evaluated. Exiting.")
            return

    print(f"Will evaluate {len(species_ids)} species")

    dm_config["data_dir"] = data_dir
    dm = DataModule(**dm_config)
    dm.setup(stage="fit")

    # Match the normalization setting used during training.
    nan_only_transform = dm._make_transforms(normalize=normalize_predictors)
    dm.data_train.transforms = nan_only_transform
    dm.data_val.dataset.transforms = nan_only_transform
    dm.data_test.dataset.transforms = nan_only_transform

    val_batch = dm.data_val
    test_batch = dm.data_test

    print("Loading validation and test data...")
    val_data = next(iter(torch.utils.data.DataLoader(val_batch, batch_size=len(val_batch), shuffle=False)))
    test_data = next(iter(torch.utils.data.DataLoader(test_batch, batch_size=len(test_batch), shuffle=False)))
    val_range_mask = val_data.get("range_mask", None)
    test_range_mask = test_data.get("range_mask", None)

    val_locations = val_data["observation_locations"]
    val_predictors = val_data["observation_predictors"]
    val_targets_all = val_data["observation_targets"]

    test_locations = test_data["observation_locations"]
    test_predictors = test_data["observation_predictors"]
    test_targets_all = test_data["observation_targets"]

    if use_location:
        val_locations_enc = encode_locations(val_locations)
        test_locations_enc = encode_locations(test_locations)
        val_features_all = torch.cat((val_locations_enc, val_predictors), dim=-1)
        test_features_all = torch.cat((test_locations_enc, test_predictors), dim=-1)
    else:
        val_features_all = val_predictors
        test_features_all = test_predictors

    # Create metric instances
    val_metrics = {name: get_binary_metric(conf) for name, conf in val_metrics_conf.items()}
    test_metrics = {name: get_binary_metric(conf) for name, conf in test_metrics_conf.items()}

    create_metrics_h5(metrics_h5_path, num_species, all_metric_names)
    metrics_h5 = h5py.File(metrics_h5_path, "a")

    # Load continent_map_configs for map generation
    continent_map_configs = model_config["model"]["init_args"].get("continent_map_configs", []) or []
    map_species_ids = set()
    for cfg in continent_map_configs:
        for sid in cfg.get("species_ids", []):
            map_species_ids.add(sid)

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

    print(f"Starting evaluation for {len(species_ids)} species...")
    if map_species_ids:
        print(f"Will generate maps for species: {sorted(map_species_ids)}")
    skipped = 0
    evaluated = 0

    for species_id in tqdm(species_ids):
        species_eval_start = time.perf_counter()
        species_range_mask_val = val_range_mask[:, species_id] if val_range_mask is not None else None
        species_range_mask_test = test_range_mask[:, species_id] if test_range_mask is not None else None
        if species_range_mask_val is not None:
            if species_range_mask_val.dtype != torch.bool:
                species_range_mask_val = species_range_mask_val != 0
            if species_range_mask_test.dtype != torch.bool:
                species_range_mask_test = species_range_mask_test != 0

        val_targets = val_targets_all[:, species_id].float()
        test_targets = test_targets_all[:, species_id].float()

        if species_range_mask_val is not None:
            val_features = val_features_all[species_range_mask_val]
            val_targets = val_targets[species_range_mask_val]
            test_features = test_features_all[species_range_mask_test]
            test_targets = test_targets[species_range_mask_test]
            test_locs = test_locations[species_range_mask_test]
        else:
            val_features = val_features_all
            test_features = test_features_all
            test_locs = test_locations

        if val_features.shape[0] == 0 or test_features.shape[0] == 0:
            print(f"  Species {species_id}: no in-range samples, skipping.")
            skipped += 1
            continue

        model_path = os.path.join(model_dir, f"species_{species_id}.rds")
        if not os.path.exists(model_path):
            print(f"  Model not found: {model_path}")
            skipped += 1
            continue

        try:
            # Use R for predictions
            val_probs, test_probs = predict_with_maxent_r(model_path, val_features, test_features)

            val_probs_tensor = torch.from_numpy(val_probs).squeeze()
            test_probs_tensor = torch.from_numpy(test_probs).squeeze()
            val_targets_tensor = val_targets.int()
            test_targets_tensor = test_targets.int()

            torch_val_results = {}
            for name, metric in val_metrics.items():
                torch_val_results[f"val_{name}"] = _compute_metric(metric, val_probs_tensor, val_targets_tensor)

            torch_test_results = {}
            for name, metric in test_metrics.items():
                torch_test_results[f"test_{name}"] = _compute_metric(metric, test_probs_tensor, test_targets_tensor)

            for k, v in torch_val_results.items():
                metrics_h5[k][species_id] = v
            for k, v in torch_test_results.items():
                metrics_h5[k][species_id] = v
            metrics_h5.flush()

            evaluated += 1
            eval_time_records.append({
                "species_id": species_id,
                "eval_time_seconds": time.perf_counter() - species_eval_start,
            })

            # Generate maps for species of interest
            if species_id in map_species_ids:
                for map_cfg in continent_map_configs:
                    if species_id not in map_cfg.get("species_ids", []):
                        continue
                    continent = map_cfg.get("continent", "europe")
                    species_name = species_df.loc[species_df["Index"] == species_id, "Species Name"].values[0]
                    print(f"  Generating map for species {species_id} ({species_name}) in {continent}...")
                    # Create a closure so the R model can predict on the
                    # continental grid features built inside the map generator.
                    def r_predict(feats, _mp=model_path):
                        return predict_grid_with_maxent_r(_mp, feats)
                    generate_maps_for_species(
                        model=None,
                        species_id=species_id,
                        species_name=species_name,
                        data_dir=data_dir,
                        output_dir=map_dir,
                        continent=continent,
                        predictor_subset=dm_config.get("predictors_subset"),
                        transforms=nan_only_transform,
                        use_location=use_location,
                        obs_locations=test_locs.numpy(),
                        obs_targets=test_targets_tensor.numpy(),
                        obs_probs=test_probs_tensor.numpy(),
                        threshold=0.5,
                        tn_subsample=10_000,
                        mode="eval",
                        predict_func=r_predict,
                    )

        except Exception as e:
            print(f"  Species {species_id}: evaluation failed - {e}")
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
