import os
import time
import yaml
import rootutils
import torch
from torch.utils.data import DataLoader, Subset
import numpy as np
from omegaconf import OmegaConf
import h5py
import hdf5plugin  # noqa: F401  — registers the Blosc filters the data is written with
import random
import subprocess
from tqdm import tqdm
import pandas as pd
import sys


rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from utils import set_global_seed, get_binary_metric, _compute_metric, create_metrics_h5
from src.data_modules.datamodule import DataModule
from src.models.resnet import encode_locations

root = os.getenv("PROJECT_ROOT")
local_default_cfg = OmegaConf.load(os.path.join(root, "configs/local_default.yaml"))
data_dir = local_default_cfg["data_dir"]
log_dir = local_default_cfg["log_dir"]

# Avoid HDF5 locking issues
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

CACHE_DIR = os.path.join(root, "cache")

# Models that train on a large (up to 10k) background pool rather than a 1:1
# presence-matched sample. GLM/GAM/BRT additionally down-weight the background
# (see WEIGHTED_MODELS); RF handles imbalance via per-tree down-sampling.
TEN_K_BG_MODELS = {"maxent", "glm", "gam", "randomforest", "brt"}
# Models that receive IPP-style case weights so that the summed presence weight
# equals the summed background weight.
WEIGHTED_MODELS = {"glm", "gam", "brt"}

# for data transfer between python and R maxent
def write_maxent_input(h5_path, X_train, y_train, X_val, X_test):


    def ensure_2d(arr):
        arr_np = arr.cpu().numpy() if hasattr(arr, "cpu") else np.array(arr)
        arr_np = arr_np.astype(np.float32)
        if arr_np.ndim == 1:
            arr_np = arr_np.reshape(-1, 1)  # Changed: samples as rows
        elif arr_np.ndim > 2:
            arr_np = arr_np.reshape(arr_np.shape[0], -1)
        return arr_np

    X_train_np = ensure_2d(X_train)
    X_val_np   = ensure_2d(X_val)
    X_test_np  = ensure_2d(X_test)
    y_train_np = np.array(y_train.cpu().numpy().ravel() if hasattr(y_train, "cpu") else y_train, dtype=np.float32)

    with h5py.File(h5_path, "w") as f:
        f.create_dataset("X_train", data=X_train_np)
        f.create_dataset("y_train", data=y_train_np)
        f.create_dataset("X_val",   data=X_val_np)
        f.create_dataset("X_test",  data=X_test_np)



def run_maxent_r(input_h5, output_h5, model_path, time_log, species_id, seed=42):
    cmd = [
        "Rscript",
        os.path.join(os.path.dirname(__file__), "maxent_train.R"),
        input_h5,
        output_h5,
        model_path,
        time_log,
        str(species_id),
        str(seed),
    ]
    subprocess.run(cmd, check=True)

def read_maxent_output(path):
    with h5py.File(path, "r") as h5f:
        val_preds = h5f["val"][:]
        test_preds = h5f["test"][:]
    return val_preds, test_preds



def get_data(datamodule, species_id: int, split: str, num_workers: int = 4, cap: int = None,
                     model_choice: str = "maxent", bg_type: str = "random", suffix: str = "",
                     skip_background: bool = False, seed: int = 0):
    """
    Get collated location/predictor/target data for a given species and split.
    Since maxent is not trained in batches, we load all data at once.

    ``skip_background`` (train split only) returns presences alone, skipping the
    background-pool load entirely — used by presence-only models (e.g. envelopes)
    for which loading background is pure wasted I/O.
    """

    def load_batch(dataset, indices=None):
        if indices is not None:
            #subsample indices
            if datamodule.subsample_indices is not None and split == "train":
                full_to_subsample = {full_idx: sub_idx for sub_idx, full_idx in enumerate(datamodule.subsample_indices)}
                mapped_indices = [full_to_subsample[idx] for idx in indices if idx in full_to_subsample]
                indices = mapped_indices

            dataset = Subset(dataset, indices)
            batch_size = len(indices)
        else:
            batch_size = len(dataset) if cap is None else min(len(dataset), cap)

        loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers,
                            pin_memory=True, shuffle=False)

        return next(iter(loader)), batch_size

    if split == "train":
        # get indices for species of interest from the targets HDF5 file
        targets_h5_path = (
            f"{datamodule.hparams.data_dir}targets/"
            f"{datamodule.train_dataset}_targets{suffix}.h5"
        )
        with h5py.File(targets_h5_path, "r") as f:
            idxs_lookup = f["target_location_indices_lookup"]
            idxs = f["target_location_indices"]
            start, end = int(idxs_lookup[species_id]), int(idxs_lookup[species_id + 1])
            location_indices = idxs[start:end].tolist()

        # Apply subsampling if subsample_indices are available
        if datamodule.subsample_indices is not None:
            subsample_set = set(datamodule.subsample_indices)
            location_indices = [idx for idx in location_indices if idx in subsample_set]

            if len(location_indices) == 0:
                print(f"Warning: No observations for species {species_id} after subsampling!")

        # cap positives if requested
        if cap is not None and len(location_indices) > cap:
            location_indices = random.sample(location_indices, cap)

        batch, _ = load_batch(datamodule.data_train, location_indices)
        locations = batch["observation_locations"]
        predictors = batch["observation_predictors"]
        targets = torch.ones((locations.shape[0],), dtype=torch.float32)

        # Presence-only models skip the (expensive) background load entirely.
        if skip_background:
            return {
                "locations": locations,
                "predictors": predictors,
                "targets": targets,
                "observation_location_indices": torch.arange(len(location_indices)),
            }

        # background cutoff
        bg_cutoff = 10_000 if model_choice in TEN_K_BG_MODELS else len(location_indices)

        # choose background index file
        if bg_type == "tgb":
            path = f"{datamodule.hparams.data_dir}/range_masks{suffix}/train_species_index.h5"
        else:
            path = f"{datamodule.hparams.data_dir}/range_masks/background_species_index.h5"

        with h5py.File(path, "r") as f:
            idxs_tgb = f["target_location_indices"]
            idxs_lookup_tgb = f["target_location_indices_lookup"]
            start_tgb, end_tgb = int(idxs_lookup_tgb[species_id]), int(idxs_lookup_tgb[species_id + 1])
            bg_indices = list(set(idxs_tgb[start_tgb:end_tgb].tolist()) - set(location_indices))

        # Apply subsampling to background indices if needed
        if datamodule.subsample_indices is not None:
            subsample_set = set(datamodule.subsample_indices)
            bg_indices = [idx for idx in bg_indices if idx in subsample_set]

        if len(bg_indices) >= len(location_indices):
            # shuffle and cut to bg_cutoff

            bg_gen = torch.Generator().manual_seed(seed * 1_000_003 + species_id)
            bg_indices = torch.tensor(bg_indices)[
                torch.randperm(len(bg_indices), generator=bg_gen)[:bg_cutoff]
            ].tolist()
        else:
            print("Not enough background points available, take all.")

        bg_batch, _ = load_batch(datamodule.data_train, bg_indices)
        if bg_type == "tgb":
            bg_locations = bg_batch["observation_locations"]
            bg_predictors = bg_batch["observation_predictors"]
        else:
            bg_locations = bg_batch["bg_locations"]
            bg_predictors = bg_batch["bg_predictors"]
        bg_targets = torch.zeros((bg_locations.shape[0],), dtype=torch.float32)

        return {
            "locations": torch.cat((locations, bg_locations), dim=0),
            "predictors": torch.cat((predictors, bg_predictors), dim=0),
            "targets": torch.cat((targets, bg_targets), dim=0),
            "observation_location_indices": torch.arange(len(location_indices))
        }

    elif split in {"val", "test"}:
        dataset = datamodule.data_val if split == "val" else datamodule.data_test
        batch, batch_size = load_batch(dataset)
        locations = batch["observation_locations"]
        predictors = batch["observation_predictors"]
        targets = batch["observation_targets"][:, :].float()
        location_indices = torch.arange(batch_size)

        return {
            "locations": locations,
            "predictors": predictors,
            "targets": targets,
            "location_indices": location_indices
        }

    else:
        raise ValueError(f"Unknown split: {split}")


def prepare_train_data(batch, use_location):
    """add location encoding if needed and prepare X, y for training"""
    locations = batch["locations"]
    predictors = batch["predictors"]
    targets = batch["targets"]
    if use_location:
        locations = encode_locations(locations)
        X = torch.cat((locations, predictors), dim=-1)
    else:
        X = predictors
    y = targets
    return X, y






def main():

    start_idx = int(sys.argv[1])
    end_idx   = int(sys.argv[2])

    cfg_path = f"{root}/configs/model/maxent.yaml"
    if "--config" in sys.argv:
        cfg_path = sys.argv[sys.argv.index("--config") + 1]

    with open(cfg_path) as stream:
        rf_config = yaml.safe_load(stream)

    model_choice = rf_config["model"].lower()
    species_ids = rf_config["species_ids"]
    bg_type = rf_config.get("bg_type", "random")
    seed = rf_config.get("seed", 42)
    train_cap = rf_config.get("train_cap", 10000)
    skip_existing = rf_config.get("skip_existing", False)
    normalize_predictors = rf_config.get("normalize_predictors", True)

    config_tag = rf_config.get("config_tag", None)
    run_name = f"{model_choice}_{bg_type}_s{seed}"
    if config_tag:
        run_name = f"{run_name}_{config_tag}"
    out_dir = os.path.join(log_dir, "traditional_sdms", run_name, "predictions")
    model_dir = os.path.join(log_dir, "traditional_sdms", run_name, "models")
    map_dir = os.path.join(log_dir, "traditional_sdms", run_name, "maps")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(map_dir, exist_ok=True)

    set_global_seed(seed)

    # Load datamodule config
    with open(f"{root}/configs/data/datamodule.yaml") as stream:
        dm_config = yaml.safe_load(stream)["data"]["init_args"]
        SUFFIX = "_1km" if dm_config.get("aggregated", True) else ""

    # Load model config for metrics and use_location
    with open(f"{root}/configs/model/resnet.yaml") as stream:
        model_config = yaml.safe_load(stream)
        use_location = model_config["model"]["init_args"]["net"]["init_args"].get("use_location", True)

    # Read num_species from canonical species list
    with open(f"{root}/configs/local_default.yaml") as stream:
        _data_dir = yaml.safe_load(stream)["data_dir"]
    species_df = pd.read_csv(os.path.join(_data_dir, "targets", "species_names.csv"))
    num_species = len(species_df)

    if species_ids == "all":
        species_ids = list(range(num_species))
    else:
        species_ids = [int(sid) for sid in species_ids]

    species_ids = species_ids[start_idx:end_idx+1]

    # Filter out species that already have models if skip_existing is True
    if skip_existing:
        original_count = len(species_ids)
        species_ids = [
            sid for sid in species_ids
            if not os.path.exists(os.path.join(model_dir, f"species_{sid}.rds"))
        ]
        skipped_count = original_count - len(species_ids)
        if skipped_count > 0:
            print(f"Skipping {skipped_count} species with existing models")
            print(f"Training {len(species_ids)} species")

        if len(species_ids) == 0:
            print("All species in this range already have trained models. Exiting.")
            return

    # Add data_dir from local config
    dm_config["data_dir"] = data_dir

    # Initialize the data module
    print("Initializing data module")
    dm = DataModule(**dm_config)
    dm.setup(stage="fit")

    # MaxEnt traditionally uses raw predictors with NaN imputation only.
    # Set normalize_predictors: true in maxent.yaml to z-score normalize instead.
    nan_only_transform = dm._make_transforms(normalize=normalize_predictors)
    dm.data_train.transforms = nan_only_transform
    # data_val / data_test are Subsets wrapping a PADataset
    dm.data_val.dataset.transforms = nan_only_transform
    dm.data_test.dataset.transforms = nan_only_transform

    # Create separate HDF5 file for this job to avoid locking issues
    h5_path = os.path.join(out_dir, f"predictions_by_species_{start_idx}_{end_idx}.h5")

    # ── Load val / test once (with range masks for inline eval) ──────────
    print("Loading val/test data")
    _val_loader = torch.utils.data.DataLoader(
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

    # Compute feature matrices for R prediction
    val_locations  = val_data_full["observation_locations"]
    val_predictors = val_data_full["observation_predictors"]
    test_locations  = test_data_full["observation_locations"]
    test_predictors = test_data_full["observation_predictors"]

    if use_location:
        val_locations_enc  = encode_locations(val_locations)
        test_locations_enc = encode_locations(test_locations)
        val_data = torch.cat((val_locations_enc, val_predictors), dim=-1)
        test_data = torch.cat((test_locations_enc, test_predictors), dim=-1)
        del val_locations_enc, test_locations_enc
    else:
        val_data = val_predictors
        test_data = test_predictors

    # Free intermediate tensors — keep only val_data/test_data, range masks, targets
    del val_data_full, test_data_full, _val_loader, _test_loader
    del val_locations, val_predictors, test_locations, test_predictors

    # Create HDF5 file for predictions for THIS job's species range
    num_species_in_job = end_idx - start_idx + 1
    if not os.path.exists(h5_path):
        with h5py.File(h5_path, "w") as h5f:
            h5f.create_dataset("val", shape=(num_species_in_job, num_val), dtype="float32", fillvalue=np.nan)
            h5f.create_dataset("test", shape=(num_species_in_job, num_test), dtype="float32", fillvalue=np.nan)
            # Store metadata for merging later
            h5f.attrs["start_idx"] = start_idx
            h5f.attrs["end_idx"] = end_idx

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

    # Open HDF5 files and iterate species
    with h5py.File(h5_path, "a") as h5f, \
         h5py.File(metrics_h5_path, "a") as mhf:

        for idx, species_id in enumerate(tqdm(species_ids)):
            # Check again if model exists (in case of race conditions in parallel jobs)
            model_filename = os.path.join(model_dir, f"species_{species_id}.rds")
            if skip_existing and os.path.exists(model_filename):
                print(f"Model already exists for species {species_id}, skipping...")
                continue

            # load training data for this species
            species_start_time = time.perf_counter()
            print(f"Loading training data for species {species_id}...")
            try:
                train_batch = get_data(dm, species_id, num_workers=0, cap=train_cap,
                                               split="train", model_choice=model_choice, bg_type=bg_type, suffix=SUFFIX,
                                               seed=seed)
            except ValueError as e:
                print(f"Skipping species {species_id}: {e}")
                continue

            X_train, y_train = prepare_train_data(train_batch, use_location)
            print(f"Training data loaded for species {species_id} (N={X_train.shape[0]})")


            # shuffle training data
            perm = torch.randperm(X_train.size(0), generator=torch.Generator().manual_seed(seed))
            X_train = X_train[perm]
            y_train = y_train[perm]


            print(f"Training model for species {species_id} with {model_choice}...")

            # Call R MaxEnt implementation
            tmp_dir = os.path.join(out_dir, "tmp_maxent")
            os.makedirs(tmp_dir, exist_ok=True)

            input_h5 = os.path.join(tmp_dir, f"species_{species_id}_input.h5")
            output_h5 = os.path.join(tmp_dir, f"species_{species_id}_output.h5")

            write_maxent_input(
                input_h5,
                X_train,
                y_train,
                val_data,
                test_data,
            )

            # Create time log CSV with job-specific name to avoid locking
            time_log_csv = os.path.join(
                log_dir, "traditional_sdms", run_name,
                f"training_times_{start_idx}_{end_idx}.csv",
            )

            # Create CSV with pre-allocated rows if it doesn't exist
            if not os.path.exists(time_log_csv):
                df = pd.DataFrame({
                    'species_id': species_ids,
                    'train_time_seconds': [np.nan] * len(species_ids),
                    'eval_time_seconds': [np.nan] * len(species_ids),
                    'n_samples': [-1] * len(species_ids),
                    'n_presences': [-1] * len(species_ids),
                    'n_features': [-1] * len(species_ids),
                    'feature_classes': [''] * len(species_ids)
                })
                df.to_csv(time_log_csv, index=False)

            # NOTE only support R MaxEnt for now
            if model_choice == "maxent":
                print(f"Running R MaxEnt for species {species_id}...")
                try:
                    run_maxent_r(input_h5, output_h5, model_filename, time_log_csv, species_id, seed)
                    val_preds, test_preds = read_maxent_output(output_h5)
                except subprocess.CalledProcessError:
                    print(f"MaxEnt failed for species {species_id}, skipping...")
                    continue
            else:
                pass

            # Overwrite R's train_time with total time including data loading
            total_species_time = time.perf_counter() - species_start_time
            time_df = pd.read_csv(time_log_csv)
            row_mask = time_df['species_id'] == species_id
            if row_mask.any():
                time_df.loc[row_mask, 'train_time_seconds'] = total_species_time
                time_df.to_csv(time_log_csv, index=False)

            # store predictions using local index (idx) instead of global species_id
            h5f["val"][idx, :] = val_preds
            h5f["test"][idx, :] = test_preds

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

            val_preds_t  = torch.from_numpy(val_preds).float().squeeze()
            test_preds_t = torch.from_numpy(test_preds).float().squeeze()

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

            # Update eval time in the time log
            time_df = pd.read_csv(time_log_csv)
            row_mask = time_df['species_id'] == species_id
            if row_mask.any():
                time_df.loc[row_mask, 'eval_time_seconds'] = eval_elapsed
                time_df.to_csv(time_log_csv, index=False)


if __name__ == "__main__":
    main()
