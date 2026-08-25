import os
import glob
import shutil
import numpy as np
import pandas as pd
import h5py
import hdf5plugin  # noqa: F401  — registers the Blosc filters the data is written with
from torchmetrics.classification import BinaryAUROC, BinaryF1Score, BinaryAveragePrecision
from src.metrics.precision_recall_gain import MultilabelAUPRG

CONTINENT_BOUNDS = {
    'europe': {'lon_min': -10, 'lon_max': 35, 'lat_min': 34, 'lat_max': 72},
    'australia': {'lon_min': 110, 'lon_max': 155, 'lat_min': -45, 'lat_max': -10},
    'north_america': {'lon_min': -170, 'lon_max': -50, 'lat_min': 15, 'lat_max': 75},
    'south_america': {'lon_min': -85, 'lon_max': -30, 'lat_min': -60, 'lat_max': 15},
    'africa': {'lon_min': -20, 'lon_max': 55, 'lat_min': -40, 'lat_max': 40},
    'asia': {'lon_min': 25, 'lon_max': 140, 'lat_min': -10, 'lat_max': 75}
}

def cleanup_cache(cache_dir):
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
        print(f"Deleted cache folder: {cache_dir}")

def set_global_seed(seed: int = 42):
    import torch
    import random
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def pick_continent(positive_locations):
    counts = {}
    for continent, bounds in CONTINENT_BOUNDS.items():
        lon_min, lon_max = bounds['lon_min'], bounds['lon_max']
        lat_min, lat_max = bounds['lat_min'], bounds['lat_max']
        mask = (
            (positive_locations[:, 0] >= lon_min) & (positive_locations[:, 0] <= lon_max) &
            (positive_locations[:, 1] >= lat_min) & (positive_locations[:, 1] <= lat_max)
        )
        counts[continent] = mask.sum()
    return max(counts, key=counts.get)


# ── Per-species metric helpers (shared by the sklearn/maxent train + eval scripts) ──
def get_binary_metric(metric_conf):
    """Instantiate a binary torchmetrics metric from the model-config dict.

    The shared model config declares ``Multilabel*`` metrics for the deep models;
    the per-species traditional baselines evaluate one species at a time, so the
    corresponding ``Binary*`` variant is used instead (and AUPRG via a 1-label
    ``MultilabelAUPRG``).
    """
    class_path = metric_conf["class_path"]
    init_args = metric_conf.get("init_args", {})
    metric_name = class_path.split(".")[-1]

    if metric_name.startswith("Multilabel"):
        metric_name = metric_name.replace("Multilabel", "Binary")

    metric_map = {
        "BinaryAUROC": BinaryAUROC,
        "BinaryF1Score": BinaryF1Score,
        "BinaryAveragePrecision": BinaryAveragePrecision,
    }

    unsupported = {"num_labels", "ignore_index", "average", "thresholds"}
    clean_args = {k: v for k, v in init_args.items() if k not in unsupported}

    if "AUPRG" in metric_name:
        return MultilabelAUPRG(num_labels=1, average="macro")

    if metric_name not in metric_map:
        raise ValueError(
            f"Unsupported metric: {metric_name}. "
            f"Available: {list(metric_map.keys()) + ['AUPRG']}"
        )
    return metric_map[metric_name](**clean_args)


def _compute_metric(metric, preds, targets):
    """Compute a metric handling both standard torchmetrics and AUPRG."""
    metric.reset()
    if isinstance(metric, MultilabelAUPRG):
        metric.update(preds.unsqueeze(1), targets.unsqueeze(1))
        return float(metric.compute().item())
    return float(metric(preds, targets).item())


def create_metrics_h5(h5_path, num_species, metric_names):
    """Create or extend an HDF5 file with per-species metric datasets."""
    if not os.path.exists(h5_path):
        with h5py.File(h5_path, "w") as hf:
            for m in metric_names:
                hf.create_dataset(m, shape=(num_species,), dtype="float32", fillvalue=np.nan)
    else:
        with h5py.File(h5_path, "a") as hf:
            for m in metric_names:
                if m not in hf:
                    hf.create_dataset(m, shape=(num_species,), dtype="float32", fillvalue=np.nan)


# ── Per-job result merging (shared by merge_sklearn_results / merge_maxent_results) ──
def merge_predictions(run_name, log_dir, num_species):
    """Merge all per-job ``predictions_by_species_*_*.h5`` into one file."""
    pred_dir = os.path.join(log_dir, "traditional_sdms", run_name, "predictions")
    pattern = os.path.join(pred_dir, "predictions_by_species_*_*.h5")
    job_files = sorted(glob.glob(pattern))

    if not job_files:
        print("No prediction files found!")
        return None

    print(f"Found {len(job_files)} prediction files")

    with h5py.File(job_files[0], "r") as f:
        num_val = f["val"].shape[1]
        num_test = f["test"].shape[1]

    merged_path = os.path.join(pred_dir, "predictions_by_species.h5")

    with h5py.File(merged_path, "w") as merged:
        merged.create_dataset("val", shape=(num_species, num_val), dtype="float32", fillvalue=np.nan)
        merged.create_dataset("test", shape=(num_species, num_test), dtype="float32", fillvalue=np.nan)

        for jf in job_files:
            print(f"  Processing {os.path.basename(jf)}")
            with h5py.File(jf, "r") as f:
                s = int(f.attrs["start_idx"])
                e = int(f.attrs["end_idx"])
                merged["val"][s : e + 1, :] = f["val"][:]
                merged["test"][s : e + 1, :] = f["test"][:]

    print(f"Merged predictions saved to {merged_path}")
    return merged_path


def merge_training_times(run_name, log_dir):
    """Merge all per-job ``training_times_*_*.csv`` into one file."""
    time_dir = os.path.join(log_dir, "traditional_sdms", run_name)
    pattern = os.path.join(time_dir, "training_times_*_*.csv")
    csv_files = sorted(glob.glob(pattern))

    if not csv_files:
        print("No training-time files found!")
        return None

    print(f"\nFound {len(csv_files)} CSV files to merge")
    dfs = [pd.read_csv(f) for f in csv_files]
    merged_df = (
        pd.concat(dfs, ignore_index=True)
        .drop_duplicates(subset="species_id", keep="first")
        .sort_values("species_id")
        .reset_index(drop=True)
    )

    merged_path = os.path.join(time_dir, "training_times.csv")
    merged_df.to_csv(merged_path, index=False)

    ok = merged_df["train_time_seconds"].notna().sum()
    fail = merged_df["train_time_seconds"].isna().sum()
    print(f"Merged {len(merged_df)} species  (ok={ok}, failed/missing={fail})")
    print(f"Saved to: {merged_path}")
    return merged_path


def merge_eval_metrics(run_name, log_dir, num_species):
    """Merge per-job ``metrics_by_species_*_*.h5`` into one file.

    Each per-job file has the same datasets (one per metric, shape
    ``(num_species,)``).  Non-NaN values in each job file overwrite the
    corresponding slot in the merged file.
    """
    run_dir = os.path.join(log_dir, "traditional_sdms", run_name)
    pattern = os.path.join(run_dir, "metrics_by_species_*_*.h5")
    job_files = sorted(glob.glob(pattern))

    if not job_files:
        print("No per-job eval metric files found.")
        return None

    print(f"\nFound {len(job_files)} per-job eval metric files")

    merged_path = os.path.join(run_dir, "metrics_by_species.h5")

    # Discover all dataset (metric) names from the first file
    with h5py.File(job_files[0], "r") as f:
        metric_names = list(f.keys())

    if not os.path.exists(merged_path):
        with h5py.File(merged_path, "w") as mf:
            for m in metric_names:
                mf.create_dataset(m, shape=(num_species,), dtype="float32", fillvalue=np.nan)

    with h5py.File(merged_path, "a") as mf:
        for m in metric_names:
            if m not in mf:
                mf.create_dataset(m, shape=(num_species,), dtype="float32", fillvalue=np.nan)

        for jf in job_files:
            with h5py.File(jf, "r") as f:
                for m in metric_names:
                    if m not in f:
                        continue
                    data = f[m][:]
                    valid = ~np.isnan(data)
                    if valid.any():
                        merged = mf[m][:]
                        merged[valid] = data[valid]
                        mf[m][:] = merged

    n_ok = 0
    with h5py.File(merged_path, "r") as mf:
        if metric_names:
            n_ok = int((~np.isnan(mf[metric_names[0]][:])).sum())
    print(f"Merged eval metrics for {n_ok}/{num_species} species")
    print(f"Saved to: {merged_path}")
    return merged_path



