#!/usr/bin/env python
"""
Merge per-job HDF5 prediction files, training-time CSVs, and eval metrics
produced by maxent_train.py into single consolidated files, and log the total
training time to WandB for paper comparisons.

Usage
-----
    python scripts/traditional_sdms/merge_maxent_results.py [--config path]
"""

import os
import yaml
import argparse

import pandas as pd
import rootutils

root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from utils import merge_predictions, merge_training_times, merge_eval_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge MaxEnt SDM results")
    parser.add_argument(
        "--config",
        default=os.path.join(root, "configs/model/maxent.yaml"),
        help="Path to the MaxEnt YAML config file",
    )
    args = parser.parse_args()

    with open(os.path.join(root, "configs/local_default.yaml")) as f:
        local_cfg = yaml.safe_load(f)
    log_dir = local_cfg["log_dir"]

    with open(args.config) as f:
        maxent_cfg = yaml.safe_load(f)

    species_df = pd.read_csv(os.path.join(local_cfg["data_dir"], "targets", "species_names.csv"))
    num_species = len(species_df)

    model_choice = maxent_cfg["model"].lower()
    bg_type = maxent_cfg.get("bg_type", "random")
    seed = maxent_cfg.get("seed", 42)
    config_tag = maxent_cfg.get("config_tag", None)
    run_name = f"{model_choice}_{bg_type}_s{seed}"
    if config_tag:
        run_name = f"{run_name}_{config_tag}"

    print(f"Merging results for {run_name}...\n")

    merge_predictions(run_name, log_dir, num_species)
    time_result = merge_training_times(run_name, log_dir)
    merge_eval_metrics(run_name, log_dir, num_species)

    # Log total training time to WandB for paper comparisons
    if time_result is not None:
        import wandb
        times_df = pd.read_csv(time_result)
        total_s = times_df["train_time_seconds"].sum(skipna=True)
        n_species = times_df["train_time_seconds"].notna().sum()
        wandb.init(
            project="sdm_benchmark_final",
            name=run_name,
            tags=["traditional_sdm"],
            config={"model": model_choice, "bg_type": bg_type, "n_species": int(num_species)},
        )
        wandb.summary["perf/total_train_time_s"] = total_s
        wandb.summary["perf/total_train_time_h"] = total_s / 3600
        wandb.summary["perf/n_species_timed"] = int(n_species)
        wandb.summary["perf/mean_train_time_per_species_s"] = times_df["train_time_seconds"].mean(skipna=True)
        print(f"\nLogged to WandB: total_train_time={total_s:.1f}s ({total_s/3600:.2f}h) over {n_species} species")
        wandb.finish()
