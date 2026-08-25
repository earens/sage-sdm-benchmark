#!/usr/bin/env python
"""
Merge per-job HDF5 prediction files, training-time CSVs, and eval metrics
produced by sklearn_train.py into single consolidated files.

Usage
-----
    python scripts/traditional_sdms/merge_sklearn_results.py [--config path]
"""

import os
import yaml
import argparse

import pandas as pd
import rootutils

root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from utils import merge_predictions, merge_training_times, merge_eval_metrics


def main():
    parser = argparse.ArgumentParser(description="Merge sklearn SDM results")
    parser.add_argument(
        "--config",
        default=os.path.join(root, "configs/model/sklearn.yaml"),
        help="Path to the sklearn YAML config file",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    with open(os.path.join(root, "configs/local_default.yaml")) as f:
        local_cfg = yaml.safe_load(f)
    log_dir = local_cfg["log_dir"]
    data_dir = local_cfg["data_dir"]

    species_df = pd.read_csv(os.path.join(data_dir, "targets", "species_names.csv"))
    num_species = len(species_df)

    model_choice = cfg["model"].lower()
    bg_type = cfg.get("bg_type", "random")
    seed = cfg.get("seed", 42)
    config_tag = cfg.get("config_tag", None)
    run_name = f"{model_choice}_{bg_type}_s{seed}"
    if config_tag:
        run_name = f"{run_name}_{config_tag}"

    print(f"Merging results for {run_name} …\n")
    merge_predictions(run_name, log_dir, num_species)
    merge_training_times(run_name, log_dir)
    merge_eval_metrics(run_name, log_dir, num_species)


if __name__ == "__main__":
    main()
