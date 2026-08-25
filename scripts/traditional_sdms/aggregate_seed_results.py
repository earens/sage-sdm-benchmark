#!/usr/bin/env python
"""Aggregate per-species test metrics across the 5 final seed runs into
mean ± std (per species, and overall) for the results table.

Each final run is ``<model>_<bg>_s<seed>[_<tag>]`` (per-species frozen
hyperparameter, seeds vary only the background draw). This stacks their merged
``metrics_by_species.h5`` files and summarises the ``test_*`` metrics.

Usage
-----
    python scripts/traditional_sdms/aggregate_seed_results.py \
        --model glm --bg tgb --tag v3 --seeds 0 1 2 3 4
"""
import os
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import h5py
import hdf5plugin  # noqa: F401  — registers the Blosc filters the data is written with
import yaml

root = os.getenv("PROJECT_ROOT") or str(Path(__file__).resolve().parents[2])
with open(os.path.join(root, "configs/local_default.yaml")) as f:
    log_dir = yaml.safe_load(f)["log_dir"]


def main():
    ap = argparse.ArgumentParser(description="Aggregate test metrics across seeds")
    ap.add_argument("--model", required=True)
    ap.add_argument("--bg", default="tgb")
    # Runs are tagged by campaign (config_tag in the model YAML). Without this the
    # untagged run directories of an earlier campaign are picked up silently, since
    # they still exist on disk and the path resolves.
    ap.add_argument("--tag", default=None, help="config_tag of the campaign, e.g. v3")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    per_seed = {}        # metric -> list of per-species arrays
    metric_names = None
    for seed in args.seeds:
        run = f"{args.model}_{args.bg}_s{seed}"
        if args.tag:
            run = f"{run}_{args.tag}"
        mpath = os.path.join(log_dir, "traditional_sdms", run, "metrics_by_species.h5")
        if not os.path.exists(mpath):
            raise FileNotFoundError(f"Missing final run metrics: {mpath}")
        with h5py.File(mpath, "r") as f:
            names = [k for k in f.keys() if k.startswith("test_")]
            metric_names = metric_names or names
            for m in metric_names:
                per_seed.setdefault(m, []).append(f[m][:])

    num_species = len(next(iter(per_seed.values()))[0])
    n_seeds = len(args.seeds)
    cols = {"species_id": np.arange(num_species)}
    summary = {}
    coverage = None
    for m, arrs in per_seed.items():
        stack = np.vstack(arrs)                      # (n_seeds, num_species)
        ok = ~np.isnan(stack)
        n_ok = ok.sum(axis=0)                        # seeds per species
        common = n_ok == n_seeds


        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            cols[f"{m}_mean"] = np.where(n_ok > 0, np.nanmean(np.where(ok, stack, np.nan), axis=0), np.nan)
            cols[f"{m}_std"] = np.where(n_ok > 1, np.nanstd(np.where(ok, stack, np.nan), axis=0), np.nan)
        cols[f"{m}_n_seeds"] = n_ok

        # Macro over species, per seed. "union" lets each seed use its own species
        # set; "common" fixes the set so the spread is comparable across models.
        macro_union = np.nanmean(stack, axis=1)
        macro_union_med = np.nanmedian(stack, axis=1)
        macro_common = stack[:, common].mean(axis=1) if common.any() else np.full(n_seeds, np.nan)
        macro_common_med = np.median(stack[:, common], axis=1) if common.any() else np.full(n_seeds, np.nan)
        summary[m] = {
            "union_mean": (float(np.mean(macro_union)), float(np.std(macro_union))),
            "union_median": (float(np.mean(macro_union_med)), float(np.std(macro_union_med))),
            "common_mean": (float(np.mean(macro_common)), float(np.std(macro_common))),
            "common_median": (float(np.mean(macro_common_med)), float(np.std(macro_common_med))),
        }
        coverage = (int((n_ok > 0).sum()), int(common.sum()), n_ok)

    run_label = f"{args.model}_{args.bg}" + (f"_{args.tag}" if args.tag else "")
    out = args.out or os.path.join(log_dir, "traditional_sdms", f"{run_label}_test_summary.csv")
    pd.DataFrame(cols).to_csv(out, index=False)
    print(f"Wrote per-species summary ({num_species} species) -> {out}")

    n_union, n_common, n_ok = coverage
    print(f"\nCoverage over {n_seeds} seeds: {n_union}/{num_species} species fitted by "
          f"at least one seed, {n_common} by all {n_seeds}.")
    if n_union < num_species or n_common < n_union:
        hist = np.bincount(n_ok, minlength=n_seeds + 1)
        print("  species by seed count: " + ", ".join(f"{k}:{hist[k]}" for k in range(n_seeds + 1)))

    print("\nOverall test metrics (mean ± std over seeds):")
    for m, s in summary.items():
        print(f"  {m}")
        for key, label in (("union_mean", "union   macro-mean  "),
                           ("union_median", "union   macro-median"),
                           ("common_mean", "common  macro-mean  "),
                           ("common_median", "common  macro-median")):
            mu, sd = s[key]
            print(f"    {label}: {mu:.4f} ± {sd:.4f}")


if __name__ == "__main__":
    main()
