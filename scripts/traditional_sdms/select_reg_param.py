#!/usr/bin/env python
"""Select, per species, the grid hyperparameter value with the best validation
metric across the 3 reference runs, and write a freeze CSV for the final 5-seed
runs (consumed by sklearn_train.py's ``per_species_param_file``).

Each reference run shares model/bg/seed but used a different ``config_tag`` and a
different value of the swept hyperparameter (C / lam / max_features / tc). This
reads each run's merged ``metrics_by_species.h5``, picks per species the value
that maximised the validation metric, and writes ``species_id,param``.

Usage
-----
    python scripts/traditional_sdms/select_reg_param.py \
        --model glm --bg tgb --seed 0 \
        --grid C0p001:0.001 C0p01:0.01 C0p1:0.1 \
        --metric val_auroc
"""
import os
import argparse
from collections import Counter
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
    ap = argparse.ArgumentParser(description="Per-species hyperparameter selection on val")
    ap.add_argument("--model", required=True)
    ap.add_argument("--bg", default="tgb")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--grid", nargs="+", required=True,
        help="tag:value pairs, one per reference run, e.g. C0p01:0.01 C0p1:0.1 C1:1.0",
    )
    ap.add_argument("--metric", default="val_auroc",
                    help="HDF5 dataset name to maximise (default: val_auroc)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    tags, values = [], []
    for item in args.grid:
        tag, val = item.split(":")
        tags.append(tag)
        values.append(float(val))
    values = np.array(values)

    scores = []  # (n_runs, num_species)
    for tag in tags:
        run = f"{args.model}_{args.bg}_s{args.seed}_{tag}"
        mpath = os.path.join(log_dir, "traditional_sdms", run, "metrics_by_species.h5")
        if not os.path.exists(mpath):
            raise FileNotFoundError(f"Missing reference run metrics: {mpath}")
        with h5py.File(mpath, "r") as f:
            if args.metric not in f:
                raise KeyError(f"{args.metric} not in {mpath}; have {list(f.keys())}")
            scores.append(f[args.metric][:])
    scores = np.vstack(scores)

    valid = ~np.all(np.isnan(scores), axis=0)
    safe = np.where(np.isnan(scores), -np.inf, scores)  # never pick a NaN run
    best_run = np.argmax(safe, axis=0)
    chosen = values[best_run]

    species_ids = np.arange(scores.shape[1])[valid]
    chosen = chosen[valid]

    out = args.out or os.path.join(
        log_dir, "traditional_sdms", f"{args.model}_{args.bg}_param.csv"
    )
    pd.DataFrame({"species_id": species_ids, "param": chosen}).to_csv(out, index=False)

    counts = {float(v): int(c) for v, c in Counter(chosen.tolist()).items()}
    print(f"Wrote {len(species_ids)} species -> {out}")
    print(f"Selected-value distribution (saturation check): {counts}")
    endpoints = {values.min(), values.max()}
    at_ends = sum(c for v, c in counts.items() if v in endpoints)
    if len(species_ids) and at_ends / len(species_ids) > 0.5:
        print("WARNING: >50% of species selected a grid endpoint — consider shifting the grid.")


if __name__ == "__main__":
    main()
