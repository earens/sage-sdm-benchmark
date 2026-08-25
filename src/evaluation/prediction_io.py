"""I/O and validation for the SDM benchmark prediction format.


HDF5 (recommended, extension ``.h5`` / ``.hdf5``)
    probabilities  float  (N_locations, N_species)  predicted scores, ideally in [0, 1]
    targets        float  (N_locations, N_species)  {0, 1} presence/absence, -1 = ignore (optional)

CSV
    A wide table with one probability column per species, in the canonical species
    order. Column names are ignored -- only the order matters; every column is read
    as a species (any ``longitude`` / ``latitude`` columns, if present, are dropped).

Row order and species order
    Rows must be aligned by position to the benchmark's evaluation locations, and
    species must be in the canonical order of ``targets/species_names.csv`` (column
    ``i`` is species ``i``). ``targets`` are the benchmark's ground-truth presence-
    absence labels.
"""

from __future__ import annotations

import os

import h5py
import hdf5plugin  # noqa: F401  — registers the Blosc filters the data is written with
import numpy as np
import pandas as pd

PREDICTION_FORMAT = __doc__


def _is_hdf5(path: str) -> bool:
    return path.lower().endswith((".h5", ".hdf5"))


def load_predictions(
    path: str,
    targets_path: str | None = None,
    num_species: int | None = None,
) -> dict:
    """Load a predictions file and validate its structure.

    Args:
        path: Path to an HDF5 or CSV predictions file (see module docstring).
        targets_path: Optional path to a separate ground-truth file (HDF5 with a
            ``targets`` dataset, or a ``test_predictions.h5``). Used when the
            predictions file itself has no ``targets``.
        num_species: Expected number of species (columns). If given, the
            probabilities are checked against it.

    Returns:
        dict with keys ``probabilities`` (N, S) float32, ``locations`` (N, 2)
        float32 or None, and ``targets`` (N, S) float32 or None.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Predictions file not found: {path}")

    if _is_hdf5(path):
        data = _load_hdf5_predictions(path)
    else:
        data = _load_csv_predictions(path)

    probs = data["probabilities"]
    if probs.ndim != 2:
        raise ValueError(f"'probabilities' must be 2-D (locations, species); got shape {probs.shape}")
    if num_species is not None and probs.shape[1] != num_species:
        raise ValueError(
            f"'probabilities' has {probs.shape[1]} species columns but {num_species} were expected. "
            "Species must be in the canonical order of targets/species_names.csv."
        )
    finite = probs[np.isfinite(probs)]
    if finite.size and (finite.min() < -1e-6 or finite.max() > 1 + 1e-6):
        raise ValueError(
            f"'probabilities' should lie in [0, 1]; observed range "
            f"[{float(finite.min()):.3f}, {float(finite.max()):.3f}]."
        )

    # Resolve targets: prefer in-file, else the separate ground-truth file.
    targets = data.get("targets")
    if targets is None and targets_path is not None:
        targets = load_ground_truth_targets(targets_path)
    if targets is not None:
        if targets.shape != probs.shape:
            raise ValueError(
                f"targets shape {targets.shape} does not match probabilities shape {probs.shape}. "
                "Predictions and ground truth must be row- and species-aligned."
            )
    data["targets"] = targets
    return data


def _load_hdf5_predictions(path: str) -> dict:
    with h5py.File(path, "r") as f:
        if "probabilities" not in f:
            raise ValueError(
                f"{path}: HDF5 predictions must contain a 'probabilities' dataset. "
                f"Found: {list(f.keys())}"
            )
        probs = f["probabilities"][:].astype(np.float32)
        locations = f["locations"][:].astype(np.float32) if "locations" in f else None
        targets = f["targets"][:].astype(np.float32) if "targets" in f else None
    return {"probabilities": probs, "locations": locations, "targets": targets}


def _load_csv_predictions(path: str) -> dict:
    df = pd.read_csv(path)
    lon_col = next((c for c in df.columns if c.lower() in ("longitude", "lon", "x")), None)
    lat_col = next((c for c in df.columns if c.lower() in ("latitude", "lat", "y")), None)
    locations = None
    if lon_col is not None and lat_col is not None:
        locations = df[[lon_col, lat_col]].to_numpy(dtype=np.float32)
        df = df.drop(columns=[lon_col, lat_col])
    probs = df.to_numpy(dtype=np.float32)
    return {"probabilities": probs, "locations": locations, "targets": None}


def load_ground_truth_targets(path: str) -> np.ndarray:
    """Load benchmark ground-truth presence-absence targets from an HDF5 file.

    Accepts either a file with a top-level ``targets`` dataset or a benchmark
    ``test_predictions.h5`` (which bundles ``targets`` alongside predictions).
    Returns a (N, S) float32 array using {0, 1} with -1 for ignored entries.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Ground-truth targets file not found: {path}")
    with h5py.File(path, "r") as f:
        if "targets" not in f:
            raise ValueError(f"{path}: expected a 'targets' dataset, found {list(f.keys())}")
        return f["targets"][:].astype(np.float32)
