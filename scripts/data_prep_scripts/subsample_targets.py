"""Cap the number of training occurrences per species.

Heavily recorded species dominate presence-only training, so the benchmark draws at most
``num_samples_per_species`` occurrence cells per species and trains on the union of those
draws. The output is a sorted array of row indices into the training targets, which
``configs/data/datamodule.yaml`` selects via ``subsample_indices_file``.

    python scripts/data_prep_scripts/subsample_targets.py --num_samples_per_species 1000
"""

import argparse
import os

import h5py
import hdf5plugin  # noqa: F401  (registers the blosc filters the masks are written with)
import numpy as np
import rootutils
import yaml

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
root = os.getenv("PROJECT_ROOT")

with open(f"{root}/configs/local_default.yaml") as stream:
    data_dir = yaml.safe_load(stream)["data_dir"]

with open(f"{root}/configs/data/datamodule.yaml") as stream:
    data_config = yaml.safe_load(stream)
    _aggregated = data_config.get("data", {}).get("init_args", {}).get("aggregated", True)
    SUFFIX = "_1km" if _aggregated else ""


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--num_samples_per_species", type=int, default=1000,
                   help="Maximum occurrence cells to draw per species")
    p.add_argument("--seed", type=int, default=42, help="Base RNG seed (offset by species index)")
    p.add_argument("--targets", default=f"targets/train_targets{SUFFIX}.h5",
                   help="Training targets HDF5, relative to data_dir")
    p.add_argument("--mask", default=f"range_masks{SUFFIX}/train_species_index.h5",
                   help="Species-major native-range mask, relative to data_dir")
    p.add_argument("--no-range-mask", action="store_true",
                   help="Draw from every occurrence cell instead of the in-range ones")
    p.add_argument("--out", default=None, help="Output .npy, relative to data_dir")
    p.add_argument("--force", action="store_true", help="Overwrite an existing output file")
    return p.parse_args()


def _species_rows(idx_ds, lookup, s):
    """Sorted row indices of species `s` from a species-major index dataset."""
    return np.sort(idx_ds[lookup[s]:lookup[s + 1]])


def _intersect_sorted(occ, rng):
    """Elements of sorted `occ` that also appear in sorted `rng`."""
    if occ.size == 0 or rng.size == 0:
        return occ[:0]
    pos = np.searchsorted(rng, occ)
    pos[pos >= rng.size] = 0
    return occ[rng[pos] == occ]


def main():
    args = parse_args()
    n_samples = args.num_samples_per_species

    out_rel = args.out or f"targets/subsample_indices/train_indices_{n_samples}{SUFFIX}.npy"
    out_path = os.path.join(data_dir, out_rel)
    if os.path.exists(out_path) and not args.force:
        raise SystemExit(f"{out_path} exists; pass --force to overwrite")

    tgt_path = os.path.join(data_dir, args.targets)
    mask_file = None if args.no_range_mask else h5py.File(os.path.join(data_dir, args.mask), "r")

    print(f"Drawing <= {n_samples} cells/species (seed {args.seed}) from "
          f"{'all' if args.no_range_mask else 'in-range'} occurrences.")

    selected = set()
    try:
        with h5py.File(tgt_path, "r") as tf:
            t_idx, t_look = tf["target_location_indices"], tf["target_location_indices_lookup"][()]
            num_species = t_look.size - 1
            if mask_file is not None:
                m_idx = mask_file["target_location_indices"]
                m_look = mask_file["target_location_indices_lookup"][()]
                if m_look.size - 1 != num_species:
                    raise SystemExit(
                        f"mask covers {m_look.size - 1} species but targets have {num_species}")

            n_occ = np.zeros(num_species, dtype=np.int64)
            n_eligible = np.zeros(num_species, dtype=np.int64)
            n_taken = np.zeros(num_species, dtype=np.int64)

            for s in range(num_species):
                occ = _species_rows(t_idx, t_look, s)
                n_occ[s] = occ.size
                eligible = occ if mask_file is None else _intersect_sorted(
                    occ, _species_rows(m_idx, m_look, s))
                n_eligible[s] = eligible.size

                np.random.seed(args.seed + s)
                draw = (np.random.choice(eligible, size=n_samples, replace=False)
                        if eligible.size > n_samples else eligible)
                n_taken[s] = draw.size
                selected.update(draw.tolist())
                if s % 500 == 0:
                    print(f"  species {s}/{num_species}", flush=True)
    finally:
        if mask_file is not None:
            mask_file.close()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.save(out_path, np.array(sorted(selected), dtype=np.int32))

    print(f"\nsaved {len(selected)} unique rows -> {out_path}")
    print(f"species at the full quota of {n_samples}: {(n_taken == n_samples).sum()}/{num_species}")
    print(f"species short of quota (all eligible cells taken): {(n_taken < n_samples).sum()}")
    if mask_file is not None:
        print(f"species with zero in-range occurrences: {(n_eligible == 0).sum()}")
        print(f"occurrence cells excluded as out-of-range: {int((n_occ - n_eligible).sum())} "
              f"of {int(n_occ.sum())}")


if __name__ == "__main__":
    main()
