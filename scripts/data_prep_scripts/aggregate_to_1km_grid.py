import os
import h5py
import hdf5plugin  # noqa: F401  (registers Blosc etc.; some raw predictor h5 use it)
import numpy as np
import scipy.sparse as sp
import yaml
import argparse
from tqdm import tqdm
import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from geo_utils import latlon_to_grid_cell, grid_cell_to_latlon, RESOLUTION_M
root = os.getenv("PROJECT_ROOT")

with open(f"{root}/configs/local_default.yaml") as stream:
    data_dir = yaml.safe_load(stream)["data_dir"]

with open(f"{root}/configs/data/datamodule.yaml") as stream:
    data_config = yaml.safe_load(stream)
    _aggregated = data_config.get("data", {}).get("init_args", {}).get("aggregated", True)
    SUFFIX = "_1km" if _aggregated else ""

    # Get predictors from config (nested under data.init_args)
    predictors_subset = data_config.get("data", {}).get("init_args", {}).get("predictors_subset", {})

    # Remove 'location' if present
    predictor_keys = [k for k in predictors_subset.keys() if k != "location"]

def aggregate_to_1km_grid(
    location_path,
    target_path,
    output_location_path,
    output_target_path,
    predictor_paths=None,
    output_predictor_paths=None,
):
    with h5py.File(location_path, "r") as f:
        locations = f["predictor_data"][:]

    with h5py.File(target_path, "r") as f:
        dset = f["csr"]
        targets = sp.csr_matrix(
            (dset["data"][:], dset["indices"][:], dset["indptr"][:]),
            shape=tuple(dset.attrs["shape"]),
        )

    print(f"Original: {locations.shape[0]:,} locations, {targets.shape}")

    grid_x, grid_y = latlon_to_grid_cell(locations[:, 0], locations[:, 1])
    grid_xy = np.stack([grid_x, grid_y], axis=1)

    unique_cells, inverse_indices = np.unique(grid_xy, axis=0, return_inverse=True)
    n_unique = unique_cells.shape[0]
    sort_idx = np.argsort(inverse_indices)
    sorted_inverse = inverse_indices[sort_idx]
    unique_counts = np.bincount(sorted_inverse)
    cell_ranges = np.concatenate([[0], np.cumsum(unique_counts)])

    print(
        f"Unique cells: {n_unique:,} "
        f"({100 * (1 - n_unique / len(locations)):.1f}% reduction)"
    )

    center_lon, center_lat = grid_cell_to_latlon(unique_cells[:, 0], unique_cells[:, 1])
    unique_grid_cells = np.column_stack([center_lon, center_lat])

    print("\nAggregating targets...")
    targets_sorted = targets[sort_idx, :]
    aggregated_data = []
    aggregated_indices = []
    aggregated_indptr = [0]

    for cell_idx in tqdm(range(n_unique), desc="Processing cells"):
        start, end = cell_ranges[cell_idx], cell_ranges[cell_idx + 1]
        cell_targets = targets_sorted[start:end, :]
        merged = (cell_targets.sum(axis=0) > 0).A1.astype(np.int8)
        nonzero = np.flatnonzero(merged)
        aggregated_data.extend([1] * len(nonzero))
        aggregated_indices.extend(nonzero)
        aggregated_indptr.append(aggregated_indptr[-1] + len(nonzero))

    aggregated_targets = sp.csr_matrix(
        (
            np.asarray(aggregated_data, dtype=np.int8),
            np.asarray(aggregated_indices, dtype=np.int32),
            np.asarray(aggregated_indptr, dtype=np.int64),
        ),
        shape=(n_unique, targets.shape[1]),
    )


    with h5py.File(output_location_path, "w") as f:
        loc_data = np.asarray(unique_grid_cells, dtype=np.float32)
        nrows = loc_data.shape[0]
        ncols = loc_data.shape[1] if loc_data.ndim > 1 else 1
        chunk_shape = (min(1024, nrows), ncols) if loc_data.ndim > 1 else (min(1024, nrows),)
        f.create_dataset(
            "predictor_data", data=loc_data, chunks=chunk_shape,
            compression="gzip", compression_opts=1,
        )
        f.attrs["grid_resolution_m"] = RESOLUTION_M
        f.attrs["projection"] = "EPSG:6933"
        f.attrs["original_n_locations"] = len(locations)

    with h5py.File(output_target_path, "w") as f:
        grp = f.create_group("csr")
        grp.create_dataset("data", data=aggregated_targets.data, compression="gzip")
        grp.create_dataset("indices", data=aggregated_targets.indices, compression="gzip")
        grp.create_dataset("indptr", data=aggregated_targets.indptr, compression="gzip")
        grp.attrs["shape"] = aggregated_targets.shape

    if predictor_paths:
        print("\nAggregating predictors...")
        for name, input_path in predictor_paths.items():
            with h5py.File(input_path, "r") as f:
                data = f["predictor_data"][:]
                original_dtype = data.dtype
            data_sorted = data[sort_idx, :]
            aggregated = np.zeros((n_unique, data.shape[1]), dtype=original_dtype)
            for i in range(n_unique):
                start, end = cell_ranges[i], cell_ranges[i + 1]
                aggregated[i] = data_sorted[start:end].mean(axis=0)
            with h5py.File(output_predictor_paths[name], "w") as f:
                agg_f32 = aggregated.astype(np.float32)
                nrows = agg_f32.shape[0]
                ncols = agg_f32.shape[1] if agg_f32.ndim > 1 else 1
                chunk_shape = (min(1024, nrows), ncols) if agg_f32.ndim > 1 else (min(1024, nrows),)
                f.create_dataset(
                    "predictor_data",
                    data=agg_f32,
                    chunks=chunk_shape,
                    compression="gzip",
                    compression_opts=1,
                )
                f.attrs["grid_resolution_m"] = RESOLUTION_M
            del data, data_sorted, aggregated

def aggregate_predictors_onto_reference(raw_location_path, reference_grid_path,
                                        raw_predictor_paths, output_predictor_paths):
    """Aggregate raw per-occurrence predictor(s) onto a fixed 1 km grid.

    Each raw occurrence is snapped to its cell in ``reference_grid_path`` ,
    values are nan-averaged per cell.
    """
    with h5py.File(raw_location_path, "r") as f:
        loc = f["predictor_data"][:]
    with h5py.File(reference_grid_path, "r") as f:
        ref = f["predictor_data"][:]
    n = ref.shape[0]
    rgx, rgy = latlon_to_grid_cell(loc[:, 0], loc[:, 1])
    egx, egy = latlon_to_grid_cell(ref[:, 0], ref[:, 1])
    gmx = min(int(egx.min()), int(rgx.min())); gmy = min(int(egy.min()), int(rgy.min()))
    sy = int(max(int(egy.max()), int(rgy.max())) - gmy) + 1

    def _key(gx, gy):                       # encode (gx, gy) -> one non-negative int64
        return (gx.astype(np.int64) - gmx) * sy + (gy.astype(np.int64) - gmy)

    ek = _key(egx, egy)
    assert len(np.unique(ek)) == n, "reference grid cells not unique under key encoding"
    order = np.argsort(ek); eks = ek[order]
    rk = _key(rgx, rgy)
    pos = np.clip(np.searchsorted(eks, rk), 0, n - 1)
    found = eks[pos] == rk
    cell = np.where(found, order[pos], -1)
    print(f"Reference grid: {n:,} fixed cells | mapped {int(found.sum()):,}/{len(rk):,} occurrences "
          f"({int((~found).sum()):,} boundary orphans dropped)")
    ci = cell[found]
    for name, ip in raw_predictor_paths.items():
        with h5py.File(ip, "r") as f:
            data = f["predictor_data"][:][found]
        agg = np.full((n, data.shape[1]), np.nan, np.float32)
        for j in range(data.shape[1]):
            col = data[:, j].astype(np.float64); ok = np.isfinite(col)
            cnt = np.bincount(ci[ok], minlength=n)
            s = np.bincount(ci[ok], weights=col[ok], minlength=n)
            agg[:, j] = np.where(cnt > 0, s / np.maximum(cnt, 1), np.nan)
        with h5py.File(output_predictor_paths[name], "w") as f:
            f.create_dataset("predictor_data", data=agg, chunks=(min(1024, n), agg.shape[1]),
                             compression="gzip", compression_opts=1)
            f.attrs["grid_resolution_m"] = RESOLUTION_M
            f.attrs["reference_grid"] = os.path.basename(reference_grid_path)
        print(f"  SAVED {output_predictor_paths[name]}  {agg.shape}  "
              f"all-nan (no-data) cells: {int(np.isnan(agg).all(1).sum()):,}", flush=True)
        del data, agg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["gbif", "splotopen", "both"], default="both")
    parser.add_argument("--force", action="store_true")
    # Robust "extend the predictor set" mode: snap a raw predictor onto the shipped
    # 1km grid instead of recomputing cells (PROJ-version-stable). See the function.
    parser.add_argument("--reference-grid", default=None,
                        help="Shipped *_location_1km.h5 to snap onto (needs --raw-location/-predictor/--output).")
    parser.add_argument("--raw-location", default=None, help="raw per-occurrence *_location.h5")
    parser.add_argument("--raw-predictor", default=None, help="raw per-occurrence predictor .h5 to aggregate")
    parser.add_argument("--output", default=None, help="output aggregated predictor .h5")
    args = parser.parse_args()

    if args.reference_grid:
        assert args.raw_location and args.raw_predictor and args.output, \
            "--reference-grid requires --raw-location, --raw-predictor and --output"
        aggregate_predictors_onto_reference(
            args.raw_location, args.reference_grid,
            {"pred": args.raw_predictor}, {"pred": args.output})
        return


    configs = {
        "gbif": {
            "location": f"{data_dir}/predictors/train_location.h5",
            "target": f"{data_dir}/targets/train_targets.h5",
            "output_location": f"{data_dir}/predictors/train_location{SUFFIX}.h5",
            "output_target": f"{data_dir}/targets/train_targets{SUFFIX}.h5",
        },
        "splotopen": {
            "location": f"{data_dir}/predictors/eval_location.h5",
            "target": f"{data_dir}/targets/eval_targets.h5",
            "output_location": f"{data_dir}/predictors/eval_location{SUFFIX}.h5",
            "output_target": f"{data_dir}/targets/eval_targets{SUFFIX}.h5",
        },
    }

    if not _aggregated:
        print("Aggregation is disabled (aggregated: false in config). Nothing to do.")
        return

    datasets = ["gbif", "splotopen"] if args.dataset == "both" else [args.dataset]

    dataset_prefixes = {
        "gbif": "train",
        "splotopen": "eval",
    }

    for ds in datasets:
        prefix = dataset_prefixes[ds]
        predictor_paths = {}
        output_predictor_paths = {}
        for key in predictor_keys:
            predictor_paths[key] = f"{data_dir}/predictors/{prefix}_{key}.h5"
            output_predictor_paths[key] = f"{data_dir}/predictors/{prefix}_{key}{SUFFIX}.h5"

        conf = configs[ds]

        if (
            not args.force
            and os.path.exists(conf["output_location"])
            and os.path.exists(conf["output_target"])
        ):
            print(f"Aggregated files for {ds} already exist. Use --force to overwrite.")
            continue

        aggregate_to_1km_grid(
            conf["location"],
            conf["target"],
            conf["output_location"],
            conf["output_target"],
            predictor_paths=predictor_paths,
            output_predictor_paths=output_predictor_paths,
        )

if __name__ == "__main__":
    main()
