import os
import h5py
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.strtree import STRtree
from collections import defaultdict
from multiprocessing import Pool
from tqdm import tqdm
import warnings
import argparse
import yaml
import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
root = os.getenv("PROJECT_ROOT")

with open(f"{root}/configs/local_default.yaml") as stream:
    data_dir = yaml.safe_load(stream)["data_dir"]

with open(f"{root}/configs/data/datamodule.yaml") as stream:
    data_config = yaml.safe_load(stream)
    _aggregated = data_config.get("data", {}).get("init_args", {}).get("aggregated", True)
    SUFFIX = "_1km" if _aggregated else ""


POWO_RANGES = f"{data_dir}/native_ranges"
SPECIES_CSV = f"{data_dir}/targets/species_names.csv"
L3_SHP = f"{data_dir}/maps/level3/level3.shp"

GBIF_LOC = f"{data_dir}/predictors/train_location{SUFFIX}.h5"
SPLOT_LOC = f"{data_dir}/predictors/eval_location.h5"
BG_LOC = f"{data_dir}/predictors/background_location.h5"

SPECIES_BUFFER_SIZE = 1_000_000
LENGTHS_BUFFER_SIZE = 200_000

# Try to use hdf5plugin for blosc/lz4 (much faster decompression than gzip)
try:
    import hdf5plugin
    _BLOSC_AVAILABLE = True
except ImportError:
    _BLOSC_AVAILABLE = False

def _get_compression_kwargs():
    """Return best available compression settings for HDF5 dataset creation."""
    if _BLOSC_AVAILABLE:
        return hdf5plugin.Blosc(cname='lz4', clevel=5, shuffle=hdf5plugin.Blosc.SHUFFLE)
    return {"compression": "gzip", "compression_opts": 1}

def _read_species_shapefile(args):
    sidx, species_name, shp_path = args
    try:
        gdf = gpd.read_file(shp_path)
        gdf = gdf[~gdf.geometry.is_empty]
        if gdf.empty:
            return None
        geom = gdf.geometry.union_all()
        return int(sidx), geom
    except Exception as e:
        warnings.warn(f"Failed reading {shp_path} ({species_name}): {e}")
        return None

def load_l3_areas(path):
    gdf = gpd.read_file(path)
    gdf["buffered"] = gdf.geometry.buffer(-0.001) #NOTE: Buffer to avoid neighbor issues
    gdf.loc[gdf["buffered"].is_empty, "buffered"] = gdf.loc[
        gdf["buffered"].is_empty, "geometry"
    ]

    geoms = list(gdf.geometry.values)
    tree = STRtree(geoms)
    idx_to_code = dict(enumerate(gdf["LEVEL3_COD"].tolist()))
    return gdf, tree, geoms, idx_to_code


def build_species_geometries(powo_dir, species_df, gdf_l3, workers):
    tasks = []
    for sidx in range(len(species_df)):
        name = species_df.iloc[sidx]["Species Name"]
        shp = os.path.join(powo_dir, f"{name}.shp")
        tasks.append((sidx, name, shp))

    results = []
    if workers > 1:
        with Pool(workers) as p:
            for res in tqdm(
                p.imap_unordered(_read_species_shapefile, tasks),
                total=len(tasks),
                desc="Reading species ranges",
            ):
                if res is not None:
                    results.append(res)
    else:
        for t in tqdm(tasks, desc="Reading species ranges"):
            r = _read_species_shapefile(t)
            if r is not None:
                results.append(r)

    if not results:
        raise RuntimeError("No species geometries read.")

    sids, geoms = zip(*results)
    return gpd.GeoDataFrame(
        {"species_idx": list(sids)},
        geometry=list(geoms),
        crs=gdf_l3.crs,
    )


def map_species_to_areas(gdf_species, gdf_l3):
    gdf_l3_buf = gpd.GeoDataFrame(
        gdf_l3[["LEVEL3_COD", "buffered"]],
        geometry="buffered",
        crs=gdf_l3.crs,
    )

    joined = gpd.sjoin(
        gdf_species, gdf_l3_buf, how="left", predicate="intersects"
    )

    species_to_areas = defaultdict(set)
    area_to_species = defaultdict(list)

    for row in joined.itertuples():
        if pd.notna(row.LEVEL3_COD):
            species_to_areas[int(row.species_idx)].add(row.LEVEL3_COD)

    for sidx, areas in species_to_areas.items():
        for a in areas:
            area_to_species[a].append(sidx)

    return species_to_areas, area_to_species


def assign_area_codes(lon_arr, lat_arr, tree, idx_to_code):
    import shapely
    pts = shapely.points(lon_arr, lat_arr)
    n = len(pts)

    # Build integer code lookup
    # idx_to_code maps polygon index -> string code
    # We work with polygon indices first, convert to strings at the end
    poly_idx_arr = np.full(n, -1, dtype=np.int32)

    hit_pt, hit_poly = tree.query(pts, predicate="within")
    if len(hit_pt) > 0:
        poly_idx_arr[hit_pt] = hit_poly

    missing = np.where(poly_idx_arr == -1)[0]
    if len(missing):
        nearest = tree.nearest(pts[missing])
        poly_idx_arr[missing] = nearest

    # Convert polygon indices to area code strings
    codes = np.array([idx_to_code[pi] for pi in poly_idx_arr], dtype=object)
    return codes

_WORKER = {}


def _init_area_worker(l3_shp, locations_h5):
    """Build the STRtree once per worker; h5py handles are opened after the fork."""
    gdf, tree, _, idx_to_code = load_l3_areas(l3_shp)
    _WORKER["tree"] = tree
    _WORKER["idx_to_code"] = idx_to_code
    _WORKER["lf"] = h5py.File(locations_h5, "r")


def _area_codes_for_span(span):
    """Polygon index per location for one [start, end) span of the location table."""
    import shapely
    start, end = span
    chunk = _WORKER["lf"]["predictor_data"][start:end]
    pts = shapely.points(chunk[:, 0], chunk[:, 1])
    out = np.full(len(pts), -1, dtype=np.int32)
    hit_pt, hit_poly = _WORKER["tree"].query(pts, predicate="within")
    if len(hit_pt) > 0:
        out[hit_pt] = hit_poly
    missing = np.where(out == -1)[0]
    if len(missing):
        out[missing] = _WORKER["tree"].nearest(pts[missing])
    return start, out


def compute_area_codes(locations_h5, l3_shp, num_locations, chunk_size, num_workers):
    """Assign every location its L3 polygon index, in parallel.

    The geometry query dominates mask building and both downstream passes used to redo it,
    so it is computed once here and shared. Returns polygon indices rather than code strings
    to keep 30M-row runs cheap; callers map through idx_to_code.
    """
    spans = [(s, min(num_locations, s + chunk_size))
             for s in range(0, num_locations, chunk_size)]
    poly_idx = np.empty(num_locations, dtype=np.int32)

    if num_workers > 1 and len(spans) > 1:
        with Pool(num_workers, initializer=_init_area_worker,
                  initargs=(l3_shp, locations_h5)) as pool:
            for start, arr in tqdm(pool.imap_unordered(_area_codes_for_span, spans),
                                   total=len(spans), desc="Assigning L3 areas"):
                poly_idx[start:start + arr.size] = arr
    else:
        _init_area_worker(l3_shp, locations_h5)
        for span in tqdm(spans, desc="Assigning L3 areas"):
            start, arr = _area_codes_for_span(span)
            poly_idx[start:start + arr.size] = arr
    return poly_idx


def _process_chunks_location_indexed(
    lf, num_locations, chunk_size, poly_idx, codes_arr, area_to_species_arr,
    species_index_dset, lengths_dset, cumulative_dset,
    species_counts=None,
):
    """Single pass: builds location-indexed data and optionally counts per-species lengths."""
    species_buf, lengths_buf, cumulative_buf = [], [], []
    current_nnz = 0

    for start in tqdm(range(0, num_locations, chunk_size), desc="Pass 1: location-indexed + counting"):
        end = min(num_locations, start + chunk_size)
        area_codes = codes_arr[poly_idx[start:end]]

        for local_idx, ac in enumerate(area_codes):
            species_list = area_to_species_arr.get(ac)
            if species_list is None:
                lengths_buf.append(0)
                cumulative_buf.append(current_nnz)
                continue

            species_buf.extend(species_list.tolist())
            L = len(species_list)
            current_nnz += L
            lengths_buf.append(L)
            cumulative_buf.append(current_nnz)

        # Vectorized species counting for the whole chunk
        if species_counts is not None:
            all_species_for_chunk = []
            for ac in area_codes:
                species_list = area_to_species_arr.get(ac)
                if species_list is not None:
                    all_species_for_chunk.append(species_list)
            if all_species_for_chunk:
                concat = np.concatenate(all_species_for_chunk)
                np.add.at(species_counts, concat, 1)

        if len(species_buf) >= SPECIES_BUFFER_SIZE:
            old = species_index_dset.shape[0]
            species_index_dset.resize((old + len(species_buf),))
            species_index_dset[old:] = species_buf
            species_buf.clear()

        if len(lengths_buf) >= LENGTHS_BUFFER_SIZE:
            old = lengths_dset.shape[0]
            lengths_dset.resize((old + len(lengths_buf),))
            lengths_dset[old:] = lengths_buf

            oldc = cumulative_dset.shape[0]
            cumulative_dset.resize((oldc + len(cumulative_buf),))
            cumulative_dset[oldc:] = cumulative_buf

            lengths_buf.clear()
            cumulative_buf.clear()

    # Flush remaining buffers
    if species_buf:
        old = species_index_dset.shape[0]
        species_index_dset.resize((old + len(species_buf),))
        species_index_dset[old:] = species_buf
    if lengths_buf:
        old = lengths_dset.shape[0]
        lengths_dset.resize((old + len(lengths_buf),))
        lengths_dset[old:] = lengths_buf
        oldc = cumulative_dset.shape[0]
        cumulative_dset.resize((oldc + len(cumulative_buf),))
        cumulative_dset[oldc:] = cumulative_buf

    return current_nnz


def _count_species_lengths(
    lf, num_locations, chunk_size, poly_idx, codes_arr, area_to_species_arr,
    num_species,
):
    """Count-only pass: tallies how many locations fall in each species range.

    Also caches the area code for every location as an encoded int16 array
    so that Pass 2 can skip the expensive geometry queries entirely.
    """
    species_counts = np.zeros(num_species, dtype=np.int64)

    # Build string→int encoding for area codes (TDWG L3 has ~369 regions)
    code_to_int = {}
    int_to_code = []
    MISSING_CODE = -1

    # Cache area codes as int16 per location (~60 MB for 30M locations)
    area_code_cache = np.full(num_locations, MISSING_CODE, dtype=np.int16)

    for start in tqdm(range(0, num_locations, chunk_size), desc="Pass 1: counting species lengths"):
        end = min(num_locations, start + chunk_size)
        area_codes = codes_arr[poly_idx[start:end]]

        # Encode area codes as integers and cache them
        for local_idx, ac in enumerate(area_codes):
            ac_int = code_to_int.get(ac)
            if ac_int is None:
                ac_int = len(int_to_code)
                code_to_int[ac] = ac_int
                int_to_code.append(ac)
            area_code_cache[start + local_idx] = ac_int

        # Vectorized species counting: gather all species indices, then bulk-add
        all_species_for_chunk = []
        for ac in area_codes:
            species_list = area_to_species_arr.get(ac)
            if species_list is not None:
                all_species_for_chunk.append(species_list)
        if all_species_for_chunk:
            concat = np.concatenate(all_species_for_chunk)
            np.add.at(species_counts, concat, 1)

    # Build int-encoded area_to_species lookup for Pass 2
    area_int_to_species = {}
    for ac_str, ac_int in code_to_int.items():
        species_list = area_to_species_arr.get(ac_str)
        if species_list is not None:
            area_int_to_species[ac_int] = species_list

    return species_counts, area_code_cache, area_int_to_species


def _write_species_indexed(
    num_locations, chunk_size, area_code_cache, area_int_to_species,
    out_species_indexed_h5, species_counts, num_species,
):
    """Pass 2: writes location indices directly into a compressed HDF5.

    Uses the cached area codes from Pass 1 (int16 array) so no geometry
    queries are needed. Processes species in batches to keep memory bounded.
    Each batch scans the cached area codes (pure numpy/dict lookups, very fast).
    """
    lengths_arr = species_counts.astype(np.int32)
    total = int(species_counts.sum())
    offsets = np.concatenate([[0], np.cumsum(species_counts)]).astype(np.int64)

    SPECIES_BATCH = 500  # can be larger now since scanning is cheap

    # Build reverse lookup: for each area code int, which species as a set
    area_int_to_species_set = {
        k: set(v.tolist()) for k, v in area_int_to_species.items()
    }

    with h5py.File(out_species_indexed_h5, "w") as f:
        # Use ~256KB chunks for good random-access + compression balance
        chunk_elems = max(1024, 256 * 1024 // 4)  # 65536 int32 elements = 256 KB
        comp_kwargs = _get_compression_kwargs()
        dset = f.create_dataset(
            "target_location_indices",
            shape=(total,),
            dtype="int32",
            chunks=(min(total, chunk_elems),),
            **comp_kwargs,
        )

        num_batches = (num_species + SPECIES_BATCH - 1) // SPECIES_BATCH

        for batch_start in tqdm(
            range(0, num_species, SPECIES_BATCH),
            total=num_batches,
            desc="Pass 2: writing species index",
        ):
            batch_end = min(num_species, batch_start + SPECIES_BATCH)
            batch_species = set(range(batch_start, batch_end))

            # Pre-filter: which encoded area codes contain species in this batch?
            relevant_areas = {}
            for ac_int, species_set in area_int_to_species_set.items():
                overlap = species_set & batch_species
                if overlap:
                    relevant_areas[ac_int] = np.array(sorted(overlap), dtype=np.int32)

            if not relevant_areas:
                continue

            # Pre-allocate numpy arrays per species (4 bytes per entry, not 28)
            species_arrays = {}
            species_write_pos = {}
            for sidx in range(batch_start, batch_end):
                count = int(species_counts[sidx])
                if count > 0:
                    species_arrays[sidx] = np.empty(count, dtype=np.int32)
                    species_write_pos[sidx] = 0

            if not species_arrays:
                continue

            # Scan cached area codes in chunks (vectorized with numpy)
            relevant_ac_ints = np.array(list(relevant_areas.keys()), dtype=np.int16)
            for start in range(0, num_locations, chunk_size):
                end = min(num_locations, start + chunk_size)
                chunk_codes = area_code_cache[start:end]

                # Find locations matching any relevant area code (vectorized)
                for ac_int in relevant_ac_ints:
                    mask = chunk_codes == ac_int
                    if not mask.any():
                        continue
                    local_indices = np.nonzero(mask)[0]
                    global_indices = (start + local_indices).astype(np.int32)
                    batch_species_in_area = relevant_areas[ac_int]
                    for sidx in batch_species_in_area:
                        sidx_int = int(sidx)
                        arr = species_arrays.get(sidx_int)
                        if arr is not None:
                            pos = species_write_pos[sidx_int]
                            n_hits = len(global_indices)
                            arr[pos:pos + n_hits] = global_indices
                            species_write_pos[sidx_int] = pos + n_hits

            # Write this batch contiguously to the HDF5 dataset
            for sidx in range(batch_start, batch_end):
                arr = species_arrays.get(sidx)
                if arr is not None:
                    pos = species_write_pos[sidx]
                    offset = int(offsets[sidx])
                    dset[offset : offset + pos] = arr[:pos]

            del species_arrays, species_write_pos

        comp_kwargs_small = _get_compression_kwargs()
        f.create_dataset("lengths", data=lengths_arr,
                         chunks=(min(len(lengths_arr), 65536),), **comp_kwargs_small)
        f.create_dataset(
            "target_location_indices_lookup",
            data=offsets,
            chunks=(min(len(offsets), 32768),),
            **comp_kwargs_small,
        )
        f.attrs["num_species"] = num_species
        f.attrs["num_locations"] = num_locations
        f.attrs["total_location_entries"] = total


def build_indices(
    locations_h5,
    out_location_indexed_h5,
    out_species_indexed_h5,
    powo_dir,
    species_csv,
    l3_shp,
    chunk_size=100_000,
    num_workers=16,
    force=False,
    build_location_indexed=True,
    build_species_indexed=True,
):
    if (not force) and build_location_indexed and os.path.exists(out_location_indexed_h5):
        build_location_indexed = False
    if (not force) and build_species_indexed and os.path.exists(out_species_indexed_h5):
        build_species_indexed = False
    if not (build_location_indexed or build_species_indexed):
        return out_location_indexed_h5, out_species_indexed_h5

    species_df = pd.read_csv(species_csv)
    num_species = len(species_df)

    gdf_l3, l3_tree, _, l3_idx_to_code = load_l3_areas(l3_shp)
    gdf_species = build_species_geometries(
        powo_dir, species_df, gdf_l3, num_workers
    )

    _, area_to_species = map_species_to_areas(gdf_species, gdf_l3)

    area_to_species_arr = {
        k: np.asarray(v, dtype=np.int32)
        for k, v in area_to_species.items()
    }

    with h5py.File(locations_h5, "r") as f:
        num_locations = int(f["predictor_data"].shape[0])

    # The point-in-polygon assignment dominates runtime and both passes below need it, so do
    # it once, in parallel, and share the result.
    poly_idx = compute_area_codes(locations_h5, l3_shp, num_locations, chunk_size, num_workers)
    codes_arr = np.array([l3_idx_to_code[i] for i in range(len(l3_idx_to_code))], dtype=object)

    if build_location_indexed:
        f_loc = h5py.File(out_location_indexed_h5, "w")
        loc_comp = _get_compression_kwargs()
        species_index_dset = f_loc.create_dataset(
            "species_index", shape=(0,), maxshape=(None,),
            dtype="int32", chunks=(65536,), **loc_comp,
        )
        lengths_dset = f_loc.create_dataset(
            "lengths", shape=(0,), maxshape=(None,),
            dtype="int32", chunks=(65536,), **loc_comp,
        )
        cumulative_dset = f_loc.create_dataset(
            "cumulative_lengths", shape=(1,), maxshape=(None,),
            dtype="int64", chunks=(32768,), **loc_comp,
        )
        cumulative_dset[0] = 0

        # If we also need species-indexed, piggyback counting onto this pass
        species_counts = np.zeros(num_species, dtype=np.int64) if build_species_indexed else None

        with h5py.File(locations_h5, "r") as lf:
            current_nnz = _process_chunks_location_indexed(
                lf, num_locations, chunk_size, poly_idx, codes_arr,
                area_to_species_arr, species_index_dset, lengths_dset,
                cumulative_dset, species_counts=species_counts,
            )

        f_loc.attrs["num_locations"] = num_locations
        f_loc.attrs["num_species"] = num_species
        f_loc.attrs["total_species_entries"] = current_nnz
        f_loc.close()
    else:
        species_counts = None

    if build_species_indexed:
        with h5py.File(locations_h5, "r") as lf:
            # Pass 1: count + cache area codes
            species_counts, area_code_cache, area_int_to_species = (
                _count_species_lengths(
                    lf, num_locations, chunk_size, poly_idx, codes_arr,
                    area_to_species_arr, num_species,
                )
            )

            # Pass 2: write using cached area codes (no geometry queries)
            _write_species_indexed(
                num_locations, chunk_size, area_code_cache,
                area_int_to_species, out_species_indexed_h5, species_counts,
                num_species,
            )

    return out_location_indexed_h5, out_species_indexed_h5


def run_range_mask_builder(
    dataset="all",
    force=False,
    chunk_size=1_000_000,
    num_workers=16,
    mode="both",
    data_dir_override=None,
    out_dir_override=None,
    powo_dir_override=None,
    train_suffix_override=None,
):
    _data_dir = data_dir_override or data_dir

    _suffix = SUFFIX if train_suffix_override is None else train_suffix_override
    configs = {
        "train": f"{_data_dir}/predictors/train_location{_suffix}.h5",
        "eval": f"{_data_dir}/predictors/eval_location.h5",
        "background": f"{_data_dir}/predictors/background_location.h5",
    }

    selected = configs.items() if dataset == "all" else [(dataset, configs[dataset])]

    out_dir = out_dir_override or f"{_data_dir}/range_masks{SUFFIX}"
    powo_dir = powo_dir_override or f"{_data_dir}/native_ranges"
    os.makedirs(out_dir, exist_ok=True)

    for tag, loc_path in selected:
        build_indices(
            locations_h5=loc_path,
            out_location_indexed_h5=f"{out_dir}/{tag}_range_mask.h5",
            out_species_indexed_h5=f"{out_dir}/{tag}_species_index.h5",
            powo_dir=powo_dir,
            species_csv=f"{_data_dir}/targets/species_names.csv",
            l3_shp=f"{_data_dir}/maps/level3/level3.shp",
            chunk_size=chunk_size,
            num_workers=num_workers,
            force=force,
            build_location_indexed=mode in ("location", "both"),
            build_species_indexed=mode in ("species", "both"),
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["train", "eval", "background", "all"], default="all")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--chunk-size", type=int, default=100_000)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--mode", choices=["location", "species", "both"], default="both")
    parser.add_argument("--out_dir", default=None,
                        help="Write the masks here instead of <data_dir>/range_masks<SUFFIX>")
    parser.add_argument("--powo_dir", default=None,
                        help="Native-range shapefiles to build from "
                             "(default: <data_dir>/native_ranges)")
    parser.add_argument("--train_suffix", default=None,
                        help="Overrides the train-location suffix taken from "
                             "datamodule.yaml's `aggregated` ('_1km' or '')")
    args = parser.parse_args()

    run_range_mask_builder(
        dataset=args.dataset,
        force=args.force,
        chunk_size=args.chunk_size,
        num_workers=args.num_workers,
        mode=args.mode,
        out_dir_override=args.out_dir,
        powo_dir_override=args.powo_dir,
        train_suffix_override=args.train_suffix,
    )


if __name__ == "__main__":
    main()
