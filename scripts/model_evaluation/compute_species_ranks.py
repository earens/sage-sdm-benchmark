"""Compute per-species stratification proxies for the SDM evaluation.

Writes one row per species to ``<data_dir>/targets/species_properties{_1km}{_sub}.csv``,
holding the sampling descriptors used to stratify the paper's
evaluation analyses (the delta heatmaps and geographic-performance plots). Run
this *after* the data-prep pipeline (targets → predictors → masks) and *before*
the stratified evaluation / plotting steps; those consume ``species_properties*.csv`` via
``src/evaluation/stratify.py`` and ``src/visualizations/`` (delta heatmaps, maps),
as well as ``scripts/evaluate_predictions.py``.

Key proxy columns:
    sampling_effort: ``n_cells_visited / n_cells_in_range * 100`` — how thoroughly
        a species' native range was sampled (x-axis of the delta heatmaps).
    relative_prevalence: ``n_species_obs / n_cells_visited * 100`` — how often the
        species is recorded in the cells where it was actually sampled (y-axis of
        the delta heatmaps).

Also copies constant per-species metadata (``range_size_km2``, plot counts,
``test_prevalence``, WCVP taxonomy, lifeform / growth form) from any sibling
``species_properties*.csv`` when present (see ``SHARED_COLUMNS``).

Inputs (paths resolved from ``configs/local_default.yaml`` + ``configs/data``):
    the aggregated 1 km targets HDF5, range masks, environmental predictors, and
    the WCVP names table. Aggregation / subsampling variants are selected via CLI
    flags and reflected in the output filename suffix.
"""

import argparse
import os
from pathlib import Path

import json

import geopandas as gpd
import h5py
import hdf5plugin  # noqa: F401 – registers HDF5 filter plugins
import numpy as np
import pandas as pd
import rasterio.features
import rasterio.transform
import rootutils
import scipy.sparse as sp
import yaml
from tqdm import tqdm


rootutils.setup_root(".", indicator=".project-root", pythonpath=True)
ROOT = os.getenv("PROJECT_ROOT")

with open(f"{ROOT}/configs/local_default.yaml") as _f:
    DATA_DIR: str = yaml.safe_load(_f)["data_dir"]

with open(f"{ROOT}/configs/data/datamodule.yaml") as _f:
    _dm_cfg = yaml.safe_load(_f)
    _init = _dm_cfg.get("data", {}).get("init_args", _dm_cfg)
    AGGREGATED: bool = _init.get("aggregated", True)
    SUBSAMPLE_FILE: str | None = _init.get("subsample_indices_file")

# Columns that are constant regardless of aggregation / subsampling and can be copied from any sibling proxies*.csv.
SHARED_COLUMNS = {
    "range_size_km2",
    "n_cells_in_range",
    "num_plots_in_range",
    "test_prevalence",
    "kingdom",
    "phylum",
    "class",
    "genus",
    "family",
    "order",
    "lifeform_description",
    "growth_form",
}

# Path to the WCVP names file (contains lifeform_description).
# Adjust if your raw POWO / WCVP data lives elsewhere.
_RAW_DATA_DIR = str(Path(DATA_DIR).parent / "data_raw")
WCVP_NAMES_PATH: str = os.path.join(_RAW_DATA_DIR, "powo", "wcvp_names.csv")


# Helper: load sparse targets from an HDF5 file if file fits in mem
def _load_sparse_targets(path: str | Path) -> sp.csr_matrix:
    with h5py.File(path, "r") as f:
        d = f["csr"]
        return sp.csr_matrix(
            (d["data"][:], d["indices"][:], d["indptr"][:]),
            shape=tuple(d.attrs["shape"]),
        )

def compute_train_prevalence(
    df: pd.DataFrame,
    data_dir: str,
    suffix: str,
    subsample_indices: np.ndarray | None,
) -> dict[str, np.ndarray]:
    targets = _load_sparse_targets(f"{data_dir}targets/train_targets{suffix}.h5")
    if subsample_indices is not None:
        targets = targets[subsample_indices, :]
    n = targets.shape[0]
    positives = np.asarray((targets > 0).sum(axis=0)).flatten()
    return {"train_prevalence": positives / n * 100}


def compute_test_prevalence(
    df: pd.DataFrame,
    data_dir: str,
) -> dict[str, np.ndarray]:
    targets = _load_sparse_targets(f"{data_dir}targets/eval_targets.h5")

    # Number of plots in range per species
    range_mask_path = f"{data_dir}range_masks_1km/eval_species_index.h5"
    with h5py.File(range_mask_path, "r") as f:
        points_in_range = f["lengths"][:]

    species_counts = np.asarray((targets > 0).sum(axis=0)).flatten()
    prevalence = np.where(
        points_in_range > 0,
        species_counts / points_in_range * 100,
        0.0,
    )
    return {"test_prevalence": prevalence}


def compute_range_sizes(
    df: pd.DataFrame,
    data_dir: str,
) -> dict[str, np.ndarray]:
    RESOLUTION = 1000
    range_dir = Path(data_dir) / "native_ranges"

    sizes_km2 = []
    cell_counts = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Range sizes + cell counts"):
        shp = range_dir / f"{row['species_name']}.shp"
        if not shp.exists():
            sizes_km2.append(np.nan)
            cell_counts.append(0)
            continue
        try:
            gdf = gpd.read_file(shp).to_crs("EPSG:6933")
            gdf = gdf[~gdf.geometry.is_empty & gdf.geometry.notnull()]

            # Polygon area
            area_km2 = gdf.geometry.area.sum() / 1e6
            sizes_km2.append(area_km2)

            # Tight bounding box aligned to the 1 km grid
            minx, miny, maxx, maxy = gdf.total_bounds
            # Snap to grid origins (floor / ceil)
            col_off = int(np.floor(minx / RESOLUTION))
            row_off = int(np.floor(miny / RESOLUTION))
            col_end = int(np.ceil(maxx / RESOLUTION))
            row_end = int(np.ceil(maxy / RESOLUTION))
            width = col_end - col_off
            height = row_end - row_off

            if width <= 0 or height <= 0:
                cell_counts.append(0)
                continue

            # Affine transform for this window
            transform = rasterio.transform.from_bounds(
                col_off * RESOLUTION,
                row_off * RESOLUTION,
                col_end * RESOLUTION,
                row_end * RESOLUTION,
                width, height,
            )

            # Rasterise: False inside polygon, True outside
            mask = rasterio.features.geometry_mask(
                gdf.geometry,
                out_shape=(height, width),
                transform=transform,
                invert=True,
                all_touched=True,
            )
            cell_counts.append(int(mask.sum()))

        except Exception as e:
            print(f"  Warning: {row['species_name']}: {e}")
            sizes_km2.append(np.nan)
            cell_counts.append(0)

    return {
        "range_size_km2": np.array(sizes_km2),
        "n_cells_in_range": np.array(cell_counts, dtype=np.int64),
    }


def compute_num_plots_in_range(
    df: pd.DataFrame,
    data_dir: str,
) -> dict[str, np.ndarray]:
    path = f"{data_dir}range_masks_1km/eval_species_index.h5"
    with h5py.File(path, "r") as f:
        return {"num_plots_in_range": f["lengths"][:]}


# ---------------------------------------------------------------------------
# Growth form classification
# ---------------------------------------------------------------------------

# Primary keyword → growth_form mapping.  Order matters: first match wins.
_GROWTH_FORM_RULES: list[tuple[str, str]] = [
    ("tree",     "tree"),
    ("bamboo",   "tree"),
    ("liana",    "shrub"),
    ("shrub",    "shrub"),     # also catches subshrub, climbing shrub, …
]
_DEFAULT_GROWTH_FORM = "herb"  # covers perennial, annual, geophyte, epiphyte, …


def _classify_growth_form(lifeform: str | float) -> str:
    """Map a WCVP lifeform_description string to one of {tree, shrub, herb}."""
    if not isinstance(lifeform, str):
        return "unknown"
    lf = lifeform.lower()
    for keyword, form in _GROWTH_FORM_RULES:
        if keyword in lf:
            return form
    return _DEFAULT_GROWTH_FORM


# ---------------------------------------------------------------------------
# Taxonomy lookup via GBIF backbone (pygbif)
# ---------------------------------------------------------------------------
# Results are cached to a JSON file so the API is only queried once per family.
_TAXONOMY_CACHE_PATH = os.path.join(
    os.path.dirname(__file__), "cache", "family_taxonomy_cache.json"
)

_TAXONOMY_LEVELS = ("kingdom", "phylum", "class", "order")


def _resolve_families_via_gbif(families: list[str]) -> dict[str, dict[str, str]]:
    """Query GBIF backbone for each *family* and return higher taxonomy.

    Returns ``{family_name: {"kingdom": …, "phylum": …, "class": …, "order": …}}``.
    Results are read from / written to a local JSON cache so network
    calls only happen for families not yet cached.
    """
    from pygbif import species as gbif_species  # lazy import

    # Load existing cache
    cache: dict[str, dict[str, str]] = {}
    if os.path.exists(_TAXONOMY_CACHE_PATH):
        with open(_TAXONOMY_CACHE_PATH) as fh:
            cache = json.load(fh)

    new_families = [f for f in families if f not in cache]
    if new_families:
        print(f"  Querying GBIF backbone for {len(new_families)} families …")
        for fam in tqdm(new_families, desc="GBIF taxonomy"):
            try:
                r = gbif_species.name_backbone(
                    name=fam, rank="family", kingdom="Plantae"
                )
                cache[fam] = {lvl: r.get(lvl, "unknown") for lvl in _TAXONOMY_LEVELS}
            except Exception as e:
                print(f"    Warning: GBIF lookup failed for {fam}: {e}")
                cache[fam] = {lvl: "unknown" for lvl in _TAXONOMY_LEVELS}

        os.makedirs(os.path.dirname(_TAXONOMY_CACHE_PATH), exist_ok=True)
        with open(_TAXONOMY_CACHE_PATH, "w") as fh:
            json.dump(cache, fh, indent=2, sort_keys=True)
        print(f"  Cached taxonomy to {_TAXONOMY_CACHE_PATH}")

    return cache


def _load_wcvp_accepted() -> pd.DataFrame:
    """Load WCVP accepted species with taxonomy + lifeform columns."""
    wcvp = pd.read_csv(
        WCVP_NAMES_PATH,
        delimiter="|",
        usecols=[
            "taxon_rank", "taxon_status", "taxon_name",
            "genus", "family", "lifeform_description",
        ],
    )
    return (
        wcvp[(wcvp.taxon_rank == "Species") & (wcvp.taxon_status == "Accepted")]
        .drop_duplicates(subset="taxon_name", keep="first")
    )


def compute_taxonomy(
    df: pd.DataFrame,
    data_dir: str,
) -> dict[str, np.ndarray]:
    """Look up genus, family, order (+ kingdom, phylum, class) via WCVP + GBIF."""
    accepted = _load_wcvp_accepted()

    merged = df[["species_name"]].merge(
        accepted[["taxon_name", "genus", "family"]],
        left_on="species_name",
        right_on="taxon_name",
        how="left",
    )

    # genus fallback: first word of species name
    genus = merged["genus"].values.copy()
    missing_genus = pd.isna(merged["genus"])
    if missing_genus.any():
        genus[missing_genus] = (
            df.loc[missing_genus, "species_name"].str.split().str[0].values
        )

    family = merged["family"].values

    # Resolve higher taxonomy (order, class, phylum, kingdom) via GBIF
    unique_families = [f for f in pd.unique(family) if isinstance(f, str)]
    fam_taxonomy = _resolve_families_via_gbif(unique_families)

    results: dict[str, np.ndarray] = {
        "genus": genus,
        "family": family,
    }
    for lvl in _TAXONOMY_LEVELS:
        results[lvl] = np.array([
            fam_taxonomy.get(f, {}).get(lvl, "unknown")
            if isinstance(f, str) else "unknown"
            for f in family
        ])

    # Summary
    n_fam = pd.notna(merged["family"]).sum()
    n_ord = (results["order"] != "unknown").sum()
    print(f"  genus:  {pd.notna(pd.Series(genus)).sum()}/{len(df)}")
    print(f"  family: {n_fam}/{len(df)} ({merged['family'].nunique()} unique)")
    print(f"  order:  {n_ord}/{len(df)} ({len(set(results['order']) - {'unknown'})} unique)")

    return results


def compute_growth_form(
    df: pd.DataFrame,
    data_dir: str,
) -> dict[str, np.ndarray]:
    """Look up WCVP lifeform_description and derive a coarse growth_form.

    Returns both the raw ``lifeform_description`` and the simplified
    ``growth_form`` (tree / shrub / herb / unknown).
    """
    accepted = _load_wcvp_accepted()

    merged = df[["species_name"]].merge(
        accepted[["taxon_name", "lifeform_description"]],
        left_on="species_name",
        right_on="taxon_name",
        how="left",
    )
    lifeform = merged["lifeform_description"].values
    growth_form = np.array([_classify_growth_form(lf) for lf in lifeform])

    n_matched = pd.notna(merged["lifeform_description"]).sum()
    print(f"  Matched {n_matched}/{len(df)} species to WCVP lifeform data")
    for gf in ["tree", "shrub", "herb", "unknown"]:
        print(f"    {gf}: {(growth_form == gf).sum()}")

    return {
        "lifeform_description": lifeform,
        "growth_form": growth_form,
    }


def compute_biases(
    df: pd.DataFrame,
    data_dir: str,
    suffix: str,
    aggregated: bool,
    subsample_indices: np.ndarray | None,
) -> dict[str, np.ndarray]:

    num_species = len(df)

    n_cells_in_range = df["n_cells_in_range"].values.astype(np.int64)

    agg_suffix = "_1km"
    range_mask_path = (
        Path(data_dir) / f"range_masks{agg_suffix}" / "train_species_index.h5"
    )
    subsample_set = set(subsample_indices) if subsample_indices is not None else None

    n_cells_visited = np.zeros(num_species, dtype=np.int64)
    with h5py.File(range_mask_path, "r") as f:
        idx_arr = f["target_location_indices"]
        lookup = f["target_location_indices_lookup"]
        for sid in tqdm(range(num_species), desc="Visited cells"):
            s, e = lookup[sid], lookup[sid + 1]
            if e <= s:
                continue
            locs = idx_arr[s:e]
            if subsample_set is not None:
                n_cells_visited[sid] = sum(1 for i in locs if i in subsample_set)
            else:
                n_cells_visited[sid] = len(locs)

    agg_targets = _load_sparse_targets(
        f"{data_dir}targets/train_targets{agg_suffix}.h5"
    )
    if subsample_indices is not None:
        agg_targets = agg_targets[subsample_indices, :]

    if not aggregated:
        raw_targets = _load_sparse_targets(
            f"{data_dir}targets/train_targets.h5"
        )
        if subsample_indices is not None:
            raw_targets = raw_targets[subsample_indices, :]
        n_species_obs = np.asarray(raw_targets.sum(axis=0)).flatten()
    else:
        n_species_obs = np.asarray(agg_targets.sum(axis=0)).flatten()

    sampling_effort = n_cells_visited / n_cells_in_range * 100.0
    relative_prevalence = n_species_obs / n_cells_visited * 100.0

    return {
        "sampling_effort": sampling_effort,
        "relative_prevalence": relative_prevalence,
        "n_cells_visited_in_range": n_cells_visited,
        "n_species_obs": n_species_obs,
    }



COLUMN_BLOCKS = [
    (
        "taxonomy (genus, family, order + higher via GBIF)",
        compute_taxonomy,
        ["data_dir"],
        ["genus", "family", "kingdom", "phylum", "class", "order"],
    ),
    (
        "growth form (WCVP lifeform)",
        compute_growth_form,
        ["data_dir"],
        ["lifeform_description", "growth_form"],
    ),
    (
        "train prevalence",
        compute_train_prevalence,
        ["data_dir", "suffix", "subsample_indices"],
        ["train_prevalence"],
    ),
    (
        "test prevalence",
        compute_test_prevalence,
        ["data_dir"],
        ["test_prevalence"],
    ),
    (
        "range sizes",
        compute_range_sizes,
        ["data_dir"],
        ["range_size_km2", "n_cells_in_range"],
    ),
    (
        "num plots in range",
        compute_num_plots_in_range,
        ["data_dir"],
        ["num_plots_in_range"],
    ),
    (
        "biases (sampling effort + taxonomic preference)",
        compute_biases,
        ["data_dir", "suffix", "aggregated", "subsample_indices"],
        [
            "sampling_effort",
            "relative_prevalence",
            "n_cells_visited_in_range",
            "n_species_obs",
        ],
    ),
]


#copy shared columns from other species_properties csv
def _copy_shared_columns_from_siblings(
    df: pd.DataFrame,
    csv_path: Path,
) -> tuple[pd.DataFrame, list[str]]:
    targets_dir = csv_path.parent
    missing = SHARED_COLUMNS - set(df.columns)
    if not missing:
        return df, []

    copied: list[str] = []
    for sibling in sorted(targets_dir.glob("species_properties*.csv")):
        if sibling == csv_path:
            continue
        df_sib = pd.read_csv(sibling)
        for col in list(missing):
            if col in df_sib.columns:
                df[col] = df_sib[col].values
                copied.append(col)
                missing.discard(col)
        if not missing:
            break

    return df, copied


def build_proxies(
    data_dir: str,
    aggregated: bool,
    subsample_file: str | None,
    add_only: bool,
    force: bool = False,
) -> pd.DataFrame:
    suffix = "_1km" if aggregated else ""

    subsample_indices: np.ndarray | None = None
    if subsample_file:
        path = Path(data_dir) / "targets" / subsample_file
        subsample_indices = np.load(path)

    #out path
    sub_tag = ""
    if subsample_file:
        name = Path(subsample_file).stem
        #split by _
        parts = name.split("_")[-1]
        sub_tag = f"_{parts}"

    agg_tag = "_1km" if aggregated else ""
    csv_path = Path(data_dir) / "targets" / f"species_properties{agg_tag}{sub_tag}.csv"

    #base data frame
    if not force and add_only and csv_path.exists():
        print(f"Loading existing CSV: {csv_path}")
        df = pd.read_csv(csv_path)
    else:
        df = pd.read_csv(Path(data_dir) / "targets" / "species_names.csv")
        df = df.rename(columns={"Index": "species_id", "Species Name": "species_name"})

    # copy shared columns
    if not force:
        df, copied = _copy_shared_columns_from_siblings(df, csv_path)

    all_kwargs = dict(
        data_dir=data_dir,
        suffix=suffix,
        aggregated=aggregated,
        subsample_indices=subsample_indices,
    )


    # gather values
    for label, func, needed_keys, output_cols in COLUMN_BLOCKS:
        missing_cols = [col for col in output_cols if col not in df.columns]
        if not force and not missing_cols:
            print(f"Skipping '{label}': all columns already present: {output_cols}")
            continue

        print(f"\nComputing '{label}' — missing columns: {missing_cols}")
        block_kwargs = {k: all_kwargs[k] for k in needed_keys}
        new_cols = func(df, **block_kwargs)

        added, skipped = [], []
        for col, values in new_cols.items():
            if not force and col in df.columns:
                skipped.append(col)
            else:
                df[col] = values
                added.append(col)

        if added:
            print(f"  Added: {added}")
        if skipped:
            print(f"  Skipped (already present): {skipped}")

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f"Saved {csv_path}")
    return df



def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--add-only",
        action="store_true",
        help="Load existing CSV and only add missing columns.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force recompute all columns (ignore existing CSV and sibling CSVs).",
    )
    args = parser.parse_args()

    print(f"Config: aggregated={AGGREGATED}, subsample_file={SUBSAMPLE_FILE}")

    build_proxies(
        data_dir=DATA_DIR,
        aggregated=AGGREGATED,
        subsample_file=SUBSAMPLE_FILE,
        add_only=args.add_only,
        force=args.force,
    )


if __name__ == "__main__":
    main()
