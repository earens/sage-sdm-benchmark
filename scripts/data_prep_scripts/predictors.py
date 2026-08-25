import os
import argparse
import sys

import rasterio
import rootutils
import yaml
import h5py
import hdf5plugin  # noqa: F401  — registers the Blosc filters the data is written with
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from rasterio.warp import transform
from rasterio.windows import Window
from scipy.ndimage import distance_transform_edt
from tqdm import tqdm

import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

import pathlib
os.environ["PROJ_DATA"] = str(pathlib.Path(rasterio.__file__).parent / "proj_data")

# --- Setup project root and config ---
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
root = os.getenv("PROJECT_ROOT")
with open(f"{root}/configs/local_default.yaml") as stream:
    data_dir = yaml.safe_load(stream)["data_dir"]

tqdm.pandas()


# --- Define data paths ---
occurrence_data_dir = os.path.join(data_dir, "occurrence_data/")
raster_data_dir = os.path.join(data_dir, "environmental_rasters/")
predictor_data_dir = os.path.join(data_dir, "predictors/")

chelsa_data_dir = os.path.join(raster_data_dir, "chelsa/")
soil_data_dir = os.path.join(raster_data_dir, "soilgrids250/")
human_data_dir = os.path.join(raster_data_dir, "human_footprint/")
topo_data_dir = os.path.join(raster_data_dir, "topography/")

os.makedirs(predictor_data_dir, exist_ok=True)
os.makedirs(chelsa_data_dir, exist_ok=True)
os.makedirs(soil_data_dir, exist_ok=True)
os.makedirs(human_data_dir, exist_ok=True)
os.makedirs(topo_data_dir, exist_ok=True)



def _download_file(url, local_path, show_progress=False):
    """Download a single file with wget. Returns (filename, success)."""
    if show_progress:
        # Show wget's progress bar for this one file
        result = subprocess.run(
            ["wget", "--show-progress", "-q", "-O", local_path, url],
            capture_output=False
        )
    else:
        result = subprocess.run(
            ["wget", "-q", "-O", local_path, url],
            capture_output=True
        )
    if result.returncode != 0:
        if os.path.exists(local_path):
            os.remove(local_path)
        return (os.path.basename(local_path), False)
    return (os.path.basename(local_path), True)

def download_missing_files(missing, label, max_workers=8):
    """Download a list of (url, local_path) tuples in parallel.
    The first file shows wget progress, the rest download quietly."""
    if not missing:
        print(f"All {label} rasters already present.")
        return

    print(f"Downloading {len(missing)} missing {label} raster(s) with {max_workers} parallel workers...")

    # Download the first file with visible progress, rest quietly in background
    first_url, first_path = missing[0]
    remaining = missing[1:]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all background downloads first
        bg_futures = {executor.submit(_download_file, url, path, show_progress=False): url
                      for url, path in remaining}

        # Download the first file with progress in the foreground
        print(f"  [{1}/{len(missing)}] {os.path.basename(first_path)}")
        filename, success = _download_file(first_url, first_path, show_progress=True)
        if not success:
            print(f"  WARNING: Failed to download {first_url}")

        # Wait for remaining background downloads
        done = 1
        for future in as_completed(bg_futures):
            done += 1
            fname, ok = future.result()
            if not ok:
                print(f"  WARNING: Failed to download {bg_futures[future]}")
            else:
                print(f"  [{done}/{len(missing)}] {fname} done")
    print(f"All {label} downloads finished.")


# Auto-download missing CHELSA bioclim rasters
CHELSA_BASE_URL = "https://os.unil.cloud.switch.ch/chelsa02/chelsa/global/bioclim"
CHELSA_BIOCLIM_VARS = [f"bio{str(i).zfill(2)}" for i in range(1, 20)]

missing_chelsa = []
for var in CHELSA_BIOCLIM_VARS:
    filename = f"CHELSA_{var}_1981-2010_V.2.1.tif"
    local_path = os.path.join(chelsa_data_dir, filename)
    if not os.path.exists(local_path):
        url = f"{CHELSA_BASE_URL}/{var}/1981-2010/{filename}"
        missing_chelsa.append((url, local_path))

download_missing_files(missing_chelsa, "CHELSA")


# Auto-download missing SoilGrids250 rasters
SOILGRIDS_BASE_URL = "https://files.isric.org/soilgrids/former/2017-03-10/data"
SOILGRIDS_FILENAMES = [
    "BDTICM_M_250m_ll.tif", "BLDFIE_M_sl3_250m_ll.tif", "CECSOL_M_sl3_250m_ll.tif",
    "CLYPPT_M_sl3_250m_ll.tif", "ORCDRC_M_sl3_250m_ll.tif", "PHIHOX_M_sl3_250m_ll.tif",
    "SLTPPT_M_sl3_250m_ll.tif", "SNDPPT_M_sl3_250m_ll.tif"
]

missing_soilgrids = []
for filename in SOILGRIDS_FILENAMES:
    local_path = os.path.join(soil_data_dir, filename)
    if not os.path.exists(local_path):
        url = f"{SOILGRIDS_BASE_URL}/{filename}"
        missing_soilgrids.append((url, local_path))

download_missing_files(missing_soilgrids, "SoilGrids250")



# Auto-download missing EarthEnv topography rasters
TOPO_BASE_URL = "https://data.earthenv.org/topography"
TOPO_FILENAMES = [
    "elevation_1KMmd_GMTEDmd.tif", "roughness_1KMmd_GMTEDmd.tif", "tri_1KMmd_GMTEDmd.tif",
    "tpi_1KMmd_GMTEDmd.tif", "vrm_1KMmd_GMTEDmd.tif", "aspectcosine_1KMmd_GMTEDmd.tif",
    "aspectsine_1KMmd_GMTEDmd.tif", "slope_1KMmd_GMTEDmd.tif", "eastness_1KMmd_GMTEDmd.tif",
    "northness_1KMmd_GMTEDmd.tif", "pcurv_1KMmd_GMTEDmd.tif", "tcurv_1KMmd_GMTEDmd.tif",
    "dx_1KMmd_GMTEDmd.tif", "dy_1KMmd_GMTEDmd.tif", "dxx_1KMmd_GMTEDmd.tif",
    "dyy_1KMmd_GMTEDmd.tif"
]

missing_topo = []
for filename in TOPO_FILENAMES:
    local_path = os.path.join(topo_data_dir, filename)
    if not os.path.exists(local_path):
        url = f"{TOPO_BASE_URL}/{filename}"
        missing_topo.append((url, local_path))

download_missing_files(missing_topo, "EarthEnv Topography")

# Human Footprint rasters are NOT auto-downloaded (no stable direct-download URL).
# Manually download the 9 layers of the Global Human Footprint (Venter et al.
# 2016, Dryad https://doi.org/10.5061/dryad.052q5) into
# <data_dir>/environmental_rasters/human_footprint/ with these filenames:
#   HFP2009.tif, Built2009.tif, croplands2005.tif, Lights2009.tif,
#   NavWater2009.tif, Pasture2009.tif, Popdensity2010.tif, Railways.tif, Roads.tif
# (the exact list is in get_human_footprint_paths() below).


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets", nargs="+",
        default=["eval", "gbif_iucn", "background", "train"],
        help="Species datasets to process"
    )
    parser.add_argument(
        "--predictors", nargs="+",
        default=["location", "chelsa", "soilgrids", "human_footprint", "topography"],
        help="Predictor datasets to process"
    )
    return parser.parse_args()


# --- Datasets loading functions ---
def get_splotopen():
    df = pd.read_csv(os.path.join(occurrence_data_dir, "splotopen.csv"), low_memory=False)
    print(f"Loaded {len(df):,} sPlotOpen records")
    # Sort to match targets.py groupby order (groupby sorts by default)
    locations = df[["Longitude", "Latitude"]].drop_duplicates().sort_values(
        ["Longitude", "Latitude"]
    ).values
    print(f"Found {len(locations):,} unique locations")
    return locations


def get_splotopen_sampled_area():
    """Per-cell sampled plot area (m^2) for the sPlotOpen eval cells.

    Returns an (n_cells, 1) array aligned 1:1 with ``get_splotopen()`` locations
    (and the target rows, which share the same sorted unique (Longitude, Latitude)
    order). Each cell's value is the total sampled effort: the sum of
    ``Releve_area`` over the *distinct* plots (``PlotObservationID``) that fall in
    that 1 km cell.

    Area is computed from the cleaned *plot-level* sPlotOpen data
    (``data_raw/splotopen/splotopen.csv``) — all surveyed plots, before
    species-survival filtering and 1 km aggregation. This recovers effort for the
    all-negative / empty cells (whose plot identity was dropped downstream) and
    gives the true multi-plot sum for non-empty cells. The join is on integer 1 km
    grid keys (EPSG:6933, the same projection/resolution as the pipeline) to avoid
    float-coordinate mismatch. Cells whose plots all lack a recorded Releve_area
    stay NaN (genuine missingness); downstream ``NaN < threshold`` is False, so
    those absences are never masked.
    """
    from geo_utils import latlon_to_grid_cell

    def grid_key(lon, lat):
        return latlon_to_grid_cell(lon, lat)

    # Canonical eval rows in get_splotopen() / targets.py order, tagged with grid key.
    eval_coords = (
        pd.read_csv(os.path.join(occurrence_data_dir, "splotopen.csv"),
                    usecols=["Longitude", "Latitude"], low_memory=False)
        .drop_duplicates().sort_values(["Longitude", "Latitude"]).reset_index(drop=True)
    )
    eval_coords["gx"], eval_coords["gy"] = grid_key(
        eval_coords["Longitude"].values, eval_coords["Latitude"].values)

    raw_path = os.path.join(os.path.dirname(data_dir.rstrip("/")),
                            "data_raw", "splotopen", "splotopen.csv")
    if os.path.exists(raw_path):
        # One row per distinct plot (its coords + area), gridded and summed per cell.
        plots = pd.read_csv(raw_path, sep=";",
                            usecols=["PlotObservationID", "Longitude", "Latitude", "Releve_area"],
                            low_memory=False)
        plots = plots.dropna(subset=["PlotObservationID"]).drop_duplicates("PlotObservationID")
        plots["gx"], plots["gy"] = grid_key(plots["Longitude"].values, plots["Latitude"].values)
        per_cell = (plots.groupby(["gx", "gy"])["Releve_area"]
                    .sum(min_count=1).rename("sampled_area_m2").reset_index())
        area = eval_coords.merge(per_cell, on=["gx", "gy"], how="left")["sampled_area_m2"].values
    else:
        print(f"WARNING: {raw_path} not found; sampled_area will be all-NaN "
              f"(no absences will be masked). Run splotopen.py to regenerate it.")
        area = np.full(len(eval_coords), np.nan)

    area = area.reshape(-1, 1).astype(np.float32)
    n_known = int(np.isfinite(area).sum())
    print(f"Computed sampled_area for {len(area):,} cells from cleaned plot-level data "
          f"({n_known:,} with known area, {len(area) - n_known:,} NaN)")
    return area

def get_gbif_iucn():
    df = pd.read_csv(os.path.join(occurrence_data_dir, "gbif_iucn.csv"))
    print(f"Loaded {len(df):,} GBIF-IUCN records")
    locations = df[["decimalLongitude", "decimalLatitude"]].drop_duplicates().sort_values(
        ["decimalLongitude", "decimalLatitude"]
    ).values
    print(f"Found {len(locations):,} unique locations")
    return locations

def get_gbif_powo():
    df = pd.read_csv(os.path.join(occurrence_data_dir, "gbif.csv"))
    print(f"Loaded {len(df):,} GBIF-POWO records")
    locations = df[["decimalLongitude", "decimalLatitude"]].drop_duplicates().sort_values(
        ["decimalLongitude", "decimalLatitude"]
    ).values
    print(f"Found {len(locations):,} unique locations")
    return locations

def get_background():
    locations = generate_background_data()
    return locations


# --- Raster extraction ---
STRIP_HEIGHT = 256          # rows per strip for windowed reads
NN_MAX_RADIUS = 50


def _sample_values_striped(src, x_coords, y_coords, strip_height=256, desc=None):
    """Memory-efficient raster sampling by reading horizontal strips.

    Instead of per-pixel src.sample(), converts coordinates to pixel indices,
    groups by row-strip, and reads only needed strips. Each strip is at most
    (strip_height × raster_width) pixels — much less than the full raster.
    """
    total = len(x_coords)
    values = np.full(total, np.nan, dtype=np.float64)

    # Convert geo coords to pixel coords
    inv_t = ~src.transform
    col_px = np.floor(inv_t[0] * x_coords + inv_t[1] * y_coords + inv_t[2]).astype(np.intp)
    row_px = np.floor(inv_t[3] * x_coords + inv_t[4] * y_coords + inv_t[5]).astype(np.intp)

    h, w = src.height, src.width
    row_px = np.clip(row_px, 0, h - 1)
    col_px = np.clip(col_px, 0, w - 1)

    # Group points into horizontal strips
    strip_ids = row_px // strip_height
    unique_strips = np.unique(strip_ids)

    strip_iter = unique_strips
    if desc is not None:
        strip_iter = tqdm(strip_iter, desc=desc, leave=False, file=sys.stdout)

    for sid in strip_iter:
        mask = strip_ids == sid
        r_start = int(sid * strip_height)
        r_end = min(r_start + strip_height, h)

        window = Window(col_off=0, row_off=r_start, width=w, height=r_end - r_start)
        strip_data = src.read(1, window=window)

        local_rows = row_px[mask] - r_start
        local_cols = col_px[mask]
        values[mask] = strip_data[local_rows, local_cols].astype(np.float64)

    return values


NN_TILE_SIZE = 512  # tile size for batch NN fill (rows and cols)


def _batch_nn_fill_tiled(src, values, nodata_mask, row_px, col_px,
                         tile_size=NN_TILE_SIZE, pad=NN_MAX_RADIUS,
                         desc=None):
    """Batch nearest-neighbor fill using small tile reads + scipy EDT.

    Groups nodata query points into (tile_size × tile_size) spatial tiles,
    reads each tile with `pad`-pixel padding, runs distance_transform_edt
    to find the nearest valid pixel for every cell, then looks up the
    replacement values.  Memory per tile ≈ 4 × (tile+2*pad)² × 8 bytes
    ≈ 15 MB for tile_size=512, pad=50.  Fully vectorised per tile.

    Parameters
    ----------
    src : rasterio DatasetReader (open)
    values : 1-D float64 array (modified in-place)
    nodata_mask : bool array — True where values == nodata
    row_px, col_px : int arrays — pixel coordinates for *all* points
    tile_size, pad : int
    desc : str or None — tqdm description

    Returns
    -------
    fill_attempted, fill_success : int
    """
    h, w = src.height, src.width
    nodata_value = src.nodata
    nodata_indices = np.flatnonzero(nodata_mask)
    if len(nodata_indices) == 0:
        return 0, 0

    nd_rows = row_px[nodata_indices]
    nd_cols = col_px[nodata_indices]

    # Assign each nodata point to a 2-D tile
    tile_r = nd_rows // tile_size
    tile_c = nd_cols // tile_size
    # Encode tile (r, c) as a single int for grouping
    tile_id = tile_r.astype(np.int64) * 1_000_000 + tile_c.astype(np.int64)
    unique_tiles = np.unique(tile_id)

    fill_attempted = 0
    fill_success = 0

    tile_iter = unique_tiles
    if desc is not None:
        tile_iter = tqdm(tile_iter, desc=desc, leave=False, file=sys.stdout)

    for tid in tile_iter:
        tmask = tile_id == tid
        pts_rows = nd_rows[tmask]
        pts_cols = nd_cols[tmask]
        pts_global_idx = nodata_indices[tmask]

        tr = int(tid // 1_000_000)
        tc = int(tid % 1_000_000)

        # Padded read window (clipped to raster bounds)
        r_start = max(tr * tile_size - pad, 0)
        r_end = min((tr + 1) * tile_size + pad, h)
        c_start = max(tc * tile_size - pad, 0)
        c_end = min((tc + 1) * tile_size + pad, w)

        window = Window(col_off=c_start, row_off=r_start,
                        width=c_end - c_start, height=r_end - r_start)
        tile_data = src.read(1, window=window).astype(np.float64)

        valid = ~np.isclose(tile_data, nodata_value, atol=0)
        if not valid.any():
            fill_attempted += len(pts_rows)
            continue

        # EDT on the tile — returns distance and index of nearest valid pixel
        dist, indices = distance_transform_edt(
            ~valid, return_distances=True, return_indices=True
        )

        local_rows = pts_rows - r_start
        local_cols = pts_cols - c_start

        # Safety clamp
        local_rows = np.clip(local_rows, 0, tile_data.shape[0] - 1)
        local_cols = np.clip(local_cols, 0, tile_data.shape[1] - 1)

        nn_rows = indices[0][local_rows, local_cols]
        nn_cols = indices[1][local_rows, local_cols]
        nn_dist = dist[local_rows, local_cols]

        filled_vals = tile_data[nn_rows, nn_cols]

        # Only accept fills within max radius
        within_radius = nn_dist <= pad
        nn_valid = (within_radius
                    & np.isfinite(filled_vals)
                    & ~np.isclose(filled_vals, nodata_value, atol=0))

        fill_attempted += len(pts_rows)
        fill_success += int(nn_valid.sum())
        values[pts_global_idx[nn_valid]] = filled_vals[nn_valid]

    return fill_attempted, fill_success


def extract_values_from_raster(file_paths, locations_dict):
    """Extract raster values for multiple location arrays in a memory-safe pass.

    Uses chunked `src.sample` to avoid loading full rasters in RAM.
    For nodata values, attempts local nearest-neighbor fill for a capped number
    of points and leaves the rest as NaN.
    """
    results = {name: np.zeros((len(locs), len(file_paths)), dtype=np.float64) for name, locs in locations_dict.items()}

    for j, path in enumerate(file_paths):
        print(f"\n[{j + 1}/{len(file_paths)}] Processing: {os.path.basename(path)}")
        with rasterio.open(path) as src:
            nodata_value = src.nodata
            print(f"  Strip reads (strip_h={STRIP_HEIGHT}, nodata={nodata_value})")

            for name, locations in locations_dict.items():
                print(f"  Extracting for {name} ({len(locations):,} locations)")

                x_coords, y_coords = transform('EPSG:4326', src.crs, locations[:, 0], locations[:, 1])
                x_coords = np.asarray(x_coords)
                y_coords = np.asarray(y_coords)

                values = _sample_values_striped(
                    src,
                    x_coords,
                    y_coords,
                    desc=f"    {name} sample",
                )

                if nodata_value is None:
                    nodata_mask = np.zeros_like(values, dtype=bool)
                else:
                    nodata_mask = np.isclose(values, nodata_value, atol=0)

                fill_attempted = 0
                fill_success = 0
                if nodata_mask.any() and nodata_value is not None:
                    # Compute pixel coords for all points (reuse inverse transform)
                    inv_t = ~src.transform
                    all_col_px = np.floor(inv_t[0] * x_coords + inv_t[1] * y_coords + inv_t[2]).astype(np.intp)
                    all_row_px = np.floor(inv_t[3] * x_coords + inv_t[4] * y_coords + inv_t[5]).astype(np.intp)
                    all_row_px = np.clip(all_row_px, 0, src.height - 1)
                    all_col_px = np.clip(all_col_px, 0, src.width - 1)

                    fill_attempted, fill_success = _batch_nn_fill_tiled(
                        src, values, nodata_mask, all_row_px, all_col_px,
                        desc=f"    {name} NN fill",
                    )

                if nodata_value is not None:
                    values[np.isclose(values, nodata_value, atol=0)] = np.nan

                nan_count = int(np.isnan(values).sum())
                nodata_count = int(nodata_mask.sum())
                print(
                    f"  {name}: {nodata_count} nodata, {fill_success}/{fill_attempted} locally filled, {nan_count} NaN"
                )

                results[name][:, j] = values

    return results


# --- Predictor retrieval ---
def get_chelsa_paths():
    chelsa_variables = ['bio' + str(i+1).zfill(2) for i in range(19)]
    return [os.path.join(chelsa_data_dir, f"CHELSA_{var}_1981-2010_V.2.1.tif") for var in chelsa_variables]

def get_soilgrids250_paths():
    filenames = ["BDTICM_M_250m_ll.tif", "BLDFIE_M_sl3_250m_ll.tif", "CECSOL_M_sl3_250m_ll.tif",
                 "CLYPPT_M_sl3_250m_ll.tif", "ORCDRC_M_sl3_250m_ll.tif", "PHIHOX_M_sl3_250m_ll.tif",
                 "SLTPPT_M_sl3_250m_ll.tif", "SNDPPT_M_sl3_250m_ll.tif"]
    return [os.path.join(soil_data_dir, f) for f in filenames]

def get_human_footprint_paths():
    filenames = ["HFP2009.tif", "Built2009.tif", "croplands2005.tif", "Lights2009.tif",
                 "NavWater2009.tif", "Pasture2009.tif", "Popdensity2010.tif", "Railways.tif", "Roads.tif"]
    return [os.path.join(human_data_dir, f) for f in filenames]

def get_topography_paths():
    filenames = [
        "elevation_1KMmd_GMTEDmd.tif", "roughness_1KMmd_GMTEDmd.tif", "tri_1KMmd_GMTEDmd.tif",
        "tpi_1KMmd_GMTEDmd.tif", "vrm_1KMmd_GMTEDmd.tif", "aspectcosine_1KMmd_GMTEDmd.tif",
        "aspectsine_1KMmd_GMTEDmd.tif", "slope_1KMmd_GMTEDmd.tif", "eastness_1KMmd_GMTEDmd.tif",
        "northness_1KMmd_GMTEDmd.tif", "pcurv_1KMmd_GMTEDmd.tif", "tcurv_1KMmd_GMTEDmd.tif",
        "dx_1KMmd_GMTEDmd.tif", "dy_1KMmd_GMTEDmd.tif", "dxx_1KMmd_GMTEDmd.tif",
        "dyy_1KMmd_GMTEDmd.tif"
    ]
    return [os.path.join(topo_data_dir, f) for f in filenames]

def save_predictor_data(predictor_data, predictor_name, dataset):
    output_file = os.path.join(predictor_data_dir, f"{dataset}_{predictor_name}.h5")
    data = np.asarray(predictor_data, dtype=np.float32)
    nrows = data.shape[0]
    ncols = data.shape[1] if data.ndim > 1 else 1
    chunk_rows = min(1024, nrows)
    chunk_shape = (chunk_rows, ncols) if data.ndim > 1 else (chunk_rows,)
    with h5py.File(output_file, 'w') as f:
        f.create_dataset('predictor_data', data=data, chunks=chunk_shape,
                         compression='gzip', compression_opts=1)


# --- Background point generation ---
def generate_background_data():
    """Generate uniform background points on land using a Fibonacci sphere."""

    EARTH_RADIUS = 6371.0  # globally-average value, in km
    SPACING_KM = 5

    def fibonacci_sphere(n_points):
        """Generate Fibonacci lattice points on unit sphere."""
        indices = np.arange(0, n_points)
        phi = np.pi * (3. - np.sqrt(5)) # golden angle ~2.399963

        y = 1 - (indices / float(n_points - 1)) * 2
        radius = np.sqrt(1 - y * y)

        theta = phi * indices

        x = np.cos(theta) * radius
        z = np.sin(theta) * radius

        return x, y, z

    def spherical_to_geographic(x, y, z):
        """Convert 3D Cartesian coords to geographic lat/lon."""
        lon = np.arctan2(z, x)
        lat = np.arcsin(y)
        lon_deg = np.degrees(lon)
        lat_deg = np.degrees(lat)
        return lon_deg, lat_deg

    # Estimate number of points
    surface_area = 4 * np.pi * EARTH_RADIUS**2
    area_per_point = SPACING_KM**2
    n_points = int(surface_area / area_per_point)

    print(f"\nGenerating {n_points:,} background points for ~{SPACING_KM} km spacing")
    x, y, z = fibonacci_sphere(n_points)
    lon, lat = spherical_to_geographic(x, y, z)
    gdf = gpd.GeoDataFrame(geometry=[Point(lon[i], lat[i]) for i in range(n_points)], crs="EPSG:4326")

    print("Filtering points to land areas...")
    world = gpd.read_file(
        "https://d2ad6b4ur7yvpq.cloudfront.net/naturalearth-3.3.0/ne_50m_admin_0_countries.geojson"
    )
    land = world[world.geometry.type.isin(["Polygon", "MultiPolygon"])]
    on_land = gpd.sjoin(gdf, land, how="inner", predicate="within")
    print(f"Filtered {len(on_land):,} land points out of {n_points:,}")

    return np.column_stack((on_land.geometry.x.values, on_land.geometry.y.values))


def main():

    args = parse_args()
    print(f"Processing datasets: {args.datasets}")
    print(f"Processing predictors: {args.predictors}")

    dataset_funcs = {
        "eval": get_splotopen,
        "gbif_iucn": get_gbif_iucn,
        "background": get_background,
        "train": get_gbif_powo
    }

    predictor_path_funcs = {
        "chelsa": get_chelsa_paths,
        "soilgrids": get_soilgrids250_paths,
        "human_footprint": get_human_footprint_paths,
        "topography": get_topography_paths
    }

    # Load all requested datasets once
    all_locations = {}
    for dataset in args.datasets:
        if dataset not in dataset_funcs:
            raise ValueError(f"Unknown dataset: {dataset}")
        print(f"\nLoading dataset: {dataset}")
        all_locations[dataset] = dataset_funcs[dataset]()

    # Process predictors — each raster opened once for all datasets
    for predictor in args.predictors:
        if predictor == "location":
            # Location is just the coordinates themselves, no raster to open
            for dataset in args.datasets:
                print(f"\nSaving location for {dataset}")
                save_predictor_data(all_locations[dataset], "location", dataset)
            continue

        if predictor == "sampled_area":
            # sPlotOpen-only: per-cell sampled plot area for absence-reliability
            # weighting during eval. Aligned 1:1 with splotopen location/target rows.
            print("\nSaving sampled_area for splotopen")
            save_predictor_data(get_splotopen_sampled_area(), "sampled_area", "eval")
            continue

        if predictor not in predictor_path_funcs:
            raise ValueError(f"Unknown predictor: {predictor}")

        print(f"\n{'='*60}")
        print(f"Processing {predictor} for all datasets")
        print(f"{'='*60}")
        file_paths = predictor_path_funcs[predictor]()
        results = extract_values_from_raster(file_paths, all_locations)

        for dataset in args.datasets:
            save_predictor_data(results[dataset], predictor, dataset)


if __name__ == "__main__":
    main()
