"""Compute layer for prediction maps.

Covers the data/model side of building prediction maps:
  - Species data loading (range, presences, in-range cells)
  - Building a regular prediction grid with bg-point NN lookup
  - Model loading + inference for sklearn (.pkl), torch (.ckpt), maxent (.rds)
  - Reconstructing flat predictions into an (H, W) grid
  - Rescaling / range-masking predictions for display

The plotting (render) layer lives in src.visualizations.maps.render.
"""

import importlib
import os
import subprocess
import tempfile
from pathlib import Path

import geopandas as gpd
import h5py
import hdf5plugin  # noqa: F401 — registers compressed HDF5 filters
import numpy as np
import pandas as pd
import torch
import yaml
from scipy.spatial import cKDTree

from src.data_modules.datamodule import predictors_means, predictors_stds
from src.models.resnet import encode_locations

# ---------------------------------------------------------------------------
# Project root + data dir resolution
# ---------------------------------------------------------------------------
# From src/visualizations/maps/grid.py, parents[3] is the sdm_benchmark repo root.
_PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _resolve_data_dir():
    try:
        cfg_path = Path(__file__).resolve().parents[3] / "configs" / "local_default.yaml"
        with open(cfg_path) as f:
            return yaml.safe_load(f).get("data_dir")
    except Exception:
        return None


DATA_DIR = _resolve_data_dir()

PREDICTORS_SUBSET = {
    "chelsa":          list(range(19)),
    "soilgrids":       list(range(8)),
    "topography":      list(range(16)),
    "human_footprint": list(range(9)),
}


# ---------------------------------------------------------------------------
# Species / data loading helpers
# ---------------------------------------------------------------------------


def find_properties_csv(data_dir: str = DATA_DIR) -> str:
    """Path to the species-properties table (sampling effort x relative prevalence)."""
    path = os.path.join(data_dir, "targets", "species_properties_1km.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No species properties CSV at {path}")
    return path


def find_native_range_shp(species_name: str, data_dir: str = DATA_DIR) -> str:
    """Path to a species' POWO native-range shapefile (from native_ranges.tar.gz)."""
    path = os.path.join(data_dir, "native_ranges", f"{species_name}.shp")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No native-range shapefile for '{species_name}' at {path}")
    return path


def get_species_id(species_name: str, data_dir: str = DATA_DIR) -> int:
    """Return integer species_id for *species_name* from the properties CSV."""
    path = find_properties_csv(data_dir)
    proxies = pd.read_csv(path)
    row = proxies[proxies["species_name"] == species_name]
    if len(row) == 0:
        raise ValueError(
            f"Species '{species_name}' not found in {os.path.basename(path)}. "
            "Check spelling or list available names from that CSV."
        )
    return int(row["species_id"].iloc[0])


def load_world_map(data_dir: str = DATA_DIR) -> gpd.GeoDataFrame:
    """Load Natural Earth country polygons."""
    path = os.path.join(data_dir, "maps", "ne_110m_admin_0_countries.shp")
    return gpd.read_file(path).to_crs("EPSG:4326")


def load_species_data(species_name: str, data_dir: str = DATA_DIR):
    """
    Load species data for visualisation.

    Returns
    -------
    in_range_coords : np.ndarray, shape (N, 2)
        [lon, lat] of all 1 km cells visited within the species range.
    presence_coords : np.ndarray, shape (M, 2)
        [lon, lat] of cells where the species was actually observed.
    range_gdf : GeoDataFrame
        Native-range polygon shapefile.

    Everything here is on the 1 km grid: the mask and target index arrays both
    address ``train_location_1km.h5`` rows, so the location array must be the 1 km
    one for the two cell sets to land in the same coordinate space. (Reading the
    non-aggregated ``train_location.h5`` instead and indexing it with 1 km indices
    silently returned wrong ``in_range_coords``.) Coordinates are therefore 1 km
    cell centroids, which is the resolution the benchmark trains on.
    """
    species_id = get_species_id(species_name, data_dir)

    # All 1 km observation locations (global)
    with h5py.File(os.path.join(data_dir, "predictors", "train_location_1km.h5"), "r") as f:
        all_locations = f["predictor_data"][:]   # (N_1km, 2)

    def _load_sparse_indices(h5_path, species_id):
        with h5py.File(h5_path, "r") as f:
            lookup = f["target_location_indices_lookup"]
            tli    = f["target_location_indices"]
            s = int(lookup[species_id])
            e = int(lookup[species_id + 1])
            return tli[s:e]

    # In-range (visited) cells
    in_range_idx = _load_sparse_indices(
        os.path.join(data_dir, "range_masks_1km", "train_species_index.h5"),
        species_id,
    )
    in_range_coords = all_locations[in_range_idx]

    # Presence observations
    presence_idx = _load_sparse_indices(
        os.path.join(data_dir, "targets", "train_targets_1km.h5"),
        species_id,
    )
    presence_coords = all_locations[presence_idx]

    # Native-range shapefile
    shp_path = find_native_range_shp(species_name, data_dir)
    range_gdf = gpd.read_file(shp_path).to_crs("EPSG:4326")

    return in_range_coords, presence_coords, range_gdf


def load_splotopen_species_data(species_name: str, data_dir: str = DATA_DIR):
    """
    Load splotopen presence-absence data for a species.

    Returns
    -------
    in_range_coords : np.ndarray, shape (N, 2)
        [lon, lat] of all splotopen plots within the species range.
    presence_coords : np.ndarray, shape (M, 2)
        [lon, lat] of plots where the species was recorded as present.
    """
    species_id = get_species_id(species_name, data_dir)

    with h5py.File(os.path.join(data_dir, "predictors", "eval_location.h5"), "r") as f:
        all_locations = f["predictor_data"][:]   # (53336, 2)

    def _load_sparse_indices(h5_path, species_id):
        with h5py.File(h5_path, "r") as f:
            lookup = f["target_location_indices_lookup"]
            tli    = f["target_location_indices"]
            s = int(lookup[species_id])
            e = int(lookup[species_id + 1])
            return tli[s:e]

    in_range_idx  = _load_sparse_indices(
        os.path.join(data_dir, "range_masks_1km", "eval_species_index.h5"),
        species_id,
    )
    presence_idx  = _load_sparse_indices(
        os.path.join(data_dir, "targets", "eval_targets.h5"),
        species_id,
    )

    return all_locations[in_range_idx], all_locations[presence_idx]


def compute_crop_bounds(range_gdf: gpd.GeoDataFrame, padding_frac: float = 0.05):
    """
    Return (minx, miny, maxx, maxy) bounding box around the range polygon,
    grown by ``padding_frac`` of the range extent on each side (default 5 %).
    """
    b = range_gdf.total_bounds  # (minx, miny, maxx, maxy)
    pad_lon = (b[2] - b[0]) * padding_frac
    pad_lat = (b[3] - b[1]) * padding_frac
    return (
        max(b[0] - pad_lon, -180.0),
        max(b[1] - pad_lat,  -90.0),
        min(b[2] + pad_lon,  180.0),
        min(b[3] + pad_lat,   90.0),
    )


# ---------------------------------------------------------------------------
# Prediction grid
# ---------------------------------------------------------------------------

def load_bg_coords(data_dir: str = DATA_DIR) -> np.ndarray:
    """Load all background point coordinates. Shape (N_bg, 2) [lon, lat]."""
    with h5py.File(os.path.join(data_dir, "predictors", "background_location.h5"), "r") as f:
        return f["predictor_data"][:]


def build_prediction_grid(
    crop_bounds,
    world_gdf: gpd.GeoDataFrame,
    bg_coords: np.ndarray,
    data_dir: str = DATA_DIR,
    predictors_subset: dict = None,
    res: float = 0.1,
    normalize: bool = True,
):
    """
    Build a regular lat/lon grid over *crop_bounds*, find the nearest bg point
    for each land cell, load + normalise its environmental predictors.

    Parameters
    ----------
    crop_bounds : (minx, miny, maxx, maxy)
    world_gdf   : world country GeoDataFrame for land masking
    bg_coords   : (N_bg, 2) background coordinates [lon, lat]
    data_dir    : root data directory
    predictors_subset : dict mapping predictor-type → column indices
    res         : grid resolution in degrees (default 0.1°)

    Returns
    -------
    lons              : 1-D lon array (W,)
    lats              : 1-D lat array (H,)  — top-to-bottom
    grid_coords       : (M, 2) [lon, lat] of land grid cells
    grid_predictors   : (M, D) normalised predictor matrix
    land_flat_indices : (M,) flat indices into the H×W grid (for map reconstruction)
    """
    if predictors_subset is None:
        predictors_subset = PREDICTORS_SUBSET

    minx, miny, maxx, maxy = crop_bounds

    lons = np.arange(minx, maxx + res * 0.5, res)
    lats = np.arange(maxy, miny - res * 0.5, -res)   # top → bottom
    W, H = len(lons), len(lats)

    lon_grid, lat_grid = np.meshgrid(lons, lats)
    flat_coords = np.column_stack([lon_grid.ravel(), lat_grid.ravel()])  # (H*W, 2)

    # --- Pre-filter bg coords to crop bounds (fast numpy, avoids global KD-tree) ---
    bg_mask = (
        (bg_coords[:, 0] >= minx) & (bg_coords[:, 0] <= maxx) &
        (bg_coords[:, 1] >= miny) & (bg_coords[:, 1] <= maxy)
    )
    bg_crop_global_idx = np.where(bg_mask)[0]   # indices into the original bg_coords
    bg_crop = bg_coords[bg_crop_global_idx]     # (K, 2)

    print(f"  BG points in crop: {len(bg_crop):,}")

    # --- KD-tree on the crop subset only ---
    print("  Building KD-tree on crop bg coords …")
    tree = cKDTree(bg_crop.astype(np.float64))
    # workers=-1 uses all CPU cores for the query
    dists, nn_local = tree.query(flat_coords.astype(np.float64), k=1, workers=-1)
    print("  KD-tree done.")

    # --- Land mask via distance threshold ----------------------------------
    # Grid points with no bg point within 1.5 × res are ocean.
    max_dist = res * 1.5
    on_land = dists <= max_dist
    land_flat_indices = np.where(on_land)[0]
    grid_coords = flat_coords[land_flat_indices]          # (M, 2)

    # bg indices into the *original* bg_coords array, for HDF5 loading
    bg_nn_idx = bg_crop_global_idx[nn_local[land_flat_indices]]

    print(f"  Prediction grid: {H}×{W}, land cells: {len(grid_coords):,}")

    # --- Assemble raw predictor matrix -------------------------------------
    # Fancy h5py indexing on large files is slow (random seeks).
    # Instead: load the entire predictor array into numpy (one fast sequential
    # read), then use numpy indexing which is much faster.
    unique_bg_idx, inverse = np.unique(bg_nn_idx, return_inverse=True)

    parts = []
    raw_means, raw_stds = [], []

    for name, col_indices in predictors_subset.items():
        if name == "location":
            continue
        h5_path = os.path.join(data_dir, "predictors", f"background_{name}.h5")
        print(f"  Loading {os.path.basename(h5_path)} …")
        with h5py.File(h5_path, "r") as f:
            all_rows = f["predictor_data"][:]          # full sequential read
        data_unique = all_rows[unique_bg_idx][:, col_indices]
        del all_rows                                   # free before next file
        parts.append(data_unique[inverse].astype(np.float32))

        full_means = predictors_means[name]
        full_stds  = predictors_stds[name]
        for idx in col_indices:
            raw_means.append(full_means[idx])
            raw_stds.append(full_stds[idx])

    grid_raw = np.concatenate(parts, axis=1)              # (M, D)

    if not normalize:
        # Return raw predictors (NaNs intact) so the caller applies its own
        # normalization (e.g. a model-specific transform for the traditional SDMs).
        return lons, lats, grid_coords, grid_raw, land_flat_indices

    # --- Normalise (same as NormalizeTabular in datamodule.py) -------------
    means = np.array(raw_means, dtype=np.float32)
    stds  = np.array(raw_stds,  dtype=np.float32)
    stds  = np.where(stds == 0, 1.0, stds)

    # Impute NaN → mean, then z-score
    nan_mask = np.isnan(grid_raw)
    grid_raw[nan_mask] = np.broadcast_to(means, grid_raw.shape)[nan_mask]
    grid_predictors = (grid_raw - means) / stds           # (M, D)

    return lons, lats, grid_coords, grid_predictors, land_flat_indices


# ---------------------------------------------------------------------------
# Model inference
# ---------------------------------------------------------------------------

def load_model_score(
    model_path: str,
    species_name: str,
    metric: str = "test_auroc",
    data_dir: str = DATA_DIR,
) -> float:
    """
    Load a per-species test metric from the run's metrics_by_species.h5.

    For torch .ckpt files the run directory is two levels up (run_dir/checkpoints/x.ckpt).
    For sklearn .pkl and maxent .rds files the metrics file is expected in the
    same directory as the model file.

    Parameters
    ----------
    model_path   : path to .ckpt / .pkl / .rds
    species_name : species name (used to look up species_id)
    metric       : dataset name in metrics_by_species.h5
                   ('test_auroc', 'test_auprc', 'test_auprg', 'test_f1')
    """
    mtype = detect_model_type(model_path)

    if mtype == "torch":
        candidate_dirs = [os.path.dirname(os.path.dirname(model_path))]
    else:
        # sklearn .pkl / maxent .rds: metrics_by_species.h5 sits either next to the
        # model (legacy flat layout) or one level up when the model lives in a
        # `models/` subdir (current traditional_sdms layout).
        model_dir = os.path.dirname(model_path)
        candidate_dirs = [model_dir, os.path.dirname(model_dir)]

    for run_dir in candidate_dirs:
        metrics_path = os.path.join(run_dir, "metrics_by_species.h5")
        if os.path.exists(metrics_path):
            break
    else:
        searched = ", ".join(candidate_dirs)
        raise FileNotFoundError(
            f"metrics_by_species.h5 not found in: {searched}. "
            "Run evaluation first, or set score= manually."
        )

    species_id = get_species_id(species_name, data_dir)
    with h5py.File(metrics_path, "r") as f:
        if metric not in f:
            available = list(f.keys())
            raise KeyError(f"Metric '{metric}' not in {metrics_path}. Available: {available}")
        value = float(f[metric][species_id])

    return value


def detect_model_type(model_path: str) -> str:
    ext = os.path.splitext(model_path)[1].lower()
    if ext == ".pkl":
        return "sklearn"
    if ext == ".ckpt":
        return "torch"
    if ext == ".rds":
        return "maxent"
    raise ValueError(
        f"Cannot infer model type from extension '{ext}'. "
        "Expected .pkl (sklearn), .ckpt (torch), or .rds (maxent)."
    )


def predict_with_model(
    model_path: str,
    grid_coords: np.ndarray,
    grid_predictors: np.ndarray,
    use_location: bool = True,
    species_col: int = None,
    batch_size: int = 8192,
) -> np.ndarray:
    """
    Auto-detect model type and return predicted probabilities (M,).

    Parameters
    ----------
    model_path       : path to .pkl / .ckpt / .rds
    grid_coords      : (M, 2) [lon, lat]
    grid_predictors  : (M, D) normalised predictors
    use_location     : prepend encoded location to features (sklearn / maxent only)
    species_col      : for multi-species torch models, which output column to select
    batch_size       : batch size for torch inference
    """
    mtype = detect_model_type(model_path)
    if mtype == "sklearn":
        return _predict_sklearn(model_path, grid_coords, grid_predictors, use_location)
    if mtype == "torch":
        return _predict_torch(
            model_path, grid_coords, grid_predictors, species_col, batch_size
        )
    return _predict_maxent(model_path, grid_coords, grid_predictors, use_location)


def _predict_sklearn(
    model_path: str,
    grid_coords: np.ndarray,
    grid_predictors: np.ndarray,
    use_location: bool,
) -> np.ndarray:

    import joblib

    model = joblib.load(model_path)

    if use_location:
        loc_enc = encode_locations(
            torch.tensor(grid_coords, dtype=torch.float32)
        ).numpy()
        X = np.concatenate([loc_enc, grid_predictors], axis=1)
    else:
        X = grid_predictors

    if hasattr(model, "predict_proba"):
        p = model.predict_proba(X)
        return (p[:, 1] if p.ndim == 2 and p.shape[1] == 2 else p.ravel()).astype(
            np.float32
        )
    if hasattr(model, "decision_function"):
        from scipy.special import expit
        return expit(model.decision_function(X)).ravel().astype(np.float32)
    return model.predict(X).ravel().astype(np.float32)


def _predict_torch(
    ckpt_path: str,
    grid_coords: np.ndarray,
    grid_predictors: np.ndarray,
    species_col: int,
    batch_size: int,
) -> np.ndarray:
    """
    Load a PyTorch Lightning checkpoint.

    The config.yaml in the run directory (ckpt_path/../../config.yaml) is used
    to reconstruct the network architecture.  Only the net weights are loaded.
    """
    run_dir     = os.path.dirname(os.path.dirname(ckpt_path))
    config_path = os.path.join(run_dir, "config.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"config.yaml not found at {config_path}. "
            "The config.yaml must be in the run directory (parent of 'checkpoints/')."
        )

    with open(config_path) as fh:
        cfg = yaml.safe_load(fh)

    net_cfg  = cfg["model"]["init_args"]["net"]
    cls_path = net_cfg["class_path"]
    init_kw  = net_cfg.get("init_args", {})

    # Resolve short class paths like "models.ResNet" → "src.models.ResNet"
    if cls_path.startswith("models.") and not cls_path.startswith("src."):
        cls_path = "src." + cls_path

    mod_path, cls_name = cls_path.rsplit(".", 1)
    net_cls = getattr(importlib.import_module(mod_path), cls_name)
    net     = net_cls(**init_kw)

    # Load weights (Lightning stores them under "net.*" keys)
    raw = torch.load(ckpt_path, map_location="cpu")
    net_state = {
        k[len("net."):]: v
        for k, v in raw["state_dict"].items()
        if k.startswith("net.")
    }
    net.load_state_dict(net_state, strict=True)
    net.eval()

    locs_t  = torch.tensor(grid_coords,     dtype=torch.float32)
    preds_t = torch.tensor(grid_predictors, dtype=torch.float32)

    all_probs = []
    with torch.no_grad():
        for i in range(0, len(locs_t), batch_size):
            bl = locs_t[i: i + batch_size]
            bp = preds_t[i: i + batch_size]
            logits = net(bl, bp)

            if logits.dim() > 1 and logits.shape[1] > 1:
                if species_col is None:
                    raise ValueError(
                        f"Model has {logits.shape[1]} output classes. "
                        "Pass species_col=<int> to select the target species column."
                    )
                logits = logits[:, species_col]
            else:
                logits = logits.squeeze(-1)

            all_probs.append(torch.sigmoid(logits).numpy())

    return np.concatenate(all_probs).astype(np.float32)


def _predict_maxent(
    rds_path: str,
    grid_coords: np.ndarray,
    grid_predictors: np.ndarray,
    use_location: bool,
) -> np.ndarray:
    """
    Run MaxEnt prediction via an R subprocess.
    Writes grid_predictors to a temp HDF5, calls maxent_predict.R, reads back.
    """
    if use_location:
        loc_enc = encode_locations(
            torch.tensor(grid_coords, dtype=torch.float32)
        ).numpy()
        X = np.concatenate([loc_enc, grid_predictors], axis=1).astype(np.float32)
    else:
        X = grid_predictors.astype(np.float32)

    r_script = os.path.join(
        _PROJECT_ROOT, "scripts", "traditional_sdms", "maxent_predict.R"
    )
    if not os.path.exists(r_script):
        raise FileNotFoundError(f"R prediction script not found: {r_script}")

    with tempfile.TemporaryDirectory() as tmpdir:
        in_h5  = os.path.join(tmpdir, "input.h5")
        out_h5 = os.path.join(tmpdir, "output.h5")

        with h5py.File(in_h5, "w") as f:
            f.create_dataset("X", data=X)

        subprocess.run(
            ["Rscript", r_script, in_h5, out_h5, rds_path],
            check=True,
        )

        with h5py.File(out_h5, "r") as f:
            return f["predictions"][:].astype(np.float32)


def predict_with_live_model(
    model,
    grid_coords: np.ndarray,
    grid_predictors: np.ndarray,
    species_col: int = None,
    batch_size: int = 8192,
    device=None,
) -> np.ndarray:
    """Run an already-loaded (in-memory) torch net over the grid -> probabilities (M,).

    Live-model counterpart of :func:`predict_with_model` (which loads a saved
    ``.ckpt``). Used for per-epoch wandb map logging, where the model is already
    in memory (and typically on GPU). The net is called as ``net(locations,
    predictors)`` exactly as in training.

    Parameters
    ----------
    model           : torch.nn.Module taking (locations, predictors) -> logits
    grid_coords     : (M, 2) [lon, lat]
    grid_predictors : (M, D) normalised predictors
    species_col     : for multi-species models, which output column to select
    batch_size      : inference batch size
    device          : torch device; defaults to the model's current device
    """
    import torch

    if device is None:
        device = next(model.parameters()).device

    locs = torch.as_tensor(grid_coords, dtype=torch.float32)
    preds = torch.as_tensor(grid_predictors, dtype=torch.float32)

    was_training = model.training
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(locs), batch_size):
            bl = locs[i: i + batch_size].to(device)
            bp = preds[i: i + batch_size].to(device)
            logits = model(bl, bp)
            if logits.dim() > 1 and logits.shape[1] > 1:
                if species_col is None:
                    raise ValueError(
                        f"Model has {logits.shape[1]} output classes. "
                        "Pass species_col=<int> to select the target species column."
                    )
                logits = logits[:, species_col]
            else:
                logits = logits.squeeze(-1)
            out.append(torch.sigmoid(logits).cpu().numpy())
    if was_training:
        model.train()

    return np.concatenate(out).astype(np.float32)


# ---------------------------------------------------------------------------
# Map reconstruction helper
# ---------------------------------------------------------------------------

def predictions_to_grid(
    lons: np.ndarray,
    lats: np.ndarray,
    land_flat_indices: np.ndarray,
    predictions: np.ndarray,
) -> np.ndarray:
    """
    Scatter flat *predictions* back into a (H, W) grid; ocean cells = NaN.

    Parameters
    ----------
    lons / lats         : 1-D coordinate arrays from build_prediction_grid
    land_flat_indices   : (M,) flat indices into H×W
    predictions         : (M,) probability values
    """
    H, W = len(lats), len(lons)
    grid = np.full((H, W), np.nan, dtype=np.float32)
    rows = land_flat_indices // W
    cols = land_flat_indices %  W
    grid[rows, cols] = predictions
    return grid


# ---------------------------------------------------------------------------
# Rescale / range-mask helpers
# ---------------------------------------------------------------------------

def _in_range_mask(land_flat_indices, lons, lats, range_gdf):
    """Boolean mask of land cells that fall inside the species range polygon."""
    W = len(lons)
    cell_lons = lons[land_flat_indices % W]
    cell_lats = lats[land_flat_indices // W]
    range_geom = range_gdf.unary_union
    try:
        from shapely import contains_xy
        return contains_xy(range_geom, cell_lons, cell_lats)
    except ImportError:
        from shapely.vectorized import contains
        return contains(range_geom, cell_lons, cell_lats)


# Smallest probability the log stretch resolves; below this everything is
# indistinguishable noise from a saturated sigmoid.
_LOG_FLOOR = 1e-12


def _make_rescale_ref(src: np.ndarray, low_q: float, high_q: float, mode: str):
    """Build a reusable rescale reference from a source distribution `src`.

    Returns a small tuple that fully defines the value→[0,1] mapping, so the
    *same* mapping can be applied to several maps (see compute_shared_rescale_ref).
    """
    if mode == "none":
        return ("none",)
    if mode == "rank":
        return ("rank", np.sort(src))
    if mode == "linear":
        lo, hi = np.percentile(src, [low_q, high_q])
        if hi <= lo:
            lo, hi = float(np.min(src)), float(np.max(src))
        return ("linear", lo, hi)
    if mode == "log":
        # Same percentile stretch, but in log10 space. Needed for models whose
        # in-range predictions span many decades (the tuned DeepSDM runs reach
        # 1e-15 to 1e-1 for a narrow-range species): a linear stretch collapses
        # everything but the top few cells to 0 and the map reads as binary,
        # while the structure actually lives across decades. Monotonic in the
        # prediction, so ordering — and AUROC — are unchanged.
        lsrc = np.log10(np.maximum(np.asarray(src, dtype=float), _LOG_FLOOR))
        lo, hi = np.percentile(lsrc, [low_q, high_q])
        if hi <= lo:
            lo, hi = float(np.min(lsrc)), float(np.max(lsrc))
        return ("log", lo, hi)
    raise ValueError(f"Unknown rescale mode '{mode}'. Use 'none', 'linear', 'log', or 'rank'.")


def _apply_rescale_ref(predictions: np.ndarray, ref, gamma: float = 1.0) -> np.ndarray:
    """Apply a reference built by _make_rescale_ref to map predictions → [0, 1]."""
    if ref[0] == "none":
        # Raw probabilities, no stretch — color directly encodes the prediction.
        scaled = np.asarray(predictions, dtype=float)
    elif ref[0] == "rank":
        src_sorted = ref[1]
        scaled = np.searchsorted(src_sorted, predictions, side="right") / len(src_sorted)
    elif ref[0] == "log":
        _, lo, hi = ref
        if hi <= lo:
            return np.zeros_like(predictions)
        lp = np.log10(np.maximum(np.asarray(predictions, dtype=float), _LOG_FLOOR))
        scaled = (lp - lo) / (hi - lo)
    else:  # "linear"
        _, lo, hi = ref
        if hi <= lo:
            return np.zeros_like(predictions)
        scaled = (predictions - lo) / (hi - lo)
    scaled = np.clip(scaled, 0.0, 1.0)
    if gamma != 1.0:
        scaled = np.power(scaled, gamma)
    return scaled


def compute_shared_rescale_ref(
    predictions_list,
    land_flat_indices: np.ndarray,
    lons: np.ndarray,
    lats: np.ndarray,
    range_gdf: gpd.GeoDataFrame,
    low_q: float = 10.0,
    high_q: float = 90.0,
    mode: str = "linear",
):
    """Build ONE rescale reference shared across several models for the same species.

    Pools the in-range predictions of every model in `predictions_list` and
    derives a single value→[0,1] mapping from that pool. Pass the result as
    `rescale_ref=` to make_prediction_map_fig for each model so all maps share
    the same color scale (option "B": same colorbar = same meaning across models,
    so a diffuse RF reads as uniformly moderate rather than being independently
    stretched to full contrast).
    """
    in_range = _in_range_mask(land_flat_indices, lons, lats, range_gdf)
    pool = [(p[in_range] if in_range.any() else p) for p in predictions_list]
    return _make_rescale_ref(np.concatenate(pool), low_q, high_q, mode)


def _rescale_predictions(
    predictions: np.ndarray,
    land_flat_indices: np.ndarray,
    lons: np.ndarray,
    lats: np.ndarray,
    range_gdf: gpd.GeoDataFrame,
    low_q: float = 10.0,
    high_q: float = 90.0,
    gamma: float = 1.0,
    mode: str = "linear",
    rescale_ref=None,
) -> np.ndarray:
    """
    Rescale predictions to [0, 1] for display.

    Parameters
    ----------
    mode : "none" | "linear" | "rank"
        "none"   : no stretch — raw probabilities map directly to color on the
                   fixed [0, 1] colorbar. Honest for low/no-separation species
                   (an RF sitting at ~0.5 reads as a uniform mid-tone instead of
                   being stretched). Unaffected by rescale_ref / RESCALE_SHARED.
        "linear" : robust min-max stretch between the low_q/high_q percentiles
                   of the in-range predictions (good when amplitude carries the
                   signal).
        "rank"   : map each cell to its percentile rank among in-range cells
                   (histogram equalization). Monotonic with the prediction, so
                   ordering — and therefore AUROC — is preserved, but low-
                   amplitude / skewed fields are shown at full contrast.
    low_q, high_q : percentile bounds for the "linear" stretch (ignored for "rank")
    gamma         : optional gamma compression (>1 darkens/attenuates highs);
                    applied in both modes
    rescale_ref   : optional reference from compute_shared_rescale_ref(). When
                    given, the value→[0,1] mapping comes from it (shared across
                    models) and low_q/high_q/mode are ignored; only gamma still
                    applies. When None (default), the mapping is computed from
                    this map's own in-range cells (per-map behavior).
    """
    if rescale_ref is None:
        in_range = _in_range_mask(land_flat_indices, lons, lats, range_gdf)
        src = predictions[in_range] if in_range.any() else predictions
        rescale_ref = _make_rescale_ref(src, low_q, high_q, mode)
    return _apply_rescale_ref(predictions, rescale_ref, gamma)


def _mask_predictions_to_range(
    predictions: np.ndarray,
    land_flat_indices: np.ndarray,
    lons: np.ndarray,
    lats: np.ndarray,
    range_gdf: gpd.GeoDataFrame,
) -> np.ndarray:
    """
    Set predictions outside the species range polygon to NaN.

    Returns a copy so callers can reuse the original prediction array.
    """
    W = len(lons)
    cell_lons = lons[land_flat_indices % W]
    cell_lats = lats[land_flat_indices // W]

    range_geom = range_gdf.unary_union
    try:
        from shapely import contains_xy
        in_range = contains_xy(range_geom, cell_lons, cell_lats)
    except ImportError:
        from shapely.vectorized import contains
        in_range = contains(range_geom, cell_lons, cell_lats)

    masked = predictions.astype(np.float32, copy=True)
    masked[~in_range] = np.nan
    return masked
