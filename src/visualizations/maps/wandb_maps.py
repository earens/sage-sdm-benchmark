"""Map generation for the training loop.

Orchestrates the compute (:mod:`grid`) and render (:mod:`render`/:mod:`slider`)
layers to produce continental prediction maps for a live, in-memory model:

- :func:`generate_wandb_maps` — static prediction maps logged interactively to
  wandb every validation epoch.
- :func:`generate_test_maps` — full interactive maps (with a threshold slider and
  TP/TN/FP/FN confusion points) written to HTML at test time.

The expensive per-species grid (background nearest-neighbour lookup + predictor
loading) is cached to disk so only the forward pass runs each call.
"""

import hashlib
import logging
import os
import pickle

import geopandas as gpd
import numpy as np
import pandas as pd

from src.visualizations.maps.grid import (
    build_prediction_grid,
    compute_crop_bounds,
    load_bg_coords,
    load_world_map,
    predict_with_live_model,
)
from src.visualizations.maps.render import make_prediction_map_fig, set_continuous_colorscale
from src.visualizations.maps.slider import make_interactive_map

logger = logging.getLogger(__name__)


def _species_id_to_name(data_dir: str) -> dict:
    """Map species_id -> species_name (matching the native-range shapefile names)."""
    proxies = pd.read_csv(os.path.join(data_dir, "targets", "species_properties_1km.csv"))
    return dict(zip(proxies["species_id"], proxies["species_name"]))


def _load_range_gdf(data_dir: str, species_name: str) -> gpd.GeoDataFrame:
    shp = os.path.join(data_dir, "native_ranges", f"{species_name}.shp")
    return gpd.read_file(shp).to_crs("EPSG:4326")


def _grid_cache_key(species_id, crop_bounds, predictors_subset, res) -> str:
    payload = repr((
        int(species_id),
        tuple(round(float(b), 4) for b in crop_bounds),
        {k: list(v) for k, v in sorted(predictors_subset.items())},
        float(res),
    ))
    return hashlib.md5(payload.encode()).hexdigest()


def _build_species_grid(species_id, name, data_dir, world_gdf, bg_coords,
                        predictors_subset, res, cache_dir):
    """Return (lons, lats, grid_coords, grid_predictors, land_flat_indices,
    crop_bounds, range_gdf), building + disk-caching the grid on first use."""
    range_gdf = _load_range_gdf(data_dir, name)
    crop_bounds = compute_crop_bounds(range_gdf)

    key = _grid_cache_key(species_id, crop_bounds, predictors_subset, res)
    cache_file = os.path.join(cache_dir, f"{key}.pkl")
    if os.path.exists(cache_file):
        with open(cache_file, "rb") as f:
            grid = pickle.load(f)
    else:
        grid = build_prediction_grid(crop_bounds, world_gdf, bg_coords, data_dir,
                                     predictors_subset, res)
        with open(cache_file, "wb") as f:
            pickle.dump(grid, f)

    lons, lats, grid_coords, grid_predictors, land_flat_indices = grid
    return lons, lats, grid_coords, grid_predictors, land_flat_indices, crop_bounds, range_gdf


def _prepare(model, datamodule, cache_dir, colorscheme):
    """Shared setup: resolve palette, data_dir, world/bg data, id->name map."""
    if colorscheme is not None:
        set_continuous_colorscale(colorscheme)
    data_dir = datamodule.hparams.data_dir
    predictors_subset = datamodule.predictors_subset
    os.makedirs(cache_dir, exist_ok=True)
    return (data_dir, predictors_subset, _species_id_to_name(data_dir),
            load_world_map(data_dir), load_bg_coords(data_dir))


def generate_wandb_maps(model, datamodule, configs,
                        cache_dir="cache/wandb_maps", colorscheme=None) -> dict:
    """Static continental prediction maps for the configured species.

    Parameters
    ----------
    model      : live torch net taking (locations, predictors) -> per-species logits
    datamodule : the LightningDataModule (for data_dir / predictors_subset)
    configs    : list of dicts, each with ``continent`` (label), ``species_ids``,
                 and optional ``res`` (grid resolution in degrees, default 0.1)
    colorscheme: optional COLORSCHEMES key to apply before rendering

    Returns ``{"maps/{continent}/{species_name}": wandb.Plotly}``.
    """
    import wandb

    data_dir, predictors_subset, id_to_name, world_gdf, bg_coords = _prepare(
        model, datamodule, cache_dir, colorscheme)

    images: dict = {}
    for cfg in configs:
        continent = cfg.get("continent", "")
        res = float(cfg.get("res", 0.1))
        for species_id in cfg.get("species_ids", []):
            name = id_to_name.get(species_id, f"species_{species_id}")
            try:
                lons, lats, gc, gp, lfi, crop_bounds, range_gdf = _build_species_grid(
                    species_id, name, data_dir, world_gdf, bg_coords, predictors_subset, res, cache_dir)
                probs = predict_with_live_model(model, gc, gp, species_col=int(species_id))
                fig = make_prediction_map_fig(lons, lats, lfi, probs, crop_bounds,
                                              range_gdf, world_gdf, title=name)
                images[f"maps/{continent}/{name}"] = wandb.Plotly(fig)
            except Exception as e:
                logger.warning(f"Skipping wandb map for species {species_id} ({name}): {e}")
    return images


def generate_test_maps(model, datamodule, configs, obs_locations, obs_targets, obs_probs,
                       output_dir, cache_dir="cache/wandb_maps", colorscheme=None) -> list:
    """Interactive HTML maps (threshold slider + confusion points) per species.

    ``obs_locations`` (N, 2), ``obs_targets`` and ``obs_probs`` (N, num_species)
    are the accumulated presence-absence test predictions; for each species the
    in-range rows (target != -1) become the slider's confusion points, while the
    continental heatmap is (re)predicted from the live model.

    Returns the list of written HTML paths.
    """
    data_dir, predictors_subset, id_to_name, world_gdf, bg_coords = _prepare(
        model, datamodule, cache_dir, colorscheme)
    os.makedirs(output_dir, exist_ok=True)
    obs_locations = np.asarray(obs_locations)

    written = []
    for cfg in configs:
        continent = cfg.get("continent", "")
        res = float(cfg.get("res", 0.1))
        for species_id in cfg.get("species_ids", []):
            name = id_to_name.get(species_id, f"species_{species_id}")
            try:
                lons, lats, gc, gp, lfi, crop_bounds, range_gdf = _build_species_grid(
                    species_id, name, data_dir, world_gdf, bg_coords, predictors_subset, res, cache_dir)
                grid_probs = predict_with_live_model(model, gc, gp, species_col=int(species_id))

                col_t = np.asarray(obs_targets[:, species_id]).astype(int)
                col_p = np.asarray(obs_probs[:, species_id]).astype(float)
                in_range = col_t >= 0  # drop masked (ignore_index = -1) rows

                fig = make_interactive_map(
                    lons, lats, lfi, grid_probs,
                    obs_locations[in_range], col_t[in_range], col_p[in_range],
                    crop_bounds, range_gdf, world_gdf, title=name)

                out = os.path.join(output_dir, f"{continent}_{name}.html".replace(" ", "_"))
                fig.write_html(out, config=dict(scrollZoom=True))
                written.append(out)
            except Exception as e:
                logger.warning(f"Skipping test map for species {species_id} ({name}): {e}")
    return written
