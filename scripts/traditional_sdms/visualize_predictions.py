"""Continental prediction maps for the traditional SDM baselines.

Thin wrapper over ``src.visualizations.maps``: builds the (unnormalized) grid,
lets the caller's ``transforms`` + ``predict_func`` produce per-cell probabilities
(so each model family keeps its own normalization — z-scored for sklearn, raw for
MaxEnt), then renders an interactive map with the threshold slider and saves HTML.
"""

import os

import geopandas as gpd
import numpy as np
import pandas as pd
import torch

from src.models.resnet import encode_locations
from src.visualizations.maps import (
    build_prediction_grid,
    compute_crop_bounds,
    load_bg_coords,
    load_world_map,
    make_interactive_map,
)


def generate_maps_for_species(
    model,
    species_id: int,
    species_name: str,
    data_dir: str,
    output_dir: str,
    continent: str = "europe",
    predictor_subset: dict = None,
    transforms=None,
    use_location: bool = False,
    positive_locations=None,
    obs_locations: np.ndarray = None,
    obs_targets: np.ndarray = None,
    obs_probs: np.ndarray = None,
    threshold: float = 0.5,
    tn_subsample: int = None,
    mode: str = "train",
    predictions_h5: str = None,
    predict_func=None,
) -> str:
    """Render an interactive prediction map for one species of a traditional SDM.

    Args:
        model: unused (kept for signature compatibility); prediction goes through
            ``predict_func``.
        species_id: numeric species ID (column into multi-species arrays).
        species_name: species name (only used for logging).
        data_dir: root data directory.
        output_dir: directory the HTML map is written to.
        continent: label used in the output filename.
        predictor_subset: ``{predictor_group: [col indices]}`` for the grid.
        transforms: optional transform applied to the raw grid predictors
            (e.g. the datamodule normalizer for sklearn, NaN-imputation for MaxEnt).
        use_location: prepend the sin/cos location encoding to the grid features.
        obs_locations / obs_targets / obs_probs: the species' presence-absence
            evaluation points (lon/lat, 0/1 labels, predicted probs) for the
            confusion-point threshold slider.
        tn_subsample: cap on true-negative points drawn in the slider.
        mode: label used in the output filename (e.g. ``"train"`` / ``"eval"``).
        predict_func: ``callable(grid_features) -> probabilities`` — the trained
            sklearn / MaxEnt model wrapped as a closure by the eval script.

    Returns the path of the written HTML file.
    """
    if predict_func is None:
        raise ValueError("generate_maps_for_species requires predict_func (features -> probabilities).")

    # Resolve the range-map name (native-range shapefile filename).
    species_names_df = pd.read_csv(os.path.join(data_dir, "targets", "species_names.csv"))
    true_species_name = species_names_df.loc[
        species_names_df["Index"] == species_id, "Species Name"
    ].values[0]

    range_path = os.path.join(data_dir, "native_ranges", f"{true_species_name}.shp")
    range_gdf = gpd.read_file(range_path).to_crs("EPSG:4326")
    crop_bounds = compute_crop_bounds(range_gdf)
    world_gdf = load_world_map(data_dir)
    bg_coords = load_bg_coords(data_dir)

    # Raw grid — the caller's transform / predict_func control normalization.
    lons, lats, grid_coords, grid_raw, land_flat_indices = build_prediction_grid(
        crop_bounds, world_gdf, bg_coords, data_dir, predictor_subset, res=0.1, normalize=False
    )

    grid_tensor = torch.from_numpy(grid_raw).float()
    if transforms is not None:
        grid_tensor = transforms(grid_tensor)
    if use_location:
        enc = encode_locations(torch.from_numpy(grid_coords).float())
        grid_features = torch.cat([enc, grid_tensor], dim=-1).numpy()
    else:
        grid_features = grid_tensor.numpy()

    predictions = np.asarray(predict_func(grid_features)).ravel()

    fig = make_interactive_map(
        lons, lats, land_flat_indices, predictions,
        obs_locations, obs_targets, obs_probs,
        crop_bounds, range_gdf, world_gdf,
        title=true_species_name, tn_subsample=tn_subsample or 10000,
    )

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(
        output_dir, f"{continent}_{true_species_name}_{mode}.html".replace(" ", "_")
    )
    fig.write_html(out_path, config=dict(scrollZoom=True))
    print(f"  Saved map: {out_path}")
    return out_path
