"""Prediction map utilities: compute layer (grid) + render layer (render)."""

from src.visualizations.maps.grid import (
    build_prediction_grid,
    compute_crop_bounds,
    compute_shared_rescale_ref,
    get_species_id,
    load_bg_coords,
    load_model_score,
    load_species_data,
    load_splotopen_species_data,
    load_world_map,
    predict_with_live_model,
    predict_with_model,
    predictions_to_grid,
)
from src.visualizations.maps.render import (
    COLORSCHEMES,
    make_observations_map_fig,
    make_pa_observations_map_fig,
    make_po_observations_map_fig,
    make_prediction_map_fig,
    make_presence_map_fig,
    make_target_map_fig,
    save_fig,
    set_continuous_colorscale,
)
from src.visualizations.maps.slider import add_threshold_slider, make_interactive_map
from src.visualizations.maps.wandb_maps import generate_test_maps, generate_wandb_maps

__all__ = [
    "get_species_id",
    "load_world_map",
    "load_species_data",
    "load_splotopen_species_data",
    "compute_crop_bounds",
    "load_bg_coords",
    "build_prediction_grid",
    "predict_with_model",
    "predict_with_live_model",
    "load_model_score",
    "predictions_to_grid",
    "compute_shared_rescale_ref",
    "COLORSCHEMES",
    "set_continuous_colorscale",
    "make_prediction_map_fig",
    "make_presence_map_fig",
    "make_target_map_fig",
    "make_observations_map_fig",
    "make_po_observations_map_fig",
    "make_pa_observations_map_fig",
    "save_fig",
    "add_threshold_slider",
    "make_interactive_map",
    "generate_wandb_maps",
    "generate_test_maps",
]
