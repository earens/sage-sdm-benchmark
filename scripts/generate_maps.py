#!/usr/bin/env python
"""Generate a continental prediction map for one species from a trained model.

Works with any of the benchmark's model types (auto-detected by extension):
torch ``.ckpt``, sklearn ``.pkl``, or maxent ``.rds``. Writes a self-contained
interactive HTML map by default (pan/zoom, paper palette); add ``--png`` for a
static image (requires the optional ``kaleido`` package).

Examples
--------
    python scripts/generate_maps.py --model logs/run/checkpoints/best.ckpt \\
        --species "Quercus ilex" --out figures/quercus_ilex

    python scripts/generate_maps.py --model models/species_1234.pkl \\
        --species "Quercus ilex" --colorscheme rose_olive --png
"""

import argparse
import os

import rootutils

root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.visualizations.maps import (  # noqa: E402
    build_prediction_grid,
    compute_crop_bounds,
    get_species_id,
    load_bg_coords,
    load_species_data,
    load_world_map,
    make_prediction_map_fig,
    predict_with_model,
    set_continuous_colorscale,
)
from src.visualizations.maps.grid import DATA_DIR  # noqa: E402


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--model", required=True,
                   help="Path to a .ckpt (torch) / .pkl (sklearn) / .rds (maxent) model")
    p.add_argument("--species", required=True,
                   help="Species name (must match the native-range shapefile)")
    p.add_argument("--species-col", type=int, default=None,
                   help="Output column for multi-species torch models "
                        "(defaults to the species_id from species_properties_1km.csv)")
    p.add_argument("--data-dir", default=None,
                   help="Data root (default: data_dir from configs/local_default.yaml)")
    p.add_argument("--res", type=float, default=0.1, help="Grid resolution in degrees")
    p.add_argument("--colorscheme", default="magma_cb", help="Palette (see maps.COLORSCHEMES)")
    p.add_argument("--out", default=None,
                   help="Output path stem (default: figures/<species>)")
    p.add_argument("--no-rescale", action="store_true",
                   help="Show raw probabilities (skip the in-range contrast stretch)")
    p.add_argument("--png", action="store_true",
                   help="Also write a static PNG (requires the kaleido package)")
    p.add_argument("--rescale-low-q", type=float, default=10.0)
    p.add_argument("--rescale-high-q", type=float, default=98.0)
    p.add_argument("--rescale-gamma", type=float, default=1.35)
    p.add_argument("--rescale-mode", default="linear", choices=["linear", "log", "rank", "none"])
    p.add_argument("--score", type=float, default=None,
                   help="Metric value to show in the corner box (e.g. test AUROC)")
    p.add_argument("--score-label", default="AUROC")
    p.add_argument("--use-location", action="store_true",
                   help="Append the location encoding to the predictors "
                        "(only for models trained with use_location: true)")
    args = p.parse_args()

    data_dir = args.data_dir or DATA_DIR
    if data_dir is None:
        raise SystemExit("No data dir: pass --data-dir or set data_dir in configs/local_default.yaml")
    set_continuous_colorscale(args.colorscheme)

    species_col = args.species_col
    if species_col is None:
        species_col = get_species_id(args.species, data_dir)

    _, _, range_gdf = load_species_data(args.species, data_dir)
    crop_bounds = compute_crop_bounds(range_gdf)
    world_gdf = load_world_map(data_dir)
    bg_coords = load_bg_coords(data_dir)

    print(f"Building prediction grid for '{args.species}' (res={args.res}°) …")
    lons, lats, grid_coords, grid_predictors, land_flat_indices = build_prediction_grid(
        crop_bounds, world_gdf, bg_coords, data_dir, res=args.res)

    print(f"Predicting with {os.path.basename(args.model)} …")
    probs = predict_with_model(args.model, grid_coords, grid_predictors,
                               use_location=args.use_location, species_col=species_col)

    fig = make_prediction_map_fig(
        lons, lats, land_flat_indices, probs, crop_bounds, range_gdf, world_gdf,
        title=args.species, rescale=not args.no_rescale,
        rescale_low_q=args.rescale_low_q, rescale_high_q=args.rescale_high_q,
        rescale_gamma=args.rescale_gamma, rescale_mode=args.rescale_mode,
        score=args.score, score_label=args.score_label)

    out = args.out or os.path.join("figures", args.species.replace(" ", "_"))
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)

    html_path = f"{out}.html"
    fig.write_html(html_path, config=dict(scrollZoom=True))
    print(f"Saved: {html_path}")

    if args.png:
        try:
            fig.write_image(f"{out}.png", format="png", engine="kaleido", scale=2)
            print(f"Saved: {out}.png")
        except Exception as e:
            print(f"PNG export failed ({e}). Install kaleido:  pip install kaleido")


if __name__ == "__main__":
    main()
