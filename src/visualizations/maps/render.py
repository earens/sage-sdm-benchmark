"""Render layer for prediction maps.

Plotly figure builders that turn prediction grids and observation coordinates
into publication-style maps:
  - Prediction probability heatmaps (with range outline and country borders)
  - Presence / target scatter maps
  - PO / PA observation scatter maps
  - Saving figures to disk as PNG (+ optionally HTML)

The compute layer (data loading, grid building, model inference, rescaling)
lives in src.visualizations.maps.grid.
"""

import os

import geopandas as gpd
import numpy as np
import plotly.graph_objects as go
from shapely.geometry import box as shapely_box

from src.visualizations.maps.grid import (
    _mask_predictions_to_range,
    _rescale_predictions,
    predictions_to_grid,
)

# Colour schemes for prediction heatmaps
COLORSCHEMES = {
    "rose_olive": [
        [0.00, "#FAF6F4"],
        [0.25, "#E7CEC6"],
        [0.50, "#CFAF9F"],
        [0.75, "#88906A"],
        [1.00, "#4D5A3A"],
    ],
    "rose_brown": [
        [0.00, "#FAF6F4"],
        [0.25, "#E8CDC8"],
        [0.50, "#D1AFA7"],
        [0.75, "#9D6E62"],
        [1.00, "#5D4037"],
    ],
    "sand_plum": [
        [0.00, "#FAF6F4"],
        [0.25, "#E8D8CF"],
        [0.50, "#CCB7B0"],
        [0.75, "#8C6D7A"],
        [1.00, "#4B2E39"],
    ],
    "gray_fire": [
        [0.00, "#F5F5F5"],
        [0.25, "#D8D8D8"],
        [0.50, "#B5AFAF"],
        [0.75, "#C57A5A"],
        [1.00, "#8E3B2E"],
    ],
    # Colorblind-friendly sequential schemes (high luminance contrast)
    "magma_cb": [
        [0.00, "#FAF6F4"],
        [0.25, "#fd9f6c"],
        [0.50, "#cd4071"],
        [0.75, "#7e2f8e"],
        [1.00, "#2a1155"],
    ],
    "inferno_cb": [
        [0.00, "#FAF6F4"],
        [0.25, "#fca50a"],
        [0.50, "#dd513a"],
        [0.75, "#a63f77"],
        [1.00, "#3b1a5a"],
    ],
    "rose_mute_cb": [
        [0.00, "#f8f6f2"],
        [0.25, "#d6cdc4"],
        [0.50, "#b5aaa3"],
        [0.75, "#8b7a8f"],
        [1.00, "#4b3f5f"],
    ],
}

CONTINUOUS_COLORSCALE = COLORSCHEMES["magma_cb"]


def set_continuous_colorscale(colorscheme="rose_olive"):
    """
    Set the global prediction-map colorscale.

    Parameters
    ----------
    colorscheme : str | list
        Either a key from COLORSCHEMES or a Plotly colorscale list.
    """
    global CONTINUOUS_COLORSCALE
    if isinstance(colorscheme, str):
        if colorscheme not in COLORSCHEMES:
            options = ", ".join(sorted(COLORSCHEMES.keys()))
            raise ValueError(f"Unknown colorscheme '{colorscheme}'. Choose one of: {options}")
        CONTINUOUS_COLORSCALE = COLORSCHEMES[colorscheme]
    else:
        CONTINUOUS_COLORSCALE = colorscheme

# CONTINUOUS_COLORSCALE = [
#     [0.00, "#FAF6F4"],
#     [0.25, "#DDD0CD"],
#     [0.50, "#B8AEBE"],
#     [0.75, "#6E748E"],
#     [1.00, "#3D405B"],
# ]


# ---------------------------------------------------------------------------
# Shared figure helpers  (mirrors species_showcase.ipynb style)
# ---------------------------------------------------------------------------

def _clip_land(world_gdf: gpd.GeoDataFrame, crop_bounds):
    minx, miny, maxx, maxy = crop_bounds
    view_box = shapely_box(minx, miny, maxx, maxy)
    clipped  = gpd.clip(world_gdf, view_box)
    clipped  = clipped[~clipped.geometry.is_empty & clipped.geometry.is_valid]
    clipped  = clipped.explode(index_parts=False)
    return clipped[clipped.geometry.type == "Polygon"]


def _add_land_shapes(fig: go.Figure, land_gdf: gpd.GeoDataFrame):
    """Add land-fill shapes (below traces) — same style as species_showcase."""
    shapes = []
    for _, row in land_gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty or geom.geom_type != "Polygon":
            continue
        x, y = geom.exterior.coords.xy
        path = "M " + " L ".join(f"{xi},{yi}" for xi, yi in zip(x, y)) + " Z"
        shapes.append(
            dict(
                type="path",
                path=path,
                fillcolor="#F9F4F3",
                line=dict(color="#c0bdb6", width=0.5),
                layer="below",
            )
        )
    return shapes


def _add_border_traces(fig: go.Figure, land_gdf: gpd.GeoDataFrame):
    """Add country border line traces on top of data layers."""
    for _, row in land_gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty or geom.geom_type != "Polygon":
            continue
        x, y = geom.exterior.coords.xy
        fig.add_trace(
            go.Scatter(
                x=list(x), y=list(y), mode="lines",
                line=dict(color="black", width=0.8),
                showlegend=False, hoverinfo="skip",
            )
        )


def _add_range_outline(fig: go.Figure, range_gdf: gpd.GeoDataFrame):
    """Add the species native-range polygon outline."""
    first = True
    for _, row in range_gdf.iterrows():
        geom = row.geometry
        polys = list(geom.geoms) if geom.geom_type == "MultiPolygon" else [geom]
        for poly in polys:
            x, y = poly.exterior.coords.xy
            fig.add_trace(
                go.Scatter(
                    x=list(x), y=list(y), mode="lines",
                    line=dict(color="#E07A5F", width=4),
                    name="Native range",
                    showlegend=first,
                    legendgroup="range",
                    fill="toself",
                    fillcolor="rgba(84,105,173,0.0)",
                )
            )
            first = False


def _base_layout(title: str, crop_bounds, base_width: int = 800) -> dict:
    minx, miny, maxx, maxy = crop_bounds
    lon_range = maxx - minx
    lat_range = maxy - miny
    aspect    = max(lon_range / max(lat_range, 1e-6), 0.25)
    height    = max(int(base_width / aspect), 500)
    return dict(
        title=dict(text=f"<b>{title}</b>", font=dict(size=16)),
        xaxis=dict(
            range=[minx, maxx], title=None,
            showgrid=False, zeroline=False,
            showticklabels=False,
            ticks='',
        ),
        yaxis=dict(
            range=[miny, maxy], title=None,
            showgrid=False, zeroline=False,
            scaleanchor="x", scaleratio=1,
            showticklabels=False,
            ticks='',
        ),
        width=base_width, height=height,
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=0, r=0, t=0, b=0),
        legend=dict(
            x=1.02, y=1, bordercolor="black", borderwidth=1,
            bgcolor="rgba(255,255,255,0.8)",
        ),
        dragmode="pan", hovermode="closest",
        font=dict(family="Arial, sans-serif", size=13),
    )


# ---------------------------------------------------------------------------
# Plot functions
# ---------------------------------------------------------------------------

def make_prediction_map_fig(
    lons: np.ndarray,
    lats: np.ndarray,
    land_flat_indices: np.ndarray,
    predictions: np.ndarray,
    crop_bounds,
    range_gdf: gpd.GeoDataFrame,
    world_gdf: gpd.GeoDataFrame,
    title: str = "Prediction",
    score: float = None,
    score_label: str = "AUROC",
    rescale: bool = True,
    mask_to_range: bool = True,
    rescale_low_q: float = 10.0,
    rescale_high_q: float = 98.0,
    rescale_gamma: float = 1.35,
    rescale_mode: str = "linear",
    rescale_ref=None,
) -> go.Figure:
    """
    Prediction probability heatmap with range outline and country borders.

    Parameters
    ----------
    score       : optional metric value shown in a box in the top-right corner
    score_label : label shown next to the score (default 'AUROC')
    rescale     : if True, min-max rescale predictions to [0, 1] for display
                  using only in-range cells to compute min/max (improves
                  contrast; does not affect stored values or metrics)
    """
    if rescale:
        predictions = _rescale_predictions(
            predictions,
            land_flat_indices,
            lons,
            lats,
            range_gdf,
            low_q=rescale_low_q,
            high_q=rescale_high_q,
            gamma=rescale_gamma,
            mode=rescale_mode,
            rescale_ref=rescale_ref,
        )
    if mask_to_range:
        predictions = _mask_predictions_to_range(predictions, land_flat_indices, lons, lats, range_gdf)
    probs_map = predictions_to_grid(lons, lats, land_flat_indices, predictions)
    land_gdf  = _clip_land(world_gdf, crop_bounds)

    fig = go.Figure()

    # Heatmap must come before shapes so shapes (land fill) go below it
    fig.add_trace(
        go.Heatmap(
            z=probs_map, x=lons, y=lats,
            colorscale=CONTINUOUS_COLORSCALE,
            zmin=0, zmax=1,
            #colorbar=dict(title="Probability", thickness=15),
            showscale=False,
            connectgaps=False,
        )
    )

    _add_range_outline(fig, range_gdf)
    _add_border_traces(fig, land_gdf)

    # Add ocean shape below everything
    minx, miny, maxx, maxy = crop_bounds
    ocean_shape = dict(
        type="rect",
        xref="x", yref="y",
        x0=minx, x1=maxx, y0=miny, y1=maxy,
        fillcolor="#d4e4f7",
        line=dict(width=0),
        layer="below",
    )

    layout = _base_layout(title, crop_bounds)
    # Add ocean first, then land shapes
    layout["shapes"] = [ocean_shape] + _add_land_shapes(fig, land_gdf)
    fig.update_layout(**layout)

    # Score box intentionally not drawn — the metric is printed to stdout in the
    # notebook (see the "Run predictions" cell) so it can be added to the plot
    # manually. `score`/`score_label` are kept in the signature for compatibility.

    fig.layout.update(showlegend=False)

    return fig


def make_presence_map_fig(
    lons: np.ndarray,
    lats: np.ndarray,
    land_flat_indices: np.ndarray,
    predictions: np.ndarray,
    presence_coords: np.ndarray,
    crop_bounds,
    range_gdf: gpd.GeoDataFrame,
    world_gdf: gpd.GeoDataFrame,
    title: str = "Presence points",
    rescale: bool = True,
    mask_to_range: bool = True,
) -> go.Figure:
    """Scatter map of species presence observations."""

    if rescale:
        predictions = _rescale_predictions(predictions, land_flat_indices, lons, lats, range_gdf)
    if mask_to_range:
        predictions = _mask_predictions_to_range(predictions, land_flat_indices, lons, lats, range_gdf)
    probs_map = predictions_to_grid(lons, lats, land_flat_indices, predictions)
    land_gdf  = _clip_land(world_gdf, crop_bounds)
    fig      = go.Figure()

    # Heatmap must come before shapes so shapes (land fill) go below it
    fig.add_trace(
        go.Heatmap(
            z=probs_map, x=lons, y=lats,
            colorscale=CONTINUOUS_COLORSCALE,
            zmin=0, zmax=1,
            #colorbar=dict(title="Probability", thickness=15),
            showscale=False,
            connectgaps=False,
        )
    )

    _add_range_outline(fig, range_gdf)
    _add_border_traces(fig, land_gdf)

    # Add ocean shape below everything
    minx, miny, maxx, maxy = crop_bounds
    ocean_shape = dict(
        type="rect",
        xref="x", yref="y",
        x0=minx, x1=maxx, y0=miny, y1=maxy,
        fillcolor="#d4e4f7",
        line=dict(width=0),
        layer="below",
    )

    fig.add_trace(
        go.Scatter(
            x=presence_coords[:, 0],
            y=presence_coords[:, 1],
            mode="markers",
            marker=dict(
                color="#E07A5F", size=5, opacity=0.8,
                line=dict(width=0.2, color="white"),
            ),
            name=f"Presences ({len(presence_coords):,})",
        )
    )

    layout = _base_layout(title, crop_bounds)
    #layout["shapes"] = _add_land_shapes(fig, land_gdf)
    layout["shapes"] = [ocean_shape] + _add_land_shapes(fig, land_gdf)

    fig.update_layout(**layout)
    fig.layout.update(showlegend=False)

    return fig


def make_target_map_fig(
    lons: np.ndarray,
    lats: np.ndarray,
    land_flat_indices: np.ndarray,
    predictions: np.ndarray,
    in_range_coords: np.ndarray,
    presence_coords: np.ndarray,
    crop_bounds,
    range_gdf: gpd.GeoDataFrame,
    world_gdf: gpd.GeoDataFrame,
    title: str = "Targets",
    subsample_absence: int = 5000,
    rng_seed: int = 0,
    rescale: bool = True,
    mask_to_range: bool = True,
) -> go.Figure:
    """
    Scatter map coloured by target label.

    Green = presence (label 1).
    Red   = in-range cell without a presence observation (label 0), subsampled
            to *subsample_absence* points so the plot stays responsive.
    """
    # Absence = in-range cells NOT in presence set
    presence_set = set(map(tuple, presence_coords.tolist()))
    absence_mask = np.array(
        [tuple(c) not in presence_set for c in in_range_coords.tolist()]
    )
    absence_coords = in_range_coords[absence_mask]

    if len(absence_coords) > subsample_absence:
        rng   = np.random.default_rng(rng_seed)
        idx   = rng.choice(len(absence_coords), size=subsample_absence, replace=False)
        absence_coords = absence_coords[idx]

    if rescale:
        predictions = _rescale_predictions(predictions, land_flat_indices, lons, lats, range_gdf)
    if mask_to_range:
        predictions = _mask_predictions_to_range(predictions, land_flat_indices, lons, lats, range_gdf)
    probs_map = predictions_to_grid(lons, lats, land_flat_indices, predictions)
    land_gdf = _clip_land(world_gdf, crop_bounds)
    fig      = go.Figure()

        # Heatmap must come before shapes so shapes (land fill) go below it

    fig.add_trace(
        go.Heatmap(
            z=probs_map, x=lons, y=lats,
            colorscale=CONTINUOUS_COLORSCALE,
            zmin=0, zmax=1,
            #colorbar=dict(title="Probability", thickness=15),
            showscale=False,
            connectgaps=False,
        )
    )


    _add_range_outline(fig, range_gdf)
    _add_border_traces(fig, land_gdf)

    # Add ocean shape below everything
    minx, miny, maxx, maxy = crop_bounds
    ocean_shape = dict(
        type="rect",
        xref="x", yref="y",
        x0=minx, x1=maxx, y0=miny, y1=maxy,
        fillcolor="#d4e4f7",
        line=dict(width=0),
        layer="below",
    )


    # Absences (red, drawn first so presences appear on top)
    fig.add_trace(
        go.Scatter(
            x=absence_coords[:, 0],
            y=absence_coords[:, 1],
            mode="markers",
            marker=dict(color="#A0A0A8", size=4, opacity=0.7, line=dict(width=0.1, color="white")),
            name=f"Absence ({len(absence_coords):,})",
        )
    )
    # Presences (green)
    fig.add_trace(
        go.Scatter(
            x=presence_coords[:, 0],
            y=presence_coords[:, 1],
            mode="markers",
            marker=dict(color="#57A743", size=5, opacity=1.0, line=dict(width=0.1, color="white")),
            name=f"Presence ({len(presence_coords):,})",
        )
    )

    layout = _base_layout(title, crop_bounds)
    layout["shapes"] = [ocean_shape] + _add_land_shapes(fig, land_gdf)
    fig.update_layout(**layout)
    fig.layout.update(showlegend=False)

    return fig


def make_observations_map_fig(
    po_presence_coords: np.ndarray,
    pa_in_range_coords: np.ndarray,
    pa_presence_coords: np.ndarray,
    crop_bounds,
    range_gdf: gpd.GeoDataFrame,
    world_gdf: gpd.GeoDataFrame,
    title: str = "Observations",
    subsample_pa_absence: int = 5000,
    rng_seed: int = 0,
) -> go.Figure:
    """
    Plot PO presences and PA presences/absences in one map (no heatmap).
    """
    pa_presence_set = set(map(tuple, pa_presence_coords.tolist()))
    pa_absence_mask = np.array(
        [tuple(c) not in pa_presence_set for c in pa_in_range_coords.tolist()]
    )
    pa_absence_coords = pa_in_range_coords[pa_absence_mask]

    if len(pa_absence_coords) > subsample_pa_absence:
        rng = np.random.default_rng(rng_seed)
        idx = rng.choice(len(pa_absence_coords), size=subsample_pa_absence, replace=False)
        pa_absence_coords = pa_absence_coords[idx]

    land_gdf = _clip_land(world_gdf, crop_bounds)
    fig = go.Figure()

    _add_range_outline(fig, range_gdf)
    _add_border_traces(fig, land_gdf)

    # PA absences first
    fig.add_trace(
        go.Scatter(
            x=pa_absence_coords[:, 0],
            y=pa_absence_coords[:, 1],
            mode="markers",
            marker=dict(color="#A0A0A8", size=12, opacity=0.7, line=dict(width=0.35, color="white")),
            name=f"PA absence ({len(pa_absence_coords):,})",
        )
    )

    # PO presences second
    fig.add_trace(
        go.Scatter(
            x=po_presence_coords[:, 0],
            y=po_presence_coords[:, 1],
            mode="markers",
            marker=dict(color="#E07A5F", size=14, opacity=0.9, line=dict(width=0.35, color="white")),
            name=f"PO presence ({len(po_presence_coords):,})",
        )
    )

    # PA (test) presences last, on top of everything
    fig.add_trace(
        go.Scatter(
            x=pa_presence_coords[:, 0],
            y=pa_presence_coords[:, 1],
            mode="markers",
            marker=dict(color="#57A743", size=14, opacity=1.0, line=dict(width=0.35, color="white")),
            name=f"PA presence ({len(pa_presence_coords):,})",
        )
    )

    minx, miny, maxx, maxy = crop_bounds
    ocean_shape = dict(
        type="rect",
        xref="x", yref="y",
        x0=minx, x1=maxx, y0=miny, y1=maxy,
        fillcolor="#d4e4f7",
        line=dict(width=0),
        layer="below",
    )

    layout = _base_layout(title, crop_bounds)
    layout["shapes"] = [ocean_shape] + _add_land_shapes(fig, land_gdf)
    fig.update_layout(**layout)
    fig.layout.update(showlegend=False)

    return fig


def make_po_observations_map_fig(
    po_presence_coords: np.ndarray,
    crop_bounds,
    range_gdf: gpd.GeoDataFrame,
    world_gdf: gpd.GeoDataFrame,
    title: str = "PO observations",
) -> go.Figure:
    """
    Plot presence-only (GBIF) presences alone, on the same base map style as
    make_observations_map_fig. Splitting PO and PA into separate maps avoids the
    overplotting of the combined map (too many points on top of each other).
    """
    land_gdf = _clip_land(world_gdf, crop_bounds)
    fig = go.Figure()

    _add_range_outline(fig, range_gdf)
    _add_border_traces(fig, land_gdf)

    fig.add_trace(
        go.Scatter(
            x=po_presence_coords[:, 0],
            y=po_presence_coords[:, 1],
            mode="markers",
            marker=dict(color="#E07A5F", size=14, opacity=0.9, line=dict(width=0.35, color="white")),
            name=f"PO presence ({len(po_presence_coords):,})",
        )
    )

    minx, miny, maxx, maxy = crop_bounds
    ocean_shape = dict(
        type="rect",
        xref="x", yref="y",
        x0=minx, x1=maxx, y0=miny, y1=maxy,
        fillcolor="#d4e4f7",
        line=dict(width=0),
        layer="below",
    )

    layout = _base_layout(title, crop_bounds)
    layout["shapes"] = [ocean_shape] + _add_land_shapes(fig, land_gdf)
    fig.update_layout(**layout)
    fig.layout.update(showlegend=False)

    return fig


def make_pa_observations_map_fig(
    pa_in_range_coords: np.ndarray,
    pa_presence_coords: np.ndarray,
    crop_bounds,
    range_gdf: gpd.GeoDataFrame,
    world_gdf: gpd.GeoDataFrame,
    title: str = "PA observations",
    subsample_pa_absence: int = 5000,
    rng_seed: int = 0,
) -> go.Figure:
    """
    Plot presence-absence (sPlotOpen) presences and absences alone, on the same
    base map style as make_observations_map_fig. Companion to
    make_po_observations_map_fig — one map per data type so points don't overlap.
    """
    pa_presence_set = set(map(tuple, pa_presence_coords.tolist()))
    pa_absence_mask = np.array(
        [tuple(c) not in pa_presence_set for c in pa_in_range_coords.tolist()]
    )
    pa_absence_coords = pa_in_range_coords[pa_absence_mask]

    if len(pa_absence_coords) > subsample_pa_absence:
        rng = np.random.default_rng(rng_seed)
        idx = rng.choice(len(pa_absence_coords), size=subsample_pa_absence, replace=False)
        pa_absence_coords = pa_absence_coords[idx]

    land_gdf = _clip_land(world_gdf, crop_bounds)
    fig = go.Figure()

    _add_range_outline(fig, range_gdf)
    _add_border_traces(fig, land_gdf)

    # PA absences first (drawn underneath the presences)
    fig.add_trace(
        go.Scatter(
            x=pa_absence_coords[:, 0],
            y=pa_absence_coords[:, 1],
            mode="markers",
            marker=dict(color="#A0A0A8", size=12, opacity=0.7, line=dict(width=0.35, color="white")),
            name=f"PA absence ({len(pa_absence_coords):,})",
        )
    )

    # PA presences on top
    fig.add_trace(
        go.Scatter(
            x=pa_presence_coords[:, 0],
            y=pa_presence_coords[:, 1],
            mode="markers",
            marker=dict(color="#57A743", size=14, opacity=1.0, line=dict(width=0.35, color="white")),
            name=f"PA presence ({len(pa_presence_coords):,})",
        )
    )

    minx, miny, maxx, maxy = crop_bounds
    ocean_shape = dict(
        type="rect",
        xref="x", yref="y",
        x0=minx, x1=maxx, y0=miny, y1=maxy,
        fillcolor="#d4e4f7",
        line=dict(width=0),
        layer="below",
    )

    layout = _base_layout(title, crop_bounds)
    layout["shapes"] = [ocean_shape] + _add_land_shapes(fig, land_gdf)
    fig.update_layout(**layout)
    fig.layout.update(showlegend=False)

    return fig


# ---------------------------------------------------------------------------
# Save helper
# ---------------------------------------------------------------------------

def save_fig(
    fig: go.Figure,
    path_stem: str,
    save_html: bool = False,
    scale: int = 3,
) -> None:
    """
    Save a plotly figure to *path_stem*.png (and optionally *path_stem*.html).

    Parameters
    ----------
    fig       : plotly Figure
    path_stem : full path without extension, e.g. '../figures/Prunus_avium_model1'
    save_html : also write an interactive HTML file
    scale     : kaleido PNG scale factor (higher = higher resolution)
    """
    os.makedirs(os.path.dirname(os.path.abspath(path_stem)), exist_ok=True)
    png_path = f"{path_stem}.png"
    fig.write_image(png_path, format="png", engine="kaleido", scale=scale)
    print(f"  Saved: {png_path}")
    if save_html:
        html_path = f"{path_stem}.html"
        fig.write_html(html_path, config=dict(scrollZoom=True))
        print(f"  Saved: {html_path}")
