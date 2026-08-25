"""Per-species occurrence map for the paper (matplotlib).

Renders the paper's per-species occurrence panel (e.g. *Chryselium
gnaphalioides*, *Aspidistra attenuata*): for one species, every visited 1 km
GBIF cell inside its POWO native range is drawn as a small square, two-coloured
by whether the species was actually recorded there —

  - navy   : visited-but-absent  (in range and sampled, but not this species)
  - orange : present             (this species observed here)

The species' native-range polygon is outlined on top, the map is cropped to
that range over a light-land / blue-ocean base, and a small header reports the
species' Sampling Effort (SE) and Relative Prevalence (RP) both as raw
percentages and as percentiles across all species (e.g. "SE 0.6% · P12").

The compute-side loaders (`load_species_data`, `load_world_map`) are reused
from src.visualizations.maps.grid; this module only adds the matplotlib render
layer. Figures render inline in a notebook and fall back to "Agg" when headless.
"""

import os

import geopandas as gpd
import h5py
import hdf5plugin  # noqa: F401  — registers the Blosc filters the data is written with
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from scipy.stats import rankdata
from shapely.geometry import box as shapely_box

from src.visualizations.maps.grid import DATA_DIR, get_species_id, load_world_map

# ── Palette (matches proxies.ipynb / properties_2dhist.ipynb) ────────────────
_CELL_ABSENT = "#3D405B"   # navy — visited but this species absent
_CELL_PRESENT = "#E07A5F"  # orange — species present
_RANGE_COLOR = "#E07A5F"   # native-range outline (orange, width 2)
_LAND_FILL = "#F9F4F3"     # beige land
_BORDER_COLOR = "#c0bdb6"  # country borders
_OCEAN_FILL = "#d4e4f7"    # blue ocean
_SE_COLOR = "#2D3A4A"      # header SE label (blue)
_RP_COLOR = "#D4603A"      # header RP label (orange)

# Cap on the display grid's per-axis dimension. The visited cells are rasterised
# onto a regular grid at the data's native ~1 km spacing; for wide ranges that
# would be an enormous array, so the step is coarsened to keep the grid at most
# this many cells on a side (cells stay visible, memory stays bounded).
_MAX_GRID_DIM = 1500

# Cap on total cells for the interactive (plotly) raster map — a continental range
# would otherwise hang the browser. plot_species_occurrence_map_plotly() raises above it.
_MAX_INTERACTIVE_CELLS = 3_000_000


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_occurrences(species_name: str, data_dir: str):
    """Load a species' in-range and presence 1 km cells + its native range.

    Reads the 1 km-aggregated files directly rather than going through
    ``grid.load_species_data``; both are now on the same 1 km basis, so navy
    (absent) and orange (present) cells always line up on the same grid.

    Returns
    -------
    in_range_coords : (N, 2) [lon, lat] of visited 1 km cells in range.
    presence_coords : (M, 2) [lon, lat] of cells where the species was observed.
    range_gdf : GeoDataFrame — native-range polygon (EPSG:4326).
    """
    species_id = get_species_id(species_name, data_dir)

    with h5py.File(os.path.join(data_dir, "predictors", "train_location_1km.h5"), "r") as f:
        all_locations = f["predictor_data"][:]  # (N_1km, 2) [lon, lat]

    def _sparse_indices(h5_path):
        with h5py.File(h5_path, "r") as f:
            lookup = f["target_location_indices_lookup"]
            tli = f["target_location_indices"]
            s, e = int(lookup[species_id]), int(lookup[species_id + 1])
            return tli[s:e]

    in_range_idx = _sparse_indices(
        os.path.join(data_dir, "range_masks_1km", "train_species_index.h5")
    )
    presence_idx = _sparse_indices(
        os.path.join(data_dir, "targets", "train_targets_1km.h5")
    )

    range_gdf = gpd.read_file(
        os.path.join(data_dir, "native_ranges", f"{species_name}.shp")
    ).to_crs("EPSG:4326")

    return all_locations[in_range_idx], all_locations[presence_idx], range_gdf


def _fmt_pct(value: float) -> str:
    """Format a proxy percentage: 1 decimal, or 2 when it would read as 0.0%."""
    return f"{value:.2f}%" if abs(value) < 1.0 else f"{value:.1f}%"


def _proxy_percentiles(proxies: pd.DataFrame, species_name: str):
    """Look up a species' SE/RP raw values and cross-species percentiles.

    Percentiles are the average-rank position of the species' value among all
    rows of *proxies* (P1 = lowest, P100 = highest), matching the P-value
    convention used in the properties / grouping figures.

    Returns
    -------
    (se_value, se_pctile, rp_value, rp_pctile)
        Raw percentages and integer percentile ranks (1-100).
    """
    n = len(proxies)
    se_pctiles = rankdata(proxies["sampling_effort"], method="average") / n * 100.0
    rp_pctiles = rankdata(proxies["relative_prevalence"], method="average") / n * 100.0

    match = proxies.index[proxies["species_name"] == species_name]
    if len(match) == 0:
        raise ValueError(
            f"Species '{species_name}' not found in proxies (column 'species_name')."
        )
    i = match[0]
    return (
        float(proxies.at[i, "sampling_effort"]),
        int(round(se_pctiles[proxies.index.get_loc(i)])),
        float(proxies.at[i, "relative_prevalence"]),
        int(round(rp_pctiles[proxies.index.get_loc(i)])),
    )


def _occurrence_grid(in_range_coords: np.ndarray, presence_coords: np.ndarray):
    """Rasterise visited / presence cells onto a regular lon-lat grid.

    Builds a categorical grid over the visited cells: 0 = visited-but-absent,
    1 = presence, NaN = not visited. The step is derived from the data's own
    cell spacing but coarsened if the range is too wide (see _MAX_GRID_DIM) so
    the grid stays a manageable size.

    Returns
    -------
    grid : (H, W) float array with values {0, 1, nan}
    x_edges, y_edges : cell-edge coordinate arrays for pcolormesh
    """
    # Native cell spacing = median gap between unique coordinates (≈ 1 km).
    raw_lons = np.sort(np.unique(in_range_coords[:, 0]))
    raw_lats = np.sort(np.unique(in_range_coords[:, 1]))
    step_lon = np.median(np.diff(raw_lons)) if len(raw_lons) > 1 else 0.008333
    step_lat = np.median(np.diff(raw_lats)) if len(raw_lats) > 1 else 0.008333
    step = max(float(step_lon), float(step_lat), 1e-6)

    # Coarsen the step if the range is too wide to rasterise at native spacing.
    span = max(
        float(np.ptp(in_range_coords[:, 0])),
        float(np.ptp(in_range_coords[:, 1])),
        step,
    )
    step = max(step, span / _MAX_GRID_DIM)

    xmin = in_range_coords[:, 0].min()
    ymin = in_range_coords[:, 1].min()

    def _rc(coords):
        cols = np.round((coords[:, 0] - xmin) / step).astype(int)
        rows = np.round((coords[:, 1] - ymin) / step).astype(int)
        return rows, cols

    ar, ac = _rc(in_range_coords)
    n_row, n_col = ar.max() + 1, ac.max() + 1

    grid = np.full((n_row, n_col), np.nan, dtype=np.float32)
    grid[ar, ac] = 0.0  # visited-but-absent

    if len(presence_coords):
        pr, pc = _rc(presence_coords)
        # Guard against a presence cell landing just outside the visited bbox.
        valid = (pr >= 0) & (pr < n_row) & (pc >= 0) & (pc < n_col)
        grid[pr[valid], pc[valid]] = 1.0  # presence

    x_edges = xmin - step / 2 + np.arange(n_col + 1) * step
    y_edges = ymin - step / 2 + np.arange(n_row + 1) * step
    return grid, x_edges, y_edges


def _crop_bounds(range_gdf: gpd.GeoDataFrame, padding_frac: float):
    """Padded (minx, miny, maxx, maxy) bounding box around the range polygon."""
    b = range_gdf.total_bounds
    pad_lon = (b[2] - b[0]) * padding_frac
    pad_lat = (b[3] - b[1]) * padding_frac
    return (b[0] - pad_lon, b[1] - pad_lat, b[2] + pad_lon, b[3] + pad_lat)


def _draw_layers(ax, grid, x_edges, y_edges, range_gdf, land_gdf, crop_bounds,
                 presence_coords=None, presence_size=7.0):
    """Draw the shared map layers (ocean, land, cells, borders, range) onto ax.

    Reused for both the main axes and an optional zoom inset so the two share
    exactly the same styling. Presence cells are additionally overplotted as
    slightly enlarged markers so they stay legible even when the species occupies
    only a tiny fraction of its surveyed range (``presence_size=0`` disables it).
    """
    minx, miny, maxx, maxy = crop_bounds

    # Ocean base fills the whole axes; land is drawn on top.
    ax.set_facecolor(_OCEAN_FILL)
    if not land_gdf.empty:
        land_gdf.plot(ax=ax, facecolor=_LAND_FILL, edgecolor="none", zorder=1)

    # Two-colour occurrence cells.
    cmap = ListedColormap([_CELL_ABSENT, _CELL_PRESENT])
    norm = BoundaryNorm([-0.5, 0.5, 1.5], cmap.N)
    ax.pcolormesh(
        x_edges, y_edges, np.ma.masked_invalid(grid),
        cmap=cmap, norm=norm, zorder=2, shading="flat",
    )

    # Country borders on top of the cells (thin), then the range outline.
    if not land_gdf.empty:
        land_gdf.boundary.plot(ax=ax, color=_BORDER_COLOR, linewidth=0.5, zorder=3)
    range_gdf.boundary.plot(ax=ax, color=_RANGE_COLOR, linewidth=2.0, zorder=4)

    # Emphasise presence cells: as a fraction of the surveyed range they can be
    # a handful of pixels, so overplot them a touch larger to keep them visible.
    if presence_coords is not None and len(presence_coords) and presence_size:
        ax.scatter(presence_coords[:, 0], presence_coords[:, 1], s=presence_size,
                   marker="s", c=_CELL_PRESENT, edgecolors="none", zorder=5)

    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)
    ax.set_aspect("equal")


def _draw_header(ax, species_name, se_value, se_pct, rp_value, rp_pct):
    """Species name + SE/RP percentile header above the axes."""
    ax.text(0.5, 1.135, species_name, transform=ax.transAxes,
            ha="center", va="bottom", fontsize=15,
            fontstyle="italic", fontweight="bold", color="#222222")
    ax.text(0.5, 1.01,
            f"SE {_fmt_pct(se_value)} · P{se_pct}        "
            f"RP {_fmt_pct(rp_value)} · P{rp_pct}",
            transform=ax.transAxes, ha="center", va="bottom", fontsize=12,
            color="#444444")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def plot_species_occurrence_map(
    species_name: str,
    data_dir: str = None,
    proxies: pd.DataFrame = None,
    ax=None,
    padding_frac: float = 0.05,
    show_header: bool = True,
    inset_bounds=None,
    presence_size: float = 7.0,
    figsize=(5.0, 6.0),
) -> plt.Figure:
    """Two-colour per-species occurrence map with native range and SE/RP header.

    Every visited 1 km cell inside the species' POWO native range is drawn as a
    square, coloured navy where the species was *absent* (visited-but-absent =
    in-range cells minus presence cells) and orange where it was *present*. The
    native-range polygon is outlined on top and the map is cropped to that
    range. A header reports Sampling Effort and Relative Prevalence as both raw
    percentages and cross-species percentiles.

    Parameters
    ----------
    species_name : str
        Exact species name as in species_properties_1km.csv / the range-map shapefiles.
    data_dir : str, optional
        Root data directory. Defaults to the repo's configured DATA_DIR.
    proxies : pandas.DataFrame, optional
        Pre-loaded species_properties_1km.csv (must have ``species_name``,
        ``sampling_effort``, ``relative_prevalence``). Loaded on demand if None.
    ax : matplotlib.axes.Axes, optional
        Axes to draw into (e.g. one panel of a multi-species figure). A new
        figure + axes is created when None.
    padding_frac : float
        Fractional padding added around the range bounding box for the crop.
    show_header : bool
        Draw the species-name + SE/RP percentile header above the axes.
    inset_bounds : tuple, optional
        ``(minx, miny, maxx, maxy)`` for an optional zoom inset (upper-right),
        with a connector box indicating the zoomed region. No inset when None.
    figsize : tuple
        Figure size (inches) when creating a new figure (ignored if ``ax`` given).

    Returns
    -------
    matplotlib.figure.Figure
        The figure containing the map (``ax.figure`` when an ``ax`` was passed).
    """
    if data_dir is None:
        data_dir = DATA_DIR
    if proxies is None:
        proxies = pd.read_csv(os.path.join(data_dir, "targets", "species_properties_1km.csv"))

    # ── Load geometry / occurrences ──────────────────────────────────────────
    in_range_coords, presence_coords, range_gdf = _load_occurrences(species_name, data_dir)

    crop_bounds = _crop_bounds(range_gdf, padding_frac)
    grid, x_edges, y_edges = _occurrence_grid(in_range_coords, presence_coords)

    # Country polygons clipped to the (padded) crop window.
    world_gdf = load_world_map(data_dir)
    land_gdf = gpd.clip(world_gdf, shapely_box(*crop_bounds))
    land_gdf = land_gdf[~land_gdf.geometry.is_empty & land_gdf.geometry.is_valid]

    # ── Figure / axes ────────────────────────────────────────────────────────
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    _draw_layers(ax, grid, x_edges, y_edges, range_gdf, land_gdf, crop_bounds,
                 presence_coords=presence_coords, presence_size=presence_size)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    # ── Optional zoom inset ──────────────────────────────────────────────────
    if inset_bounds is not None:
        axins = ax.inset_axes([0.62, 0.62, 0.36, 0.36])
        _draw_layers(axins, grid, x_edges, y_edges, range_gdf, land_gdf, inset_bounds,
                     presence_coords=presence_coords, presence_size=presence_size)
        axins.set_xticks([])
        axins.set_yticks([])
        for spine in axins.spines.values():
            spine.set_edgecolor("#666666")
            spine.set_linewidth(0.8)
        ax.indicate_inset_zoom(axins, edgecolor="#666666", linewidth=0.8, alpha=0.9)

    # ── Header ───────────────────────────────────────────────────────────────
    if show_header:
        se_value, se_pct, rp_value, rp_pct = _proxy_percentiles(proxies, species_name)
        _draw_header(ax, species_name, se_value, se_pct, rp_value, rp_pct)

    return fig


def plot_species_occurrence_map_plotly(
    species_name: str,
    data_dir: str = None,
    proxies: pd.DataFrame = None,
    show_header: bool = True,
):
    """Interactive, zoomable version of the occurrence map (plotly).

    The same two-colour occurrence map as :func:`plot_species_occurrence_map`,
    but as a pan/zoom plotly figure — useful because the orange presence cells
    are often a tiny fraction of the surveyed range and only resolve up close.
    Cells are drawn as a 1 km raster (``go.Heatmap``) so they keep a true 1 km
    footprint in data coordinates while zooming, rather than fixed-size dots.
    Pair with ``.show(config={"scrollZoom": True})`` for mouse-wheel zoom.
    Renders inline in Jupyter / Colab / nbviewer (needs ``plotly`` installed).

    Returns
    -------
    plotly.graph_objects.Figure
    """
    import plotly.graph_objects as go

    if data_dir is None:
        data_dir = DATA_DIR
    if proxies is None:
        proxies = pd.read_csv(os.path.join(data_dir, "targets", "species_properties_1km.csv"))

    from rasterio.features import rasterize
    from rasterio.transform import from_bounds

    in_range_coords, presence_coords, range_gdf = _load_occurrences(species_name, data_dir)
    minx, miny, maxx, maxy = _crop_bounds(range_gdf, 0.05)

    # Native ~1 km cell size, from the spacing of the visited cells.
    ux = np.unique(in_range_coords[:, 0])
    uy = np.unique(in_range_coords[:, 1])
    step = max(float(np.median(np.diff(ux))) if len(ux) > 1 else 0.008333,
               float(np.median(np.diff(uy))) if len(uy) > 1 else 0.008333, 1e-6)
    nx = int(np.ceil((maxx - minx) / step))
    ny = int(np.ceil((maxy - miny) / step))

    # Blocker: a continental range rasterises to an enormous grid that would hang
    # the browser. Refuse it and point at the (down-sampled) static map instead.
    if nx * ny > _MAX_INTERACTIVE_CELLS:
        raise ValueError(
            f"{species_name!r} spans ~{nx * ny:,} 1 km cells ({nx}x{ny}) — too large for an "
            f"interactive raster map. Use plot_species_occurrence_map() (static) instead.")

    # Burn land + occurrences into ONE raster so there is no trace/shape layering to
    # fight (in-browser a Heatmap sits above the SVG shape layer, so a separate land
    # shape would be hidden behind the ocean): 0 = land (beige), 1 = visited-but-absent
    # (navy), 2 = present (orange), NaN = ocean (shows the blue plot background).
    land_gdf = gpd.clip(load_world_map(data_dir), shapely_box(minx, miny, maxx, maxy))
    geoms = [g for g in land_gdf.geometry if not g.is_empty]
    land = np.zeros((ny, nx), dtype="uint8")
    if geoms:
        land = rasterize(((g, 1) for g in geoms), out_shape=(ny, nx), fill=0, dtype="uint8",
                         transform=from_bounds(minx, miny, maxx, maxy, nx, ny))[::-1]

    z = np.full((ny, nx), np.nan)
    z[land == 1] = 0.0

    def _cells(coords):
        c = np.clip(((coords[:, 0] - minx) / step).astype(int), 0, nx - 1)
        r = np.clip(((coords[:, 1] - miny) / step).astype(int), 0, ny - 1)
        return r, c

    r, c = _cells(in_range_coords)
    z[r, c] = 1.0
    if len(presence_coords):
        r, c = _cells(presence_coords)
        z[r, c] = 2.0

    fig = go.Figure()
    fig.add_trace(go.Heatmap(
        z=z, x0=minx + step / 2, dx=step, y0=miny + step / 2, dy=step,
        zmin=0, zmax=2, zsmooth=False, showscale=False, hoverongaps=False,
        colorscale=[[0.0, _LAND_FILL], [1 / 3, _LAND_FILL], [1 / 3, _CELL_ABSENT],
                    [2 / 3, _CELL_ABSENT], [2 / 3, _CELL_PRESENT], [1.0, _CELL_PRESENT]],
        hovertemplate="lon %{x:.3f}<br>lat %{y:.3f}<extra></extra>"))
    # Legend proxies (a Heatmap carries no legend entry of its own).
    for _name, _color in (("visited · absent", _CELL_ABSENT), ("present", _CELL_PRESENT)):
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(size=9, color=_color, symbol="square"), name=_name))
    for geom in range_gdf.geometry:
        for poly in (geom.geoms if geom.geom_type == "MultiPolygon" else [geom]):
            xs, ys = poly.exterior.xy
            fig.add_trace(go.Scatter(
                x=list(xs), y=list(ys), mode="lines", line=dict(color=_RANGE_COLOR, width=2),
                name="native range", showlegend=False, hoverinfo="skip"))

    title = f"<i>{species_name}</i>"
    if show_header:
        se_v, se_p, rp_v, rp_p = _proxy_percentiles(proxies, species_name)
        title += f"    SE {_fmt_pct(se_v)} · P{se_p}    RP {_fmt_pct(rp_v)} · P{rp_p}"

    fig.update_layout(
        title=title, template="simple_white", plot_bgcolor=_OCEAN_FILL, height=620,
        margin=dict(l=10, r=10, t=54, b=10), dragmode="pan",
        legend=dict(orientation="h", yanchor="bottom", y=-0.06, x=0),
        xaxis=dict(title="", showgrid=False, zeroline=False, range=[minx, maxx]),
        yaxis=dict(title="", showgrid=False, zeroline=False,
                   scaleanchor="x", scaleratio=1, range=[miny, maxy]),
    )
    return fig
