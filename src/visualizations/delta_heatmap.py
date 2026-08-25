"""Multi-panel delta heatmaps for the paper (matplotlib).

Renders the paper's multi-panel delta-heatmap figure. Species are ranked along
two properties (e.g. sampling effort x relative prevalence); per-region
(coarse) or moving-window (fine) metric deltas between two models are computed
and drawn on a diverging colormap, with quadrant midlines and extreme-decile
"L-strip" annotations. Aggregation is metric-aware: mean for AUROC, median for
unbounded metrics such as AUPRG.

This module is used headless from a CLI to save figures, so it forces the
non-interactive "Agg" backend before importing pyplot.
"""

import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Polygon
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.offsetbox import TextArea, HPacker, VPacker, AnchoredOffsetbox
from scipy.stats import rankdata

from src.evaluation.regions import region_masks as _region_masks

_HEATMAP_L = 0.1    # L-strip width in normalised [0,1] coords
_HEATMAP_M = 0.5    # quadrant midpoint
_HEATMAP_VMAX = 5.0  # default ± pp colorbar range


def _sdm_cmap(palette: str = "A"):
    """Diverging colormap for delta-AUROC heatmaps.

    Parameters
    ----------
    palette : str
        "orig" — navy ↔ terracotta        (already used in other figures)
        "A"    — dusty plum ↔ warm amber   colorblind-safe, earthy
        "B"    — slate teal ↔ warm sand    colorblind-safe, cooler
        "C"    — charcoal ↔ muted rose     very desaturated
        "D"    — deep teal ↔ warm copper   vivid, colorblind-safe
        "E"    — blue-grey ↔ warm brick    sophisticated, muted
        "F"    — forest green ↔ gold       nature-inspired, colorblind-safe
        "G"    — indigo ↔ warm peach       soft contrast, colorblind-safe
        -- green-positive: positive Δ = improvement = green --
        "H"    — muted violet ↔ sage green  most colorblind-safe green option
        "I"    — dusty mauve ↔ forest green warmer/softer feel
        "J"    — warm brown ↔ olive green   earthy both ends
    """
    palettes = {
        "orig": [          # navy → terracotta (already used in other figures)
            "#28 2D 46",
            "#82 91 BE",
            "#F9 F4 F4",
            "#F5 A0 78",
            "#B4 46 28",
        ],
        "A": [             # dusty plum ↔ warm amber
            "#3D2B56",
            "#8E7BA8",
            "#F5F2EC",
            "#C8980A",
            "#8B6400",
        ],
        "B": [             # slate teal ↔ warm sand
            "#1E5B6B",
            "#6AABB8",
            "#F5F2EC",
            "#C8A96A",
            "#8B6B20",
        ],
        "C": [             # charcoal ↔ muted rose
            "#3A3A4A",
            "#8A8AA8",
            "#F5F2EC",
            "#C87880",
            "#8B3848",
        ],
        "D": [             # deep teal ↔ warm copper  (more saturated than B)
            "#1A4E6B",
            "#4A9AB8",
            "#F5F0EA",
            "#D49020",
            "#9A6000",
        ],
        "E": [             # blue-grey ↔ warm brick  (muted, sophisticated)
            "#384858",
            "#7898A8",
            "#F5F2EC",
            "#C87850",
            "#8B4828",
        ],
        "F": [             # forest green ↔ warm gold  (nature, colorblind-safe)
            "#2E5A40",
            "#6EA888",
            "#F5F2EC",
            "#C8A030",
            "#8B7010",
        ],
        "G": [             # indigo ↔ warm peach  (soft, colorblind-safe)
            "#3A3A90",
            "#8080C8",
            "#F5F2EC",
            "#E09868",
            "#A85830",
        ],
        # ── green-positive (positive = improvement = green) ──────────────
        "H": [             # muted violet ↔ forest green  (most colorblind-safe)
            "#5C3578",
            "#A87BC0",
            "#F5F2EC",
            "#4E9448",
            "#1E5020",
        ],
        "I": [             # dusty mauve ↔ forest green  (warmer/softer)
            "#7A3858",
            "#C088A0",
            "#F5F2EC",
            "#509060",
            "#285838",
        ],
        "J": [             # warm brown ↔ olive green  (earthy both ends)
            "#6A3820",
            "#B07850",
            "#F5F2EC",
            "#709050",
            "#3A5820",
        ],
    }
    colors = palettes[palette]
    if palette == "orig":
        colors = [c.replace(" ", "") for c in colors]
    return LinearSegmentedColormap.from_list(f"sdm_diverging_{palette}", colors)


def _region_geometry():
    """
    Build polygon geometry for the 8 heatmap regions.

    Quadrant polygons cover the full [0,1]^2 square (L-strips are
    rendered on top at higher z-order).  L-strip polygons use the
    diagonal corner split matching the data masks.
    """
    L, M = _HEATMAP_L, _HEATMAP_M
    return {
        "bottom_left":     {"verts": [(0, 0), (M, 0), (M, M),  (0, M)],  "label_xy": (M / 2,    M / 2)},
        "bottom_right":    {"verts": [(M, 0), (1, 0), (1, M),  (M, M)],  "label_xy": ((M+1)/2,  M / 2)},
        "top_left":        {"verts": [(0, M), (M, M), (M, 1),  (0, 1)],  "label_xy": (M / 2,    (M+1)/2)},
        "top_right":       {"verts": [(M, M), (1, M), (1, 1),  (M, 1)],  "label_xy": ((M+1)/2,  (M+1)/2)},
        "L1_bottom_left":  {"verts": [(0, 0), (M, 0), (M, L),  (L, L)],  "label_xy": (M / 2,    L / 2)},
        "L2_bottom_right": {"verts": [(M, 0), (1, 0), (1, L),  (M, L)],  "label_xy": ((M+1)/2,  L / 2)},
        "L3_left_lower":   {"verts": [(0, 0), (L, L), (L, M),  (0, M)],  "label_xy": (L / 2,    (L+M)/2)},
        "L4_left_upper":   {"verts": [(0, M), (L, M), (L, 1),  (0, 1)],  "label_xy": (L / 2,    (M+1)/2)},
    }


# `_region_masks` now lives in src.evaluation.regions (shared with the stratified
# table so the heatmap compartments and the table rows are always identical).


def _agg_fn(agg: str):
    """Resolve an aggregation name to the matching NaN-aware reducer.

    ``"median"`` is the robust choice for unbounded metrics (e.g. AUPRG,
    which has no negative lower bound) where a handful of catastrophic
    per-species deltas would otherwise dominate a regional mean.
    """
    if agg == "mean":
        return np.nanmean
    if agg in ("median", "paired_median"):
        return np.nanmedian
    raise ValueError(
        f"agg must be 'mean', 'median' or 'paired_median', got '{agg}'")


def _finite_both(m1: np.ndarray, m2: np.ndarray) -> np.ndarray:
    """Species scored finitely by both models (in every seed, if seed-stacked)."""
    f = np.isfinite(m1) & np.isfinite(m2)
    return f.all(axis=0) if f.ndim == 2 else f


def _reduce_delta(m1: np.ndarray, m2: np.ndarray, sel: np.ndarray, agg: str) -> float:
    """Regional delta under the requested aggregation.

    ``"mean"``          : mean(m2) - mean(m1), identical to the mean paired delta.
    ``"median"``        : median(m2) - median(m1) -- a difference of summary
                          statistics, which for a skewed metric such as AUPRG
                          overstates the effect (roughly 2x in practice).
    ``"paired_median"`` : median(m2 - m1), the median of the per-species
                          differences. Correctly ordered *and* robust to the
                          unbounded negative tail; the right choice for AUPRG.
    """
    reduce = _agg_fn(agg)
    if m1.ndim == 2:
        # Seed-stacked (n_seeds, n_species): reduce *within* each seed, then
        # average across seeds. Non-linear statistics such as the median do not
        # commute with seed-averaging, so this order must match the tables.
        per_seed = [
            (float(np.nanmedian(m2[s][sel] - m1[s][sel])) if agg == "paired_median"
             else float(reduce(m2[s][sel]) - reduce(m1[s][sel])))
            for s in range(m1.shape[0])
        ]
        return float(np.mean(per_seed))
    if agg == "paired_median":
        return float(np.nanmedian(m2[sel] - m1[sel]))
    return float(reduce(m2[sel]) - reduce(m1[sel]))


def _region_stats(m1: np.ndarray, m2: np.ndarray, masks: dict,
                  agg: str = "mean") -> dict:
    """Per-region delta (pp) and paired valid-species count.

    Each region reports ``agg(m2) − agg(m1)`` over species where BOTH
    models scored finitely (a paired comparison). *m1*/*m2* are the two
    models' per-species scores already on the display scale (i.e. ×100).

    For ``agg="mean"`` this equals the mean per-species delta (so AUROC
    figures are unchanged). For ``agg="median"`` it is the difference of
    regional medians — the robust choice for unbounded metrics such as
    AUPRG. Note that median(m2) − median(m1) ≠ median(m2 − m1).
    """
    both = _finite_both(m1, m2)
    out = {}
    for name, m in masks.items():
        sel = m & both
        n = int(sel.sum())
        out[name] = {
            "n":     n,
            "delta": _reduce_delta(m1, m2, sel, agg) if n > 0 else 0.0,
        }
    return out


def _draw_split_title(ax, title: str, title_colors=None, fontsize: int = 18,
                      multiline: bool = False, plain_color=None):
    """Render 'model2 vs. model1' title with colour-coded model names.

    If *plain_color* is given, both names (and the separator) are drawn
    in that colour (used for ablation-row panels where the delta
    direction, not the two-model contrast, is the message).
    Otherwise m2 is drawn in *pos_color* and m1 in *neg_color* (from
    *title_colors*), with a neutral 'vs.' separator.
    *multiline* places m2 on line 1 and 'vs. m1' on line 2 (for wide titles).
    Uses HPacker/VPacker so the coloured runs never overlap.
    """
    if title_colors is None or " − " not in title:
        kw = {"fontsize": fontsize, "pad": 5}
        if plain_color is not None:
            kw["color"] = plain_color
            kw["fontweight"] = "bold"
        ax.set_title(title, **kw)
        return
    neg_color, pos_color = title_colors
    if plain_color is not None:
        c_m2 = c_m1 = plain_color
        c_sep = plain_color
    else:
        c_m2, c_m1 = pos_color, neg_color
        c_sep = "#181818"
    m2_name, m1_name = title.split(" − ", 1)

    def _ta(text, color, weight="bold"):
        return TextArea(text, textprops=dict(color=color, fontsize=fontsize,
                                             fontweight=weight))

    if multiline:
        # line 1 (centred): m2
        # line 2 (centred): vs. <m1>
        line1 = HPacker(children=[_ta(m2_name, c_m2)], align="center", pad=0, sep=0)
        line2 = HPacker(children=[_ta("vs. ", c_sep, weight="normal"),
                                  _ta(m1_name, c_m1)],
                        align="center", pad=0, sep=0)
        box = VPacker(children=[line1, line2], align="center", pad=0, sep=4)
    else:
        box = HPacker(children=[_ta(m2_name, c_m2),
                                _ta(" vs. ", c_sep, weight="normal"),
                                _ta(m1_name, c_m1)],
                      align="center", pad=0, sep=0)

    # Anchor just above the axes, centred on the x-axis.
    anchored = AnchoredOffsetbox(
        loc="lower center", child=box, pad=0.0, borderpad=0.0,
        frameon=False, bbox_to_anchor=(0.5, 1.02),
        bbox_transform=ax.transAxes,
    )
    ax.add_artist(anchored)


def _draw_coarse_panel(ax, stats: dict, geometry: dict, cmap, norm,
                       title: str, panel_label: str,
                       title_colors=None, label_fontsize: int = 13,
                       show_L_labels: bool = True,
                       title_multiline: bool = False,
                       plain_title_color=None, show_L: bool = True):
    """Render one coarse-region delta heatmap onto *ax*.

    *show_L* toggles the extreme-decile (10th-percentile) L-strips
    entirely — their coloured polygons, dashed boundaries, and labels.
    When False the panel shows only the four quadrants.
    """
    L, M, VMAX = _HEATMAP_L, _HEATMAP_M, _HEATMAP_VMAX

    for name, info in geometry.items():
        if name.startswith("L"):
            continue
        d = stats[name]["delta"]
        ax.add_patch(Polygon(info["verts"], closed=True,
                             facecolor=cmap(norm(d)), edgecolor="none", zorder=1))

    if show_L:
        for name, info in geometry.items():
            if not name.startswith("L"):
                continue
            d = stats[name]["delta"]
            ax.add_patch(Polygon(info["verts"], closed=True,
                                 facecolor=cmap(norm(d)), edgecolor="none", zorder=3))

    for name, info in geometry.items():
        if name.startswith("L") and (not show_L or not show_L_labels):
            continue
        d = stats[name]["delta"]
        lx, ly = info["label_xy"]
        is_L = name.startswith("L")
        fs = 19 if is_L else label_fontsize
        tc = "white" if abs(d) > VMAX * 0.4 else "black"
        rot = 90 if name in ("L3_left_lower", "L4_left_upper") else 0
        ax.text(lx, ly, f"{d:+.1f}", ha="center", va="center",
                fontsize=fs, fontweight="bold", color=tc, zorder=6, rotation=rot)

    ax.plot([M, M], [0, 1], color="black", lw=0.8, zorder=5, solid_capstyle="butt")
    ax.plot([0, 1], [M, M], color="black", lw=0.8, zorder=5, solid_capstyle="butt")
    if show_L:
        ax.plot([L, 1], [L, L], color="#888888", lw=0.6, ls="--", zorder=5)
        ax.plot([L, L], [L, 1], color="#888888", lw=0.6, ls="--", zorder=5)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    _draw_split_title(ax, title, title_colors,
                      multiline=title_multiline, plain_color=plain_title_color)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    if panel_label:
        # upper-left at title height, centred on the left L-strip (x = L/2)
        ax.text(_HEATMAP_L / 2, 1.02, panel_label, transform=ax.transAxes,
                fontsize=22, fontweight="bold",
                va="bottom", ha="center",
                color="black", zorder=7, clip_on=False)


def _annotate_axis_legend(ax, draw_x: bool = True, draw_y: bool = True,
                          x_label: str = "Sampling effort",
                          y_label: str = "Relative prevalence",
                          fs_axis: int = 18,
                          fs_dim: int = 15):
    """Add axis annotations to the *margin* of a panel.

    Each dimension's two quadrant labels are centred on their own column /
    row (no directional arrows); the bold axis name sits in an outer row
    beyond them. *draw_x* puts the sampling-effort labels below the panel;
    *draw_y* puts the relative-prevalence labels to its left, so the x-axis
    can be repeated under every panel and the y-axis on each row's leftmost
    one. The panel label 'a' is not drawn here — it is attached to the panel
    frame at title height by _draw_coarse_panel / _draw_fine_panel.
    """
    color = "#222"

    if draw_x:
        # x-axis: quadrant labels centred on each column (inner), name below.
        ax.text(0.25, -0.035, "Sparse", ha="center", va="top",
                fontsize=fs_dim, fontstyle="italic",
                transform=ax.transAxes, color=color)
        ax.text(0.75, -0.035, "Dense", ha="center", va="top",
                fontsize=fs_dim, fontstyle="italic",
                transform=ax.transAxes, color=color)
        ax.text(0.5, -0.11, x_label, ha="center", va="top",
                fontsize=fs_axis, fontweight="bold",
                transform=ax.transAxes, color=color)

    if draw_y:
        # y-axis: quadrant labels centred on each row (inner), name to left.
        ax.text(-0.035, 0.25, "Infrequent", ha="right", va="center", rotation=90,
                fontsize=fs_dim, fontstyle="italic",
                transform=ax.transAxes, color=color)
        ax.text(-0.035, 0.75, "Frequent", ha="right", va="center", rotation=90,
                fontsize=fs_dim, fontstyle="italic",
                transform=ax.transAxes, color=color)
        ax.text(-0.11, 0.5, y_label, ha="right", va="center", rotation=90,
                fontsize=fs_axis, fontweight="bold",
                transform=ax.transAxes, color=color)


def _hr_delta_grid(se_rank: np.ndarray, tp_rank: np.ndarray,
                   m1: np.ndarray, m2: np.ndarray, N: int,
                   window_size: int, shift: int, agg: str = "mean"):
    """
    Smoothed delta grid via overlapping rank-space windows.

    Each cell (i, j) reports ``agg(m2) − agg(m1)`` over species within a
    *window_size*-wide rank window centred on (steps[i], steps[j]), using
    only species both models scored finitely. *agg* is ``"mean"`` or
    ``"median"``; *m1*/*m2* are on the display scale (×100).
    Returns (steps, grid) where grid[i, j] is the delta in pp.
    """
    both  = _finite_both(m1, m2)
    steps = np.arange(0, N, shift)
    half  = window_size // 2
    n     = len(steps)
    grid  = np.full((n, n), np.nan)
    for i, si in enumerate(steps):
        for j, sj in enumerate(steps):
            mask = (
                (se_rank > max(0, si - half)) & (se_rank <= min(N, si + half)) &
                (tp_rank > max(0, sj - half)) & (tp_rank <= min(N, sj + half)) &
                both
            )
            if mask.sum() > 0:
                grid[i, j] = _reduce_delta(m1, m2, mask, agg)
    return steps, grid


def _draw_fine_panel(ax, grid: np.ndarray,
                     se_rank: np.ndarray, tp_rank: np.ndarray, N: int,
                     m1: np.ndarray, m2: np.ndarray,
                     cmap, norm, title: str, panel_label: str,
                     title_colors=None, label_fontsize: int = 13,
                     show_L_labels: bool = True,
                     title_multiline: bool = False,
                     plain_title_color=None, agg: str = "mean",
                     show_L: bool = True):
    """Render one fine-grained delta heatmap (imshow) onto *ax*.

    Quadrant colours come from the smoothed rank-window grid; L-strip
    boundaries and labels are overlaid (same regions as the coarse panel)
    so the two modes share the same visual language. *agg* (``"mean"`` |
    ``"median"``) selects the L-strip per-region reducer. *show_L* toggles
    the extreme-decile L-strips (dashed boundaries + labels) entirely.
    """
    L, M, VMAX = _HEATMAP_L, _HEATMAP_M, _HEATMAP_VMAX

    # ── underlying fine-grained colours for the four quadrants ──────────
    ax.imshow(grid.T, origin="lower", extent=[0, 1, 0, 1],
              cmap=cmap, norm=norm, aspect="equal", interpolation="nearest",
              zorder=1)

    # ── compute per-region stats for L-strips (same as coarse) ──────────
    masks  = _region_masks(se_rank, tp_rank, N)
    stats  = _region_stats(m1, m2, masks, agg=agg)
    geometry = _region_geometry()

    # ── grid lines ───────────────────────────────────────────────────────
    ax.plot([M, M], [0, 1], color="black", lw=0.8, zorder=5, solid_capstyle="butt")
    ax.plot([0, 1], [M, M], color="black", lw=0.8, zorder=5, solid_capstyle="butt")
    if show_L:
        ax.plot([L, 1], [L, L], color="#888888", lw=0.6, ls="--", zorder=5)
        ax.plot([L, L], [L, 1], color="#888888", lw=0.6, ls="--", zorder=5)

    # ── labels for all regions ───────────────────────────────────────────
    quad_centers = {
        "bottom_left":  (0.25, 0.25),
        "bottom_right": (0.75, 0.25),
        "top_left":     (0.25, 0.75),
        "top_right":    (0.75, 0.75),
    }
    for name, info in geometry.items():
        if name.startswith("L") and (not show_L or not show_L_labels):
            continue
        d = stats[name]["delta"]
        if name.startswith("L"):
            lx, ly = info["label_xy"]
            fs = 15
        else:
            lx, ly = quad_centers[name]
            fs = label_fontsize
        tc = "white" if abs(d) > VMAX * 0.4 else "black"
        rot = 90 if name in ("L3_left_lower", "L4_left_upper") else 0
        ax.text(lx, ly, f"{d:+.1f}", ha="center", va="center",
                fontsize=fs, fontweight="bold", color=tc, zorder=6, rotation=rot)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    _draw_split_title(ax, title, title_colors,
                      multiline=title_multiline, plain_color=plain_title_color)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    if panel_label:
        # upper-left at title height, centred on the left L-strip (x = L/2)
        ax.text(_HEATMAP_L / 2, 1.02, panel_label, transform=ax.transAxes,
                fontsize=22, fontweight="bold",
                va="bottom", ha="center",
                color="black", zorder=7, clip_on=False)


def plot_combined_delta_heatmaps(
    proxies: pd.DataFrame,
    x_col: str,
    y_col: str,
    pairs: "list[dict]",
    mode: str = "coarse",
    fixed_vmax: float = _HEATMAP_VMAX,
    window_size: int = 2500,
    shift: int = 100,
    palette: str = "H",
    out_path: "str | None" = None,
    legend_mode: str = "panel_a",
    agg: str = "mean",
    metric_label: str = "AUROC",
    show_L: bool = True,
) -> plt.Figure:
    """
    Combined 7-panel delta heatmap figure.

    Layout
    ------
    Row 0 (3 panels — *a is larger than b, c*):
        pairs[0] -> a (largest, doubles as legend host)
        pairs[1] -> b, pairs[2] -> c (top-aligned with a)
    Row 1 (4 narrow panels): pairs[3] -> d ... pairs[6] -> g
    Rightmost column: single diverging colorbar spanning both rows.

    Parameters
    ----------
    proxies : DataFrame containing *x_col* and *y_col*.
    x_col, y_col : str — proxy dimension columns.
    pairs : list of exactly 7 dicts with keys ``m1``, ``m2``, ``label``.
    mode : ``"coarse"`` | ``"fine"``.
    fixed_vmax : float — symmetric ± colorbar range in percentage points.
    window_size, shift : int — smoothing for ``mode="fine"`` only.
    out_path : str, optional — save target (dpi=200).
    legend_mode : ``"panel_a"`` | ``"none"``
        ``"panel_a"`` annotates panel a's margin with axis directions and
        decile strip names so it doubles as the figure key.
    agg : ``"mean"`` | ``"median"``
        How per-species deltas are reduced within each region/window.
        Use ``"median"`` for unbounded metrics (e.g. AUPRG) where a few
        catastrophic species would dominate a regional mean.
    metric_label : str
        Metric name shown on the colorbar (``"Δ <metric_label> (%)"``).
    show_L : bool
        If False, drop the extreme-decile (10th-percentile) L-strips from
        every panel, leaving only the four quadrants.
    """
    if len(pairs) != 7:
        raise ValueError("Exactly 7 model pairs required (3 top + 4 bottom).")

    x       = proxies[x_col].values
    y       = proxies[y_col].values
    N       = len(x)
    se_rank = rankdata(x, method="ordinal").astype(int)
    tp_rank = rankdata(y, method="ordinal").astype(int)

    cmap   = _sdm_cmap(palette)
    norm   = mcolors.TwoSlopeNorm(vmin=-fixed_vmax, vcenter=0, vmax=fixed_vmax)
    labels = ["a", "b", "c", "d", "e", "f", "g"]

    # ── inch-based layout: all 3 top panels equal size, bottom row 4-wide ──
    _s      = 3.5
    _gap_c  = 0.70
    _gap_r  = 1.30  # a touch more row gap for the 'Δ vs. previous step →' hint
    _cbar_g = 0.35
    _cbar_w = 0.40
    _ml, _mr, _mt, _mb = 1.00, 0.15, 0.95, 1.15  # extra margins for two-tier axis labels under every panel

    # Keep total top width = bottom width: 3*_s_top + 2*_gap_c = 4*_s + 3*_gap_c
    _s_top = (4 * _s + _gap_c) / 3

    _figw = _ml + 4*_s + 3*_gap_c + _cbar_g + _cbar_w + _mr
    _figh = _mt + _s_top + _gap_r + _s + _mb

    def _rect(x, y, w, h):            # inches → figure-fraction [l, b, w, h]
        return [x/_figw, y/_figh, w/_figw, h/_figh]

    # Top row x-positions (all same-size)
    _xA = _ml
    _xB = _ml + _s_top + _gap_c
    _xC = _ml + 2*_s_top + 2*_gap_c
    # Bottom row x-positions
    _xD = _ml
    _xE = _ml + _s     + _gap_c
    _xF = _ml + 2*_s   + 2*_gap_c
    _xG = _ml + 3*_s   + 3*_gap_c

    _ybot  = _mb
    _ytop  = _mb + _s + _gap_r               # bottom of top-row panels

    fig = plt.figure(figsize=(_figw, _figh))
    axs = [
        fig.add_axes(_rect(_xA, _ytop, _s_top, _s_top)),   # a
        fig.add_axes(_rect(_xB, _ytop, _s_top, _s_top)),   # b
        fig.add_axes(_rect(_xC, _ytop, _s_top, _s_top)),   # c
        fig.add_axes(_rect(_xD, _ybot, _s, _s)),           # d
        fig.add_axes(_rect(_xE, _ybot, _s, _s)),           # e
        fig.add_axes(_rect(_xF, _ybot, _s, _s)),           # f
        fig.add_axes(_rect(_xG, _ybot, _s, _s)),           # g
    ]
    cbar_ax = fig.add_axes(_rect(_xG + _s + _cbar_g, _ybot,
                                 _cbar_w, _s_top + _gap_r + _s))

    # Per-panel font sizes (top row slightly larger than bottom row)
    _label_fs = [26, 26, 26, 22, 22, 22, 22]
    # b and c get two-line titles (otherwise they'd overflow)
    _title_multiline = [False, True, True, False, False, False, False]
    # Bottom-row titles drawn in the 'positive delta' colour (green),
    # since the ablation chain adds beneficial steps.
    _green = cmap(1.0)
    _plain_colors = [None, None, None, _green, _green, _green, _green]
    if mode == "coarse":
        title_colors = (cmap(0.0), cmap(1.0))
        geometry = _region_geometry()
        masks    = _region_masks(se_rank, tp_rank, N)
        for idx, (ax, pair, pl) in enumerate(zip(axs, pairs, labels)):
            m1, m2 = pair["m1"] * 100, pair["m2"] * 100
            stats = _region_stats(m1, m2, masks, agg=agg)
            _draw_coarse_panel(ax, stats, geometry, cmap, norm, pair["label"], pl,
                               title_colors=title_colors,
                               label_fontsize=_label_fs[idx],
                               show_L_labels=(idx < 3),
                               title_multiline=_title_multiline[idx],
                               plain_title_color=_plain_colors[idx], show_L=show_L)
    elif mode == "fine":
        title_colors = (cmap(0.0), cmap(1.0))
        for idx, (ax, pair, pl) in enumerate(zip(axs, pairs, labels)):
            m1, m2 = pair["m1"] * 100, pair["m2"] * 100
            _, grid = _hr_delta_grid(se_rank, tp_rank, m1, m2, N, window_size, shift, agg=agg)
            _draw_fine_panel(ax, grid, se_rank, tp_rank, N, m1, m2, cmap, norm, pair["label"], pl,
                             title_colors=title_colors,
                             label_fontsize=_label_fs[idx],
                             show_L_labels=(idx < 3),
                             title_multiline=_title_multiline[idx],
                             plain_title_color=_plain_colors[idx], agg=agg, show_L=show_L)
    else:
        raise ValueError(f"mode must be 'coarse' or 'fine', got '{mode}'")

    # ── axis legend: x-labels under every panel, y-labels on each row's
    #    leftmost panel (a = top row, d = bottom row) ──────────────────────
    if legend_mode == "panel_a":
        for idx, ax in enumerate(axs):
            _annotate_axis_legend(ax, draw_x=True, draw_y=(idx in (0, 3)))
    elif legend_mode != "none":
        raise ValueError(f"legend_mode must be 'panel_a' or 'none'; got '{legend_mode}'")

    # ── 'Δ vs. previous step →' hint placed below the bottom row ───────
    # Sits in the bottom margin, below the per-panel x-axis labels.
    _hint_y_in = 0.22                      # below panel d/e/f/g x-labels
    _hint_x_in = _xD                       # start at panel d's left edge
    _hint_w_in = 4*_s + 3*_gap_c           # full bottom-row width
    fig.text(
        (_hint_x_in + _hint_w_in / 2) / _figw,
        _hint_y_in / _figh,
        "Δ vs. previous step  →",
        ha="center", va="center",
        fontsize=18, fontstyle="italic", color=_green, fontweight="bold",
    )

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label(f"Δ {metric_label} (%)", fontsize=21)
    cbar.ax.tick_params(labelsize=19)

    if out_path:
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
    #plt.show()
    return fig


def plot_delta_heatmaps(
    proxies: pd.DataFrame,
    x_col: str,
    y_col: str,
    pairs: "list[dict]",
    mode: str = "coarse",
    fixed_vmax: float = _HEATMAP_VMAX,
    window_size: int = 2500,
    shift: int = 100,
    palette: str = "H",
    out_path: "str | None" = None,
    legend_mode: str = "panel_a",
    agg: str = "mean",
    metric_label: str = "AUROC",
    show_L: bool = True,
) -> plt.Figure:
    """Adaptive delta-heatmap figure for any number of model comparisons.

    Reuses the same per-panel rendering as :func:`plot_combined_delta_heatmaps`
    but lays out ``len(pairs)`` panels adaptively:

    - ``N == 1`` -> a single panel (the user-toolkit case);
    - ``N <= 3`` -> one row of equal panels;
    - ``N >= 4`` -> two tiers, ``floor(N/2)`` panels on top and ``ceil(N/2)``
      below. The row with fewer panels is widened so both rows span the same
      width, making the top ("main") panels larger. At ``N == 7`` this yields
      the same 3-big-over-4-small geometry as the paper figure.

    ``N > 8`` still renders but warns (panels get cramped; capped at two rows).
    Unlike the paper figure, no ablation-specific decorations (green titles,
    the "Δ vs. previous step" hint) are drawn.

    Args:
        proxies: DataFrame with ``x_col`` and ``y_col``.
        pairs: list of dicts, each ``{"m1", "m2", "label"}`` where ``m1``/``m2``
            are per-species scores (0-1 scale) aligned to ``proxies`` rows.
        mode: ``"coarse"`` (per-region) or ``"fine"`` (moving-window).
        agg: ``"mean"`` (AUROC) or ``"median"`` (unbounded metrics, e.g. AUPRG).
        metric_label: metric name shown on the colorbar.

    Returns the matplotlib figure.
    """
    n = len(pairs)
    if n == 0:
        raise ValueError("plot_delta_heatmaps needs at least one model pair.")
    if n > 8:
        warnings.warn(
            f"{n} comparisons is a lot; the figure is capped at two rows and panels "
            "become small — consider <= 8 pairs (or split into several figures)."
        )
    if mode not in ("coarse", "fine"):
        raise ValueError(f"mode must be 'coarse' or 'fine', got '{mode}'")

    # Row split: one row up to 3, else two tiers (fewer on top => bigger).
    rows = [n] if n <= 3 else [n // 2, n - n // 2]

    x = proxies[x_col].values
    y = proxies[y_col].values
    N = len(x)
    se_rank = rankdata(x, method="ordinal").astype(int)
    tp_rank = rankdata(y, method="ordinal").astype(int)
    cmap = _sdm_cmap(palette)
    norm = mcolors.TwoSlopeNorm(vmin=-fixed_vmax, vcenter=0, vmax=fixed_vmax)

    # Inch-based layout (base sizes match plot_combined_delta_heatmaps).
    _s, _gap_c, _gap_r = 3.5, 0.70, 1.30
    _cbar_g, _cbar_w = 0.35, 0.22
    _ml, _mr, _mt, _mb = 1.00, 0.15, 0.95, 1.15

    max_cols = max(rows)
    W = max_cols * _s + (max_cols - 1) * _gap_c                 # total content width
    row_sizes = [(W - (k - 1) * _gap_c) / k for k in rows]      # square panel side per row

    _figw = _ml + W + _cbar_g + _cbar_w + _mr
    content_h = sum(row_sizes) + (len(rows) - 1) * _gap_r
    _figh = _mt + content_h + _mb

    def _rect(x0, y0, w, h):
        return [x0 / _figw, y0 / _figh, w / _figw, h / _figh]

    # Bottom-y of each row, top row first.
    y_bottoms, acc = [], _mb
    for rs in reversed(row_sizes):
        y_bottoms.append(acc)
        acc += rs + _gap_r
    y_bottoms = list(reversed(y_bottoms))

    fig = plt.figure(figsize=(_figw, _figh))
    axs, panel_rows = [], []
    for r, (k, rs, yb) in enumerate(zip(rows, row_sizes, y_bottoms)):
        for c in range(k):
            axs.append(fig.add_axes(_rect(_ml + c * (rs + _gap_c), yb, rs, rs)))
            panel_rows.append(r)

    cbar_ax = fig.add_axes(_rect(_ml + W + _cbar_g, _mb, _cbar_w, content_h))

    # Panel letters (a, b, c, ...); a single panel needs no letter (it would
    # only collide with the title).
    labels = [""] if n == 1 else [chr(ord("a") + i) for i in range(n)]
    title_colors = (cmap(0.0), cmap(1.0))
    if mode == "coarse":
        geometry = _region_geometry()
        masks = _region_masks(se_rank, tp_rank, N)

    leftmost = {panel_rows.index(r) for r in dict.fromkeys(panel_rows)}
    for idx, (ax, pair, pl) in enumerate(zip(axs, pairs, labels)):
        m1, m2 = pair["m1"] * 100, pair["m2"] * 100
        rs = row_sizes[panel_rows[idx]]
        fs = int(round(22 * rs / _s))                    # scale delta-label font with panel size
        show_L_labels = panel_rows[idx] == 0             # L-strip labels on the main (top) row only
        if mode == "coarse":
            stats = _region_stats(m1, m2, masks, agg=agg)
            _draw_coarse_panel(ax, stats, geometry, cmap, norm, pair["label"], pl,
                               title_colors=title_colors, label_fontsize=fs,
                               show_L_labels=show_L_labels, title_multiline=False,
                               plain_title_color=None, show_L=show_L)
        else:
            _, grid = _hr_delta_grid(se_rank, tp_rank, m1, m2, N, window_size, shift, agg=agg)
            _draw_fine_panel(ax, grid, se_rank, tp_rank, N, m1, m2, cmap, norm, pair["label"], pl,
                             title_colors=title_colors, label_fontsize=fs,
                             show_L_labels=show_L_labels, title_multiline=False,
                             plain_title_color=None, agg=agg, show_L=show_L)

    if legend_mode == "panel_a":
        for idx, ax in enumerate(axs):
            _annotate_axis_legend(ax, draw_x=True, draw_y=(idx in leftmost))
    elif legend_mode != "none":
        raise ValueError(f"legend_mode must be 'panel_a' or 'none'; got '{legend_mode}'")

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label(f"Δ {metric_label} (%)", fontsize=21)
    cbar.ax.tick_params(labelsize=19)

    if out_path:
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
    return fig
