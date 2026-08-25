"""The paper's species "property-space" figure (matplotlib).

Renders the three-panel figure describing where the target species sit in the
2-D property space spanned by **sampling effort** (SE, x) and **relative
prevalence** (RP, y):

- panel **(a)** — a 2-D histogram on raw log-log axes with marginal
  histograms (SE on top in navy, RP on the right in terracotta) and
  median (solid) + 10th-percentile (dashed) reference lines;
- panel **(b)** — the same joint histogram on **percentile axes** labelled
  ``P0, P10, P50, P100`` (each species mapped to its rank fraction);
- panel **(c)** — the property space split into the eight rank-based
  **regions** (four median-split quadrants + four extreme-decile L-strips),
  each polygon filled by its species count.

Panels (a) and (b) share one count colorbar; panel (c) has its own. The eight
regions and their species masks come from :mod:`src.evaluation.regions`, so the
counts here always agree with the stratified score table and the delta heatmap.

Figures render inline in a notebook; matplotlib falls back to the "Agg"
backend automatically when this runs headless.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon
from matplotlib.colors import LinearSegmentedColormap, PowerNorm
from scipy.stats import rankdata

from src.evaluation.regions import rank_axes, region_masks

# ── Shared styling ─────────────────────────────────────────────────────
BG = "#F9F4F3"        # warm beige figure background
SE_COLOR = "#2D3A4A"  # navy — sampling effort
RP_COLOR = "#D4603A"  # terracotta — relative prevalence

# Sequential count palettes (light → dark); pick one via ``palette=``.
COUNT_PALETTES = {
    "olive_stone": ["#F5F2E9", "#D5D0B9", "#9DA37A", "#4F5A3A"],
    "grays":       ["#F9F9F9", "#D9D9D9", "#A3A3A3", "#4D4D4D"],
    "beiges":      ["#F9F4F3", "#E5DCD7", "#CFC1B8", "#A38C7D"],
}

# Font sizes (label / tick / panel-letter / colorbar / legend / region-count).
LABEL_FS, TICK_FS, LETTER_FS, CBAR_FS, LEG_FS = 22, 17, 24, 13, 14

# Normalised drawing coordinates for panel (c): median split and decile strip.
_M, _L = 0.5, 0.1


def make_count_cmap(name: str = "beiges", lo: float = 0.10, hi: float = 0.78):
    """Sequential count colormap, trimmed at both ends and BG for zero counts.

    The raw palette is resampled to the ``[lo, hi]`` slice so the lightest
    populated bin stays legible and the darkest never goes pure black; empty
    bins (below ``vmin``) fall through to the beige background via ``set_under``.
    """
    base = LinearSegmentedColormap.from_list("count_base", COUNT_PALETTES[name])
    cmap = LinearSegmentedColormap.from_list("count", base(np.linspace(lo, hi, 256)))
    cmap.set_under(BG)
    return cmap


# ── Panel (c) region geometry ──────────────────────────────────────────
# Polygons cover [0,1]^2 in normalised axis coords. Keys match the masks in
# src.evaluation.regions (L-strips are drawn on top of the quadrants). The
# extreme (L, L) corner is split diagonally between the two L-strips.
_REGION_GEOMETRY = {
    "bottom_left":     [(0, 0), (_M, 0), (_M, _M), (0, _M)],
    "bottom_right":    [(_M, 0), (1, 0), (1, _M), (_M, _M)],
    "top_left":        [(0, _M), (_M, _M), (_M, 1), (0, 1)],
    "top_right":       [(_M, _M), (1, _M), (1, 1), (_M, 1)],
    "L1_bottom_left":  [(0, 0), (_M, 0), (_M, _L), (_L, _L)],
    "L2_bottom_right": [(_M, 0), (1, 0), (1, _L), (_M, _L)],
    "L3_left_lower":   [(0, 0), (_L, _L), (_L, _M), (0, _M)],
    "L4_left_upper":   [(0, _M), (_L, _M), (_L, 1), (0, 1)],
}


def _make_panel(fig, i, left0, stride, bottom, jw, jh, marg_w, marg_h, pad):
    """Add one joint axes plus its top/right marginal axes, returning them.

    Panels are laid out in figure fractions so the joint axes come out as an
    exact square in inches; ``i`` is the 0-based column index.
    """
    left = left0 + i * stride
    axj = fig.add_axes([left, bottom, jw, jh]); axj.set_facecolor(BG)
    axt = fig.add_axes([left, bottom + jh + pad, jw, marg_h], sharex=axj); axt.set_facecolor(BG)
    axr = fig.add_axes([left + jw + pad, bottom, marg_w, jh], sharey=axj); axr.set_facecolor(BG)
    return axj, axt, axr, left


def _joint_panel(fig, axes, left, X, Y, Xb, Yb, H, norm, cmap, refs,
                 logscale, ticks, ticklabels, letter, bottom):
    """Draw a joint 2-D histogram with SE/RP marginals and reference lines."""
    axj, axt, axr = axes
    x_q10, x_q50, y_q10, y_q50 = refs
    axj.pcolormesh(Xb, Yb, H.T, cmap=cmap, norm=norm)
    axt.hist(X, bins=Xb, color=SE_COLOR, edgecolor="white", linewidth=0.3)
    axr.hist(Y, bins=Yb, color=RP_COLOR, edgecolor="white", linewidth=0.3,
             orientation="horizontal")
    if logscale:
        axj.set_xscale("log"); axj.set_yscale("log")
        axj.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, p: f"{v:g}"))
        axj.yaxis.set_major_formatter(mticker.LogFormatterMathtext())  # powers of ten
    # median (solid black) + 10th-percentile (dashed grey) reference lines
    axj.axvline(x_q50 if logscale else _M, color="black", lw=1.2)
    axj.axhline(y_q50 if logscale else _M, color="black", lw=1.2)
    axj.axvline(x_q10 if logscale else _L, color="gray", lw=1.2, ls="--")
    axj.axhline(y_q10 if logscale else _L, color="gray", lw=1.2, ls="--")
    axj.set_xlim(Xb[0], Xb[-1]); axj.set_ylim(Yb[0], Yb[-1])
    if ticks is not None:
        axj.set_xticks(ticks); axj.set_xticklabels(ticklabels)
        axj.set_yticks(ticks); axj.set_yticklabels(ticklabels)
    axj.tick_params(labelsize=TICK_FS)
    axj.set_xlabel("SE", fontsize=LABEL_FS, fontweight="bold", color=SE_COLOR, labelpad=2)
    axj.set_ylabel("RP", fontsize=LABEL_FS, fontweight="bold", color=RP_COLOR, labelpad=2)
    for a in (axt, axr):
        a.tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)
        for s in a.spines.values():
            s.set_visible(False)
    fig.text(left - 0.042, bottom - 0.055, f"{letter})", fontsize=LETTER_FS,
             ha="left", va="top")


def plot_property_space(
    proxies,
    x_col: str = "sampling_effort",
    y_col: str = "relative_prevalence",
    palette: str = "beiges",
    bins: int = 40,
    decile: int = 10,
    out_path: "str | None" = None,
) -> plt.Figure:
    """Render the three-panel species property-space figure.

    Args:
        proxies: a DataFrame with ``x_col``/``y_col``, **or** a path to the
            proxies CSV (e.g. ``"<data_dir>/targets/species_properties_1km.csv"``).
        x_col: sampling-effort column (x axis / SE marginal).
        y_col: relative-prevalence column (y axis / RP marginal).
        palette: key into :data:`COUNT_PALETTES` for the count colormap.
        bins: number of bins per axis for panels (a) and (b).
        decile: extreme-strip fraction for panel (c) regions (``1/decile``);
            passed through to :func:`src.evaluation.regions.region_masks`.
        out_path: if given, save the figure there (SVG/PNG inferred from the
            suffix; PNG at dpi=200).

    Returns:
        The matplotlib :class:`~matplotlib.figure.Figure`.
    """
    if isinstance(proxies, pd.DataFrame):
        props = proxies
    else:
        props = pd.read_csv(proxies)

    x = props[x_col].values
    y = props[y_col].values
    N = len(props)

    cmap = make_count_cmap(palette)

    # ── Rank / percentile transforms ──────────────────────────────────
    # method="average" for the smooth percentile scatter of panels (a, b);
    # rank_axes (ordinal) for the region masks so counts match the score table.
    x_pct = rankdata(x, method="average") / N
    y_pct = rankdata(y, method="average") / N

    # ── Binned 2-D histograms (shared count scale) ────────────────────
    xb = np.logspace(np.log10(x.min()), np.log10(x.max()), bins + 1)
    yb = np.logspace(np.log10(y.min()), np.log10(y.max()), bins + 1)
    pb = np.linspace(0, 1, bins + 1)
    H1 = np.histogram2d(x, y, bins=[xb, yb])[0]
    H2 = np.histogram2d(x_pct, y_pct, bins=[pb, pb])[0]
    vmax = max(H1.max(), H2.max())
    norm = PowerNorm(0.5, vmin=0.5, vmax=vmax)  # sqrt stretch; 0-count -> under
    x_q10, x_q50 = np.quantile(x, [0.1, 0.5])
    y_q10, y_q50 = np.quantile(y, [0.1, 0.5])
    refs = (x_q10, x_q50, y_q10, y_q50)

    # ── Manual layout (figure fractions) -> exact-square joints ───────
    W, H = 17, 5.9
    asp = H / W
    jh = 0.60                    # joint height (fig fraction)
    jw = jh * asp                # joint width  -> square in inches
    marg_h = 0.12                # top marginal thickness
    marg_w = marg_h * asp        # right marginal thickness (matched in inches)
    bottom, left0, gap, pad = 0.16, 0.058, 0.085, 0.008
    stride = jw + marg_w + gap

    fig = plt.figure(figsize=(W, H))
    fig.patch.set_facecolor(BG)

    # ── Panel (a): raw log-log; Panel (b): percentile axes ────────────
    a_axes = _make_panel(fig, 0, left0, stride, bottom, jw, jh, marg_w, marg_h, pad)
    _joint_panel(fig, a_axes[:3], a_axes[3], x, y, xb, yb, H1, norm, cmap, refs,
                 logscale=True, ticks=None, ticklabels=None, letter="a", bottom=bottom)

    pt = [0, 0.1, 0.5, 1.0]
    pl = ["$P_0$", "$P_{10}$", "$P_{50}$", "$P_{100}$"]
    b_axes = _make_panel(fig, 1, left0, stride, bottom, jw, jh, marg_w, marg_h, pad)
    _joint_panel(fig, b_axes[:3], b_axes[3], x_pct, y_pct, pb, pb, H2, norm, cmap, refs,
                 logscale=False, ticks=pt, ticklabels=pl, letter="b", bottom=bottom)

    # ── Panel (c): rank-based regions filled by species count ─────────
    se_rank, tp_rank = rank_axes(x, y)
    masks = region_masks(se_rank, tp_rank, N, decile=decile)
    counts = {k: int(m.sum()) for k, m in masks.items()}
    bnorm = plt.Normalize(0, max(counts.values()))

    axc, axct, axcr, lc = _make_panel(fig, 2, left0, stride, bottom, jw, jh,
                                      marg_w, marg_h, pad)
    axct.axis("off"); axcr.axis("off"); axc.set_frame_on(False)
    for name, verts in _REGION_GEOMETRY.items():
        z = 3 if name.startswith("L") else 1   # L-strips over quadrants
        axc.add_patch(Polygon(verts, closed=True,
                              facecolor=cmap(bnorm(counts[name])),
                              edgecolor="none", zorder=z))
    # median (solid) + decile (dashed, clipped to the (L, L) corner) lines
    axc.axvline(_M, color="black", lw=1.2, zorder=5)
    axc.axhline(_M, color="black", lw=1.2, zorder=5)
    axc.plot([_L, 1], [_L, _L], color="gray", lw=1.2, ls="--", zorder=5)
    axc.plot([_L, _L], [_L, 1], color="gray", lw=1.2, ls="--", zorder=5)
    axc.set_xlim(0, 1); axc.set_ylim(0, 1)
    axc.set_xticks(pt); axc.set_xticklabels(pl, fontsize=TICK_FS)
    axc.set_yticks(pt); axc.set_yticklabels(pl, fontsize=TICK_FS)
    axc.set_xlabel("SE", fontsize=LABEL_FS, fontweight="bold", color=SE_COLOR, labelpad=2)
    axc.set_ylabel("RP", fontsize=LABEL_FS, fontweight="bold", color=RP_COLOR, labelpad=2)
    fig.text(lc - 0.042, bottom - 0.055, "c)", fontsize=LETTER_FS, ha="left", va="top")

    # ── Accessory column: legend + two thin colorbars (left-aligned) ──
    acc_left = left0 + 2 * stride + jw + 0.04
    la = fig.add_axes([acc_left, 0.72, 0.085, 0.14]); la.axis("off")
    la.legend(
        handles=[Line2D([0], [0], color="black", lw=1.4, label="Median"),
                 Line2D([0], [0], color="gray", lw=1.4, ls="--", label="10th percentile")],
        loc="center left", borderaxespad=0, fontsize=LEG_FS, frameon=True, framealpha=1,
    )
    for ybot, the_norm, title in [(0.42, norm, "species / bin\n(panels a, b)"),
                                  (0.10, bnorm, "species / region\n(panel c)")]:
        cax = fig.add_axes([acc_left, ybot, 0.011, 0.22])
        cb = fig.colorbar(plt.cm.ScalarMappable(norm=the_norm, cmap=cmap), cax=cax)
        cax.set_title(title, fontsize=CBAR_FS, pad=6, loc="left")
        cb.ax.tick_params(labelsize=CBAR_FS - 1)
        cb.locator = mticker.MaxNLocator(4)
        cb.update_ticks()

    if out_path:
        dpi = 200 if str(out_path).lower().endswith(".png") else None
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=0.12, facecolor=BG)
    return fig


def plot_species_in_quadrant(
    proxies,
    species_name: str,
    x_col: str = "sampling_effort",
    y_col: str = "relative_prevalence",
    ax=None,
    decile: int = 10,
):
    """Compact percentile field locating one species in the property space.

    All species form a faint background cloud; ``species_name`` is highlighted at
    its (sampling-effort, relative-prevalence) percentile. Median (solid) and
    decile (dashed) lines mark the region boundaries SAGE stratifies on — so the
    marker reads off which of the eight regions the species falls in.

    Parameters
    ----------
    proxies : pandas.DataFrame or str
        Proxies table, or a path to the proxies CSV.
    species_name : str
        Species to highlight; must appear in ``proxies['species_name']``.
    ax : matplotlib.axes.Axes, optional
        Axis to draw into; a new square figure is created if omitted.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if isinstance(proxies, str):
        proxies = pd.read_csv(proxies)
    names = proxies["species_name"].to_numpy()
    hits = np.where(names == species_name)[0]
    if len(hits) == 0:
        raise ValueError(f"{species_name!r} not found in proxies")
    pos = int(hits[0])

    N = len(proxies)
    xp = rankdata(proxies[x_col].to_numpy(), method="average") / N * 100
    yp = rankdata(proxies[y_col].to_numpy(), method="average") / N * 100

    if ax is None:
        _, ax = plt.subplots(figsize=(3.2, 3.2))
    ax.set_facecolor(BG)
    ax.scatter(xp, yp, s=4, color="0.72", alpha=0.45, linewidths=0)
    ax.axvline(50, color="0.3", lw=1.1)
    ax.axhline(50, color="0.3", lw=1.1)
    for v in (decile, 100 - decile):
        ax.axvline(v, color="0.5", lw=0.8, ls="--")
        ax.axhline(v, color="0.5", lw=0.8, ls="--")
    ax.scatter([xp[pos]], [yp[pos]], s=190, color=RP_COLOR,
               edgecolor="white", linewidth=1.8, zorder=6)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_aspect("equal", adjustable="box")  # the property space is a square
    ax.set_xticks([0, 50, 100])
    ax.set_yticks([0, 50, 100])
    ax.set_xlabel("SE percentile", color=SE_COLOR, fontweight="bold", fontsize=10)
    ax.set_ylabel("RP percentile", color=RP_COLOR, fontweight="bold", fontsize=10)
    ax.set_title(f"SE · P{int(round(xp[pos]))}    RP · P{int(round(yp[pos]))}", fontsize=10)
    return ax.figure
