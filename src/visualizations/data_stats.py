"""Per-species occurrence-distribution histograms for the paper (matplotlib).

Renders the paper's "Occurrences per species" figure: a grid of histograms,
one panel per data-preparation level (Full, Aggregated 1 km, and one or more
Subsampled variants). Each panel is a log-x / log-y histogram of the number of
occurrence records **per species** at that level, titled with the level name
and its total occurrence count.

Counts are read from the CSR sparse targets matrices produced by
``scripts/data_prep_scripts/targets.py``. Per-species counts are the non-zero
column sums of the (optionally row-subsampled) matrix; the total occurrence
count is the matrix's non-zero count.

Figures render inline in a notebook and fall back to "Agg" when headless.
"""

import hdf5plugin  # noqa: F401  — registers the Blosc filters the data is written with
import os

import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

_BG_COLOR = "#F9F4F3"   # warm paper background (shared with the observation maps)
_BAR_COLOR = "#e8746a"  # terracotta bars (shared with the paper colour ramp)

# Canonical data-preparation levels. Each entry resolves to a display name and,
# lazily, a per-species occurrence-count array. Subsampled levels take the 1 km
# aggregated matrix restricted to the row indices of a precomputed subsample.
_TARGETS_SUBDIR = "targets"
_SUBSAMPLE_SUBDIR = "targets/subsample_indices"
_FULL_H5 = "train_targets.h5"
_AGG_H5 = "train_targets_1km.h5"
_SUBSAMPLE_SIZES = (10, 100, 1000, 10000, 100000)

# Default figure: Full, Aggregated 1 km, Subsampled N=1000.
_DEFAULT_LEVELS = ("full", "aggregated", "subsampled_1000")


# Which archive on the Zenodo record provides each targets file, so a missing
# file can say what to download instead of raising a bare h5py errno 2.
_ARCHIVE_FOR = {
    _FULL_H5: "sage_po_records_nonaggregated.tar (the non-aggregated presence-only records)",
    _AGG_H5: "sage_benchmark_data.tar (the core 1 km benchmark)",
}


def _load_targets(targets_path: str) -> sp.csr_matrix:
    """Load a CSR (cells x species) targets matrix from an HDF5 file.

    Mirrors ``targets.load_targets_from_h5`` but is inlined here so this
    visualisation module has no dependency on the data-prep scripts.
    """
    import h5py

    if not os.path.exists(targets_path):
        archive = _ARCHIVE_FOR.get(os.path.basename(targets_path))
        hint = f"\nIt ships in {archive}; extract that into your data_dir." if archive else ""
        raise FileNotFoundError(f"Targets file not found: {targets_path}{hint}")

    with h5py.File(targets_path, "r") as f:
        g = f["csr"]
        return sp.csr_matrix(
            (g["data"][:], g["indices"][:], g["indptr"][:]),
            shape=tuple(g.attrs["shape"]),
        )


def _species_counts(mat: sp.csr_matrix) -> np.ndarray:
    """Per-species occurrence count = number of non-zero cells per column."""
    return np.asarray((mat > 0).sum(axis=0)).ravel()


def _resolve_level(level: str, data_dir: str, _agg_cache: dict) -> "tuple[str, np.ndarray]":
    """Resolve a level id to ``(display_name, per_species_counts)``.

    Recognised level ids:
        ``"full"``            — non-aggregated occurrences.
        ``"aggregated"``      — occurrences aggregated to the 1 km grid.
        ``"subsampled_<N>"``  — the 1 km matrix restricted to the first <N>
                                subsample indices (N in {10,100,1000,10000,100000}).

    *_agg_cache* holds the loaded 1 km matrix so it is read at most once when
    several subsampled levels are requested.
    """
    if level == "full":
        mat = _load_targets(os.path.join(data_dir, _TARGETS_SUBDIR, _FULL_H5))
        return "Full", _species_counts(mat)

    if level == "aggregated":
        if "agg" not in _agg_cache:
            _agg_cache["agg"] = _load_targets(os.path.join(data_dir, _TARGETS_SUBDIR, _AGG_H5))
        return "Aggregated (1 km)", _species_counts(_agg_cache["agg"])

    if level.startswith("subsampled_"):
        n = int(level.split("_", 1)[1])
        if n not in _SUBSAMPLE_SIZES:
            raise ValueError(
                f"Unknown subsample size {n}; expected one of {_SUBSAMPLE_SIZES}."
            )
        if "agg" not in _agg_cache:
            _agg_cache["agg"] = _load_targets(os.path.join(data_dir, _TARGETS_SUBDIR, _AGG_H5))
        idx_path = os.path.join(
            data_dir, _SUBSAMPLE_SUBDIR, f"train_indices_{n}_1km.npy"
        )
        if not os.path.exists(idx_path):
            raise FileNotFoundError(
                f"Subsample indices not found: {idx_path}\nThey ship in "
                f"{_ARCHIVE_FOR[_AGG_H5]}, or regenerate with "
                f"scripts/data_prep_scripts/subsample_targets.py --num_samples_per_species {n}."
            )
        idx = np.load(idx_path)
        return f"Subsampled (N={n})", _species_counts(_agg_cache["agg"][idx, :])

    raise ValueError(
        f"Unknown level '{level}'. Expected 'full', 'aggregated', or "
        f"'subsampled_<N>' with N in {_SUBSAMPLE_SIZES}."
    )


def _pow10_formatter(x, _pos):
    """Format a positive tick value as ``10^n`` (``0`` stays ``0``)."""
    if x <= 0:
        return "0"
    return f"$10^{{{int(round(np.log10(x)))}}}$"


def _draw_hist(ax, name: str, counts: np.ndarray, n_bins: int,
               show_xlabel: bool, show_ylabel: bool, fontsize: int,
               xscale: str = "linear") -> None:
    """Render one per-species occurrence histogram onto *ax*.

    The y-axis is always log. *xscale* selects the x-axis: ``"linear"``
    (default, matching the main-text data figure) or ``"log"``.
    """
    ax.set_facecolor(_BG_COLOR)
    total = int(counts.sum())

    # Species with 0 occurrences at this level are dropped (they carry no
    # information and cannot sit on a log axis).
    positive = counts[counts > 0]
    hi = positive.max() if positive.size else 1
    if xscale == "log":
        bins = np.logspace(0, np.log10(hi), n_bins)
    else:
        bins = np.linspace(0, hi, n_bins)
    ax.hist(positive, bins=bins, color=_BAR_COLOR, edgecolor="white",
            linewidth=0.3, bottom=0.5)

    ax.set_yscale("log")
    ax.set_ylim(bottom=0.5)
    if xscale == "log":
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(_pow10_formatter))
    else:
        ax.set_xlim(left=0)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=4))
        ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    ax.tick_params(labelsize=fontsize - 4)
    ax.spines[["top", "right"]].set_visible(False)

    ax.set_title(f"{name}\n({total:,} total)", fontsize=fontsize)
    if show_xlabel:
        ax.set_xlabel("Occurrences per species", fontsize=fontsize)
    if show_ylabel:
        ax.set_ylabel("# Species", fontsize=fontsize)


def plot_occurrence_distributions(
    data_dir: str,
    levels: "list[str] | None" = None,
    n_bins: int = 50,
    xscale: str = "linear",
    fontsize: int = 20,
    panel_width: float = 5.5,
    panel_height: float = 4.5,
    out_path: "str | None" = None,
    verbose: bool = True,
) -> plt.Figure:
    """Grid of per-species occurrence-count histograms across data levels.

    One panel per requested data-preparation level. Each panel is a
    histogram of the number of occurrence records per species (log y-axis;
    x-axis linear by default, or log via ``xscale="log"``), titled with the
    level name and total occurrence count.

    Layout: up to three levels are laid out in a single row; with more, the
    first three form the top row and the remainder a (width-matched) second
    row — reproducing the paper's 3-over-4 arrangement when all seven levels
    are requested.

    Parameters
    ----------
    data_dir : str
        Absolute path to the data directory (contains ``targets/`` and
        ``targets/subsample_indices/``).
    levels : list[str], optional
        Ordered level ids to render. Recognised ids: ``"full"``,
        ``"aggregated"``, and ``"subsampled_<N>"`` for N in
        {10, 100, 1000, 10000, 100000}. Defaults to
        ``["full", "aggregated", "subsampled_1000"]``.
    n_bins : int
        Number of log-spaced histogram bins per panel.
    fontsize : int
        Base font size for titles and axis labels.
    panel_width, panel_height : float
        Per-panel size in inches.
    out_path : str, optional
        If given, save the figure there (dpi=150, facecolor preserved).
    verbose : bool
        If True, print each level's total occurrence count as it is computed
        (the heavy HDF5 reads make this useful progress output).

    Returns
    -------
    matplotlib.figure.Figure
    """
    levels = list(_DEFAULT_LEVELS if levels is None else levels)
    if not levels:
        raise ValueError("At least one level is required.")

    # ── compute per-species counts (loading the 1 km matrix at most once) ──
    agg_cache: dict = {}
    panels = []  # list of (display_name, counts)
    for level in levels:
        name, counts = _resolve_level(level, data_dir, agg_cache)
        panels.append((name, counts))
        if verbose:
            print(f"{name}: {int(counts.sum()):,} total occurrences "
                  f"across {counts.size:,} species")

    # ── row split: <=3 -> one row; else 3 on top, remainder below ──────────
    n = len(panels)
    rows = [panels] if n <= 3 else [panels[:3], panels[3:]]

    # ── inch-based layout (mirrors src/visualizations/delta_heatmap.py) ────
    # Every row spans the same content width; a row with fewer panels widens
    # each panel to fill it, so the top (3) and bottom (4) rows stay aligned.
    _gap_c, _gap_r = 1.2, 1.7                 # inter-panel gaps (in)
    _ml, _mr, _mt, _mb = 1.2, 0.35, 1.15, 1.0  # margins: ylabel / two-line titles / xlabel

    max_cols = max(len(r) for r in rows)
    content_w = max_cols * panel_width + (max_cols - 1) * _gap_c
    row_widths = [(content_w - (k - 1) * _gap_c) / k for k in (len(r) for r in rows)]

    figw = _ml + content_w + _mr
    figh = _mt + len(rows) * panel_height + (len(rows) - 1) * _gap_r + _mb

    def _rect(x0, y0, w, h):               # inches -> figure-fraction rect
        return [x0 / figw, y0 / figh, w / figw, h / figh]

    fig = plt.figure(figsize=(figw, figh))
    fig.patch.set_facecolor(_BG_COLOR)

    for r, (row, row_w) in enumerate(zip(rows, row_widths)):
        # top row first: its bottom edge is highest.
        y0 = _mb + (len(rows) - 1 - r) * (panel_height + _gap_r)
        for c, (name, counts) in enumerate(row):
            x0 = _ml + c * (row_w + _gap_c)
            ax = fig.add_axes(_rect(x0, y0, row_w, panel_height))
            _draw_hist(ax, name, counts, n_bins,
                       show_xlabel=True, show_ylabel=(c == 0), fontsize=fontsize,
                       xscale=xscale)

    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches="tight",
                    pad_inches=0.1, facecolor=_BG_COLOR)
    return fig
