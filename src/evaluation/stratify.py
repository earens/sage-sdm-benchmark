"""Stratified metric breakdown by species properties (the delta-heatmap axes).

Reports per-species metric means within the 2D grid used by the paper's delta
heatmaps — `sampling_effort` (x) × `relative_prevalence` (y): the four
median-split quadrants, plus the extreme outer bins of each axis. With a
reference model, every cell reports the delta (model minus reference), matching
the heatmap's "M2 − M1" reading so users see *where* one model beats another.

Species properties are read from the benchmark's `species_properties*.csv`
(`compute_species_ranks` output); rows are aligned by `species_id` (0-indexed).
"""

from __future__ import annotations


import numpy as np
import pandas as pd

from src.evaluation.regions import rank_axes, region_masks

DEFAULT_X = "sampling_effort"
DEFAULT_Y = "relative_prevalence"

# Ecological (low, high) labels for known axes; falls back to ("low", "high").
AXIS_LABELS = {
    "sampling_effort": ("sparse", "dense"),
    "relative_prevalence": ("infrequent", "frequent"),
    "test_prevalence": ("infrequent", "frequent"),
    "train_prevalence": ("infrequent", "frequent"),
    "range_size_km2": ("small-range", "wide-range"),
    "n_cells_in_range": ("small-range", "wide-range"),
}


def _axis_labels(col: str):
    return AXIS_LABELS.get(col, ("low", "high"))


def _cell_value(vals: np.ndarray, ref: np.ndarray | None, mask: np.ndarray):
    """Mean of `vals` over `mask` (percent); if `ref` given, mean per-species delta."""
    if ref is None:
        v = vals[mask]
        v = v[np.isfinite(v)]
        return (float(np.mean(v) * 100) if v.size else float("nan")), int(v.size)
    d = (vals - ref)[mask]
    d = d[np.isfinite(d)]
    return (float(np.mean(d) * 100) if d.size else float("nan")), int(d.size)


def stratified_report(
    results: dict[str, np.ndarray],
    proxies: pd.DataFrame,
    x_col: str = DEFAULT_X,
    y_col: str = DEFAULT_Y,
    reference: dict[str, np.ndarray] | None = None,
    extreme_pct: float = 10.0,
    metrics: list[str] | None = None,
) -> str:
    """Build a human-readable stratified-scores table (quadrants + extreme bins)."""
    metrics = metrics or list(results.keys())
    n = len(next(iter(results.values())))
    for col in (x_col, y_col):
        if col not in proxies.columns:
            return f"[stratified] property '{col}' not in proxies; skipping stratified report."

    idx = proxies.set_index("species_id")
    x = idx[x_col].reindex(range(n)).to_numpy(dtype=float)
    y = idx[y_col].reindex(range(n)).to_numpy(dtype=float)

    x = np.where(np.isfinite(x), x, np.nan)
    y = np.where(np.isfinite(y), y, np.nan)
    fin = np.isfinite(x) & np.isfinite(y)
    fin_idx = np.where(fin)[0]
    n_fin = int(fin.sum())


    decile = max(2, round(100.0 / extreme_pct)) if extreme_pct else 10
    lo = round(100.0 / decile)
    se_rank, tp_rank = rank_axes(x[fin], y[fin])
    sub = region_masks(se_rank, tp_rank, n_fin, decile=decile)

    def _full(submask):
        m = np.zeros(n, dtype=bool)
        m[fin_idx[submask]] = True
        return m

    xl, xh = _axis_labels(x_col)
    yl, yh = _axis_labels(y_col)
    quadrants = [
        (f"{xl}-{yl}", _full(sub["bottom_left"])),
        (f"{xl}-{yh}", _full(sub["top_left"])),
        (f"{xh}-{yl}", _full(sub["bottom_right"])),
        (f"{xh}-{yh}", _full(sub["top_right"])),
    ]
    # Extreme-decile L-strips: p10 on one axis, split at the median on the other.
    extremes = [
        (f"{xl}(p{lo}) x {yl}-half", _full(sub["L3_left_lower"])),
        (f"{xl}(p{lo}) x {yh}-half", _full(sub["L4_left_upper"])),
        (f"{yl}(p{lo}) x {xl}-half", _full(sub["L1_bottom_left"])),
        (f"{yl}(p{lo}) x {xh}-half", _full(sub["L2_bottom_right"])),
    ]

    kind = "delta vs reference (M-ref)" if reference is not None else "mean"
    header = (f"Stratified {kind} (%) — x={x_col} [{xl}<->{xh}], "
              f"y={y_col} [{yl}<->{yh}]")
    col_w = max(10, *(len(m) for m in metrics))
    lines = [header, "  " + f"{'stratum':<30}{'n':>6}  " + "".join(f"{m:>{col_w}}" for m in metrics)]

    def row(label, mask):
        cells = []
        n_used = 0
        for m in metrics:
            val, nc = _cell_value(results[m], reference[m] if reference else None, mask)
            n_used = max(n_used, nc)
            cells.append(f"{val:>{col_w}.2f}" if np.isfinite(val) else f"{'nan':>{col_w}}")
        return f"  {label:<30}{n_used:>6}  " + "".join(cells)

    lines.append("  -- quadrants (median split) --")
    for label, mask in quadrants:
        lines.append(row(label, mask))
    lines.append(f"  -- extreme L-strips (p{lo} x median split, matches heatmap) --")
    for label, mask in extremes:
        lines.append(row(label, mask))
    return "\n".join(lines)
