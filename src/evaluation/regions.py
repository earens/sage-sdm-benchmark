"""Shared species-stratification regions for the property grid.

Eight rank-based regions over two ranked properties (e.g. sampling effort x
relative prevalence):

- four median-split **quadrants** (rank <= N/2 vs > N/2 on each axis);
- four extreme-decile **L-strips** — p10 on one axis, split at the median on the
  other (so each strip is "p10 x p50"); the extreme corner is split diagonally
  between the two overlapping strips.
"""

import numpy as np
from scipy.stats import rankdata


def rank_axes(x, y):
    """Return 1-based ordinal ranks of two 1-D property arrays (grid convention)."""
    return rankdata(x, method="ordinal").astype(int), rankdata(y, method="ordinal").astype(int)


def region_masks(se_rank: np.ndarray, tp_rank: np.ndarray, N: int, decile: int = 10) -> dict:
    """Binary species masks for the 8 grid regions.

    Args:
        se_rank, tp_rank: 1-based ordinal ranks of the two properties.
        N: number of species.
        decile: the L-strip extreme fraction is ``1/decile`` (default 10 -> p10).

    Quadrants split at the median (``N // 2``); L-strips use ``N // decile`` on
    the extreme axis and the median on the other.
    """
    M, L = N // 2, N // decile
    in_corner = (se_rank <= L) & (tp_rank <= L)
    diag_to_left = in_corner & (se_rank < tp_rank)
    diag_to_bot = in_corner & (se_rank >= tp_rank)
    return {
        "bottom_left":     (se_rank <= M) & (tp_rank <= M),
        "bottom_right":    (se_rank >  M) & (tp_rank <= M),
        "top_left":        (se_rank <= M) & (tp_rank >  M),
        "top_right":       (se_rank >  M) & (tp_rank >  M),
        "L1_bottom_left":  (tp_rank <= L) & (se_rank <= M) & ~diag_to_left,
        "L2_bottom_right": (tp_rank <= L) & (se_rank >  M),
        "L3_left_lower":   (se_rank <= L) & (tp_rank <= M) & ~diag_to_bot,
        "L4_left_upper":   (se_rank <= L) & (tp_rank >  M),
    }
