"""Shared 1 km equal-area grid helpers for the data-prep pipeline.

Occurrence and plot locations are resolved to a 1 km grid in the EPSG:6933
equal-area projection to reduce spatial autocorrelation.
"""

import numpy as np
from pyproj import Transformer

CRS_WGS84 = "EPSG:4326"
CRS_EQUAL_AREA = "EPSG:6933"
RESOLUTION_M = 1000  # 1 km grid

# pyproj Transformers are stateless for `transform`; build once and reuse.
_TO_PROJECTED = Transformer.from_crs(CRS_WGS84, CRS_EQUAL_AREA, always_xy=True)
_TO_WGS84 = Transformer.from_crs(CRS_EQUAL_AREA, CRS_WGS84, always_xy=True)


def latlon_to_grid_cell(lon, lat, resolution_m=RESOLUTION_M):
    """Snap (lon, lat) to integer ``(grid_x, grid_y)`` cell indices.

    Projects to EPSG:6933 (equal-area) and floor-divides by ``resolution_m``.
    Accepts scalars or arrays; returns int64 arrays.
    """
    x, y = _TO_PROJECTED.transform(np.asarray(lon, dtype=float), np.asarray(lat, dtype=float))
    grid_x = np.floor_divide(x, resolution_m).astype(np.int64)
    grid_y = np.floor_divide(y, resolution_m).astype(np.int64)
    return grid_x, grid_y


def grid_cell_to_latlon(grid_x, grid_y, resolution_m=RESOLUTION_M):
    """Return the ``(center_lon, center_lat)`` of integer grid cell(s).

    Inverse of :func:`latlon_to_grid_cell`: takes the cell centre
    ``((grid + 0.5) * resolution_m)`` back to WGS84 lon/lat.
    """
    center_lon, center_lat = _TO_WGS84.transform(
        (np.asarray(grid_x) + 0.5) * resolution_m,
        (np.asarray(grid_y) + 0.5) * resolution_m,
    )
    return center_lon, center_lat
