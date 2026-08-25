"""Shared location encoding used by all model architectures.

Encodes (lon, lat) coordinates into 4-dimensional sine/cosine features as
proposed in SINR (Cole et al., ICML 2023).
"""

import math

import torch


def encode_locations(locations: torch.Tensor) -> torch.Tensor:
    """Encode a batch of (lon, lat) coordinates into sine/cosine features.

    Args:
        locations: Tensor of shape (N, 2) with columns [longitude, latitude]
                   in degrees (lon ∈ [-180, 180], lat ∈ [-90, 90]).

    Returns:
        Tensor of shape (N, 4): [sin(π·lon'), cos(π·lon'), sin(π·lat'), cos(π·lat')]
    """
    lon, lat = locations[:, 0], locations[:, 1]
    lon = lon / 180.0
    lat = lat / 90.0
    return _sine_cosine_encode(lon, lat)


def _sine_cosine_encode(lon: torch.Tensor, lat: torch.Tensor) -> torch.Tensor:
    pi = torch.tensor(math.pi)
    return torch.stack([
        torch.sin(pi * lon),
        torch.cos(pi * lon),
        torch.sin(pi * lat),
        torch.cos(pi * lat),
    ], dim=-1)
