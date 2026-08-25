"""Smoke tests for model forward passes and the BCE loss.

Run with:  python -m pytest tests/test_models.py -v
These tests use only CPU and synthetic data — no dataset files required.
"""

import torch
import pytest

N = 8          # batch size
N_FEATURES = 52
N_SPECIES = 100


def _batch(n_features=N_FEATURES):
    """Return (locations, predictors) with realistic coordinate ranges."""
    lons = torch.linspace(-10, 30, N)
    lats = torch.linspace(35, 70, N)
    locations = torch.stack([lons, lats], dim=1)
    predictors = torch.randn(N, n_features)
    return locations, predictors


# ---------------------------------------------------------------------------
# Location encoding
# ---------------------------------------------------------------------------

def test_encode_locations_shape():
    from src.models.location_encoding import encode_locations
    loc, _ = _batch()
    out = encode_locations(loc)
    assert out.shape == (N, 4)


def test_encode_locations_finite():
    from src.models.location_encoding import encode_locations
    loc, _ = _batch()
    assert torch.isfinite(encode_locations(loc)).all()


# ---------------------------------------------------------------------------
# ResNet
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("use_location", [True, False])
def test_resnet_forward(use_location):
    from src.models.resnet import ResNet
    model = ResNet(num_classes=N_SPECIES, use_location=use_location)
    loc, pred = _batch()
    out = model(loc, pred)
    assert out.shape == (N, N_SPECIES)
    assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# SimpleMLP
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("use_location", [True, False])
def test_simple_mlp_forward(use_location):
    from src.models.simple_mlp import SimpleMLP
    model = SimpleMLP(num_classes=N_SPECIES, use_location=use_location)
    loc, pred = _batch()
    out = model(loc, pred)
    assert out.shape == (N, N_SPECIES)
    assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# Linear (MaxEnt-style)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("maxent_features", ["l", "lq", "lqh"])
def test_linear_forward(maxent_features):
    from src.models.linear import Linear
    model = Linear(num_classes=N_SPECIES, maxent_features=maxent_features)
    loc, pred = _batch()
    out = model(loc, pred)
    assert out.shape == (N, N_SPECIES)
    assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# FTTransformer
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("use_location", [True, False])
def test_fttransformer_forward(use_location):
    from src.models.fftransformer import FTTransformer
    model = FTTransformer(
        num_classes=N_SPECIES,
        num_features=N_FEATURES,
        hidden_dim=32,
        num_blocks=2,
        n_heads=4,
        use_location=use_location,
    )
    loc, pred = _batch()
    out = model(loc, pred)
    assert out.shape == (N, N_SPECIES)
    assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# BCE loss
# ---------------------------------------------------------------------------

def test_bce_forward_no_species_weights():
    from src.criteria.bce import BCE
    loss_fn = BCE(lambda_1=512.0, lambda_2=0.5)
    logits = torch.randn(N, N_SPECIES)
    targets = torch.randint(0, 2, (N, N_SPECIES)).float()
    obs_mask = torch.cat([torch.ones(N // 2, dtype=torch.bool),
                          torch.zeros(N // 2, dtype=torch.bool)])
    loss = loss_fn(logits, targets, obs_mask=obs_mask)
    assert loss.shape == ()
    assert torch.isfinite(loss)


def test_bce_forward_with_range_mask():
    from src.criteria.bce import BCE
    loss_fn = BCE(lambda_1=512.0, lambda_2=0.5)
    logits = torch.randn(N, N_SPECIES)
    targets = torch.randint(0, 2, (N, N_SPECIES)).float()
    obs_mask = torch.cat([torch.ones(N // 2, dtype=torch.bool),
                          torch.zeros(N // 2, dtype=torch.bool)])
    range_mask = torch.randint(0, 2, (N, N_SPECIES)).float()
    loss = loss_fn(logits, targets, obs_mask=obs_mask, range_mask=range_mask)
    assert loss.shape == ()
    assert torch.isfinite(loss)


def test_bce_forward_with_species_weights():
    from src.criteria.bce import BCE
    loss_fn = BCE(lambda_1=512.0, lambda_2=0.5)
    species_w = torch.rand(N_SPECIES) + 0.1
    loss_fn.set_species_weights(species_w)
    logits = torch.randn(N, N_SPECIES)
    targets = torch.randint(0, 2, (N, N_SPECIES)).float()
    obs_mask = torch.cat([torch.ones(N // 2, dtype=torch.bool),
                          torch.zeros(N // 2, dtype=torch.bool)])
    loss = loss_fn(logits, targets, obs_mask=obs_mask)
    assert loss.shape == ()
    assert torch.isfinite(loss)
