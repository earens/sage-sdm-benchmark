"""Interactive threshold slider for prediction maps.

Decorates a prediction-heatmap figure (from :mod:`render`) with a Plotly slider:
the first position ("continuous") shows the raw probability heatmap; each
subsequent position binarizes the heatmap at a threshold and reveals TP/TN/FP/FN
confusion points (as ``legendonly`` traces the viewer can toggle on). Fully
self-contained — no training/datamodule state.
"""

import numpy as np
import plotly.graph_objects as go

from src.visualizations.maps.render import make_prediction_map_fig

# Two-tone colorscale used when the heatmap is binarized at a threshold.
BINARY_COLORSCALE = [[0.0, "#EDE7E3"], [1.0, "#2a1155"]]
# Colorscale for the "continuous" (raw probability) slider position.
CONTINUOUS_COLORSCALE = [
    [0.00, "#FAF6F4"],
    [0.25, "#E2D0CB"],
    [0.50, "#BFAFC4"],
    [0.75, "#7C88C3"],
    [1.00, "#3F3FA3"],
]

_CATEGORIES = [
    ("FN", "#bd6761"),
    ("TN", "#7fa07e"),
    ("FP", "#ee382b"),
    ("TP", "#64c95e"),
]


def add_threshold_slider(
    fig: go.Figure,
    obs_locations: np.ndarray,
    obs_targets: np.ndarray,
    obs_probs: np.ndarray,
    probs_map: np.ndarray,
    heatmap_trace_idx: int = 0,
    tn_subsample: int = 10000,
    binary_colorscale: list = None,
    continuous_map: np.ndarray = None,
    continuous_colorscale: list = None,
) -> go.Figure:
    """Add TP/TN/FP/FN confusion points at several thresholds plus a slider.

    Parameters
    ----------
    fig               : figure already containing the prediction heatmap
    obs_locations     : (N, 2) [lon, lat] of evaluation observations
    obs_targets       : (N,) binary ground-truth labels
    obs_probs         : (N,) predicted probabilities at those observations
    probs_map         : the (H, W) heatmap array shown in ``fig`` (raw probabilities)
    heatmap_trace_idx : index of the heatmap trace in ``fig.data`` (0 for a render fig)
    tn_subsample      : cap on true-negative points drawn (None = all)
    binary_colorscale : colorscale for the binarized heatmap steps
    """
    binary_colorscale = binary_colorscale or BINARY_COLORSCALE
    # By default the "continuous" step reuses the raw heatmap + slider palette; the
    # caller can pass a rescaled map + the static magma palette to match the static map.
    continuous_map = continuous_map if continuous_map is not None else probs_map
    continuous_colorscale = continuous_colorscale or CONTINUOUS_COLORSCALE
    obs_locations = np.asarray(obs_locations)
    obs_targets = np.asarray(obs_targets).astype(int)
    obs_probs = np.asarray(obs_probs).astype(float)

    # Fixed TN subsample (consistent across thresholds).
    tn_subsample_idx = None
    if tn_subsample is not None:
        neg_idx = np.where(obs_targets == 0)[0]
        if neg_idx.size > tn_subsample:
            rng = np.random.default_rng(0)
            tn_subsample_idx = set(rng.choice(neg_idx, size=tn_subsample, replace=False).tolist())

    thresholds = np.round(np.arange(0.05, 1.0, 0.05), 2)
    traces_per_threshold = len(_CATEGORIES)  # 4
    n_base_traces = len(fig.data)
    base_showlegend = [bool(t.showlegend) if t.showlegend is not None else False for t in fig.data]
    base_names = [t.name for t in fig.data]
    hm_idx = heatmap_trace_idx

    binarized_maps = {
        t: np.where(np.isnan(probs_map), np.nan, np.where(probs_map >= t, 1.0, 0.0))
        for t in thresholds
    }

    # Add 4 confusion traces per threshold (all start hidden).
    for thresh in thresholds:
        preds = (obs_probs >= thresh).astype(int)
        masks = [
            (preds == 0) & (obs_targets == 1),  # FN
            (preds == 0) & (obs_targets == 0),  # TN
            (preds == 1) & (obs_targets == 0),  # FP
            (preds == 1) & (obs_targets == 1),  # TP
        ]
        if tn_subsample_idx is not None:
            full_tn = np.where(masks[1])[0]
            keep = np.array([i for i in full_tn if i in tn_subsample_idx])
            masks[1] = np.zeros_like(masks[1], dtype=bool)
            if len(keep) > 0:
                masks[1][keep] = True

        for (name, color), mask in zip(_CATEGORIES, masks):
            pts = obs_locations[mask] if np.any(mask) else np.empty((0, 2))
            fig.add_trace(go.Scatter(
                x=pts[:, 0], y=pts[:, 1], mode="markers",
                marker=dict(color=color, size=12, line=dict(color="white", width=1)),
                name=f"{name} ({int(mask.sum())})",
                legendgroup=name, showlegend=False, visible=False,
                hovertemplate=f"{name}<br>Lon=%{{x:.3f}}<br>Lat=%{{y:.3f}}<extra></extra>",
            ))

    total_traces = len(fig.data)
    steps = []

    def _restyle(z, colorscale, visible, showlegend, name):
        return dict(method="restyle", args=[{
            "z": z, "colorscale": colorscale, "zmin": [None] * total_traces,
            "zmax": [None] * total_traces, "visible": visible,
            "showlegend": showlegend, "name": name,
        }])

    # Step 0: continuous — raw probabilities, all confusion points hidden.
    z = [None] * total_traces; cs = [None] * total_traces
    z[hm_idx] = continuous_map.tolist(); cs[hm_idx] = continuous_colorscale
    step = _restyle(
        z, cs,
        [True] * n_base_traces + [False] * (total_traces - n_base_traces),
        list(base_showlegend) + [False] * (total_traces - n_base_traces),
        list(base_names) + [None] * (total_traces - n_base_traces),
    )
    step["args"][0]["zmin"][hm_idx] = 0; step["args"][0]["zmax"][hm_idx] = 1
    step["label"] = "continuous"
    steps.append(step)

    # Steps 1..N: binarize the heatmap and reveal this threshold's confusion points.
    for t_idx, thresh in enumerate(thresholds):
        visibility = [True] * n_base_traces + [False] * (total_traces - n_base_traces)
        start = n_base_traces + t_idx * traces_per_threshold
        for k, (cat, _) in enumerate(_CATEGORIES):
            # Show TP / FP / FN as soon as the slider moves to a threshold; true
            # negatives are numerous and least informative, so leave them toggle-only.
            visibility[start + k] = "legendonly" if cat == "TN" else True

        preds = (obs_probs >= thresh).astype(int)
        counts = {
            "TP": int(((preds == 1) & (obs_targets == 1)).sum()),
            "TN": int(((preds == 0) & (obs_targets == 0)).sum()),
            "FP": int(((preds == 1) & (obs_targets == 0)).sum()),
            "FN": int(((preds == 0) & (obs_targets == 1)).sum()),
        }
        showlegend = list(base_showlegend) + [False] * (total_traces - n_base_traces)
        names = list(base_names) + [None] * (total_traces - n_base_traces)
        for k, (cat, _) in enumerate(_CATEGORIES):
            showlegend[start + k] = True
            names[start + k] = f"{cat} ({counts[cat]})"

        z = [None] * total_traces; cs = [None] * total_traces
        z[hm_idx] = binarized_maps[thresh].tolist(); cs[hm_idx] = binary_colorscale
        step = _restyle(z, cs, visibility, showlegend, names)
        step["args"][0]["zmin"][hm_idx] = 0; step["args"][0]["zmax"][hm_idx] = 1
        step["label"] = f"{thresh:.2f}"
        steps.append(step)

    # Start on a 0.5 threshold (not "continuous") so the confusion points and their
    # legend are visible on load; apply that step's state to the initial figure since
    # plotly only shows the raw traces until the slider is moved.
    default_idx = 1 + int(np.argmin(np.abs(thresholds - 0.5)))
    da = steps[default_idx]["args"][0]
    for ti in range(total_traces):
        if da["z"][ti] is not None:
            fig.data[ti].z = da["z"][ti]
        if da["colorscale"][ti] is not None:
            fig.data[ti].colorscale = da["colorscale"][ti]
        if da["zmin"][ti] is not None:
            fig.data[ti].zmin = da["zmin"][ti]; fig.data[ti].zmax = da["zmax"][ti]
        fig.data[ti].visible = da["visible"][ti]
        fig.data[ti].showlegend = da["showlegend"][ti]
        if da["name"][ti] is not None:
            fig.data[ti].name = da["name"][ti]

    fig.update_layout(
        sliders=[dict(
            active=default_idx, currentvalue={"prefix": "Threshold: "},
            pad={"t": 30}, steps=steps,
        )],
        # room for the title (top), the slider + tick labels (bottom), and the wide
        # "continuous" tick label that would otherwise overflow the left edge.
        margin=dict(t=48, b=88, l=45, r=25),
        showlegend=True,           # reveal the confusion-point legend (hidden by default)
    )
    return fig


def make_interactive_map(
    lons: np.ndarray,
    lats: np.ndarray,
    land_flat_indices: np.ndarray,
    predictions: np.ndarray,
    obs_locations: np.ndarray,
    obs_targets: np.ndarray,
    obs_probs: np.ndarray,
    crop_bounds,
    range_gdf,
    world_gdf,
    title: str = "Prediction",
    tn_subsample: int = 10000,
) -> go.Figure:
    """Build a prediction map with an interactive threshold slider.

    The "continuous" slider position shows the **rescaled** surface — identical
    palette and contrast stretch to the static prediction map
    (:func:`make_prediction_map_fig` / ``generate_maps.py``) — while the
    thresholded positions binarize the **raw** probabilities, so a 0.5 cut-off
    means an actual 0.5 probability and stays consistent with the raw ``obs_probs``
    of the confusion points.
    """
    from src.visualizations.maps import render

    fig = make_prediction_map_fig(
        lons, lats, land_flat_indices, predictions,
        crop_bounds, range_gdf, world_gdf, title=title,
        rescale=True, mask_to_range=True,
    )
    continuous_map = np.asarray(fig.data[0].z)

    raw_fig = make_prediction_map_fig(
        lons, lats, land_flat_indices, predictions,
        crop_bounds, range_gdf, world_gdf,
        rescale=False, mask_to_range=True,
    )
    probs_map = np.asarray(raw_fig.data[0].z)

    return add_threshold_slider(
        fig, obs_locations, obs_targets, obs_probs, probs_map,
        heatmap_trace_idx=0, tn_subsample=tn_subsample,
        continuous_map=continuous_map, continuous_colorscale=render.CONTINUOUS_COLORSCALE,
    )
