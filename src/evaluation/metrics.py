"""Trainer-independent per-species metric computation.

This reproduces the metrics computed at test time in
``src.modules.lightning_module.LightningModule.on_test_epoch_end`` — AUROC and
AUPRG (Flach & Kull 2015) by default — but from a
plain ``(probabilities, targets)`` pair instead of the Lightning loop. It uses the *same* metric classes
(``torchmetrics`` + ``src.metrics.precision_recall_gain.MultilabelAUPRG``), so
results match the training-time ``metrics_by_species.h5`` exactly.

All metrics are per-species (``average=None``) and ignore entries where the
target equals ``-1`` (out-of-range species), matching the benchmark's
range-masked evaluation.
"""

from __future__ import annotations


import h5py
import hdf5plugin  # noqa: F401  — registers the Blosc filters the data is written with
import numpy as np
import torch
from torchmetrics.classification import (
    MultilabelAUROC,
    MultilabelAveragePrecision,
    MultilabelF1Score,
)

from src.metrics.precision_recall_gain import MultilabelAUPRG

IGNORE_INDEX = -1


DEFAULT_METRICS: list[str] = ["auroc", "auprg"]


def _build_metrics(num_labels: int, names: list[str]) -> dict[str, object]:
    factory = {
        "auroc": lambda: MultilabelAUROC(num_labels=num_labels, average=None, ignore_index=IGNORE_INDEX),
        "f1": lambda: MultilabelF1Score(
            num_labels=num_labels, average=None, threshold=0.5, ignore_index=IGNORE_INDEX
        ),
        "auprg": lambda: MultilabelAUPRG(num_labels=num_labels, average=None, ignore_index=IGNORE_INDEX),
        "auprc": lambda: MultilabelAveragePrecision(
            num_labels=num_labels, average=None, ignore_index=IGNORE_INDEX
        ),
    }
    unknown = set(names) - set(factory)
    if unknown:
        raise ValueError(f"Unknown metric(s): {sorted(unknown)}. Available: {sorted(factory)}")
    return {name: factory[name]() for name in names}


def compute_species_metrics(
    probabilities: np.ndarray,
    targets: np.ndarray,
    metrics: list[str] | None = None,
    batch_size: int = 8192,
    device: str = "cpu",
) -> dict[str, np.ndarray]:
    """Compute per-species metrics from predictions and ground-truth targets.

    Args:
        probabilities: (N_locations, N_species) predicted probabilities in [0, 1].
        targets: (N_locations, N_species) with {0, 1} and ``-1`` for ignored.
        metrics: Metric names to compute (default: AUROC, AUPRG).
        batch_size: Number of locations pushed into each ``update`` call.
        device: Torch device for the metric state (``"cpu"`` or ``"cuda"``).

    Returns:
        dict mapping metric name -> (N_species,) float32 array of per-species
        scores (``nan`` where a species has no valid positives/negatives).
    """
    if probabilities.shape != targets.shape:
        raise ValueError(
            f"probabilities {probabilities.shape} and targets {targets.shape} must have the same shape."
        )
    metric_names = metrics or DEFAULT_METRICS
    num_labels = probabilities.shape[1]
    metric_objs = {name: m.to(device) for name, m in _build_metrics(num_labels, metric_names).items()}

    n = probabilities.shape[0]
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        preds = torch.from_numpy(np.ascontiguousarray(probabilities[start:end])).float().to(device)
        tgt = torch.from_numpy(np.ascontiguousarray(targets[start:end])).long().to(device)
        for m in metric_objs.values():
            m.update(preds, tgt)

    results = {}
    for name, m in metric_objs.items():
        scores = m.compute().detach().cpu().numpy().astype(np.float32)
        results[name] = scores
    return results


def macro_summary(results: dict[str, np.ndarray]) -> dict[str, float]:
    """Macro-average each per-species metric over its finite entries, in percent."""
    summary = {}
    for name, scores in results.items():
        valid = scores[np.isfinite(scores)]
        summary[name] = float(np.mean(valid) * 100) if valid.size else float("nan")
        if name == "auprg":
            summary["auprg_median"] = float(np.median(valid) * 100) if valid.size else float("nan")
    return summary


def save_metrics_h5(results: dict[str, np.ndarray], path: str, prefix: str = "test") -> None:
    """Save per-species metrics to HDF5 as ``<prefix>_<metric>`` datasets.

    Matches the layout written during training so downstream plotting code can
    read either interchangeably.
    """
    with h5py.File(path, "w") as f:
        for name, scores in results.items():
            f.create_dataset(f"{prefix}_{name}", data=scores.astype(np.float32), dtype="float32")
