"""Standalone evaluation toolkit for the SDM benchmark.

This package lets anyone evaluate their own model predictions against the
benchmark's presence-absence test set *without* running the PyTorch Lightning
training loop. It reuses the exact same metric definitions used during training
(``src.metrics``), so numbers are directly comparable to the paper.

Entry points:
    - ``prediction_io``: load / validate / save the prediction file format
    - ``metrics``: compute per-species AUROC / AUPRG

See ``scripts/evaluate_predictions.py`` for the command-line interface.
"""

from src.evaluation.metrics import (
    compute_species_metrics,
    macro_summary,
    save_metrics_h5,
)
from src.evaluation.prediction_io import (
    PREDICTION_FORMAT,
    load_ground_truth_targets,
    load_predictions,
)

__all__ = [
    "compute_species_metrics",
    "macro_summary",
    "save_metrics_h5",
    "load_predictions",
    "load_ground_truth_targets",
    "PREDICTION_FORMAT",
]
