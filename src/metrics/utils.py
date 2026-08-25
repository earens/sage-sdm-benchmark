"""Metric computations for the SDM LightningModule.

Defines ``MetricLogger``, which registers the configured torchmetrics on the
model, updates them incrementally, and computes/logs macro and per-species scores
at epoch end.
"""

import logging

import torch
from pytorch_lightning import LightningModule
from torch.nn import ModuleDict
from torchmetrics import MaxMetric, MeanMetric, Metric
from collections.abc import Mapping
import numpy as np

logger = logging.getLogger(__name__)


class MetricLogger:
    """Manages per-stage torchmetrics for a LightningModule.

    Registers the configured train/val/test metrics on the model, accumulates
    them incrementally during each epoch, and at epoch end computes both the
    macro score (logged to the Lightning logger / WandB) and the per-species
    score array (returned for saving to HDF5). Also tracks the running loss and
    the best validation metrics.

    Args:
        metrics: Nested ``{stage: {name: metric}}`` dict of the torchmetrics to
            manage, where stage is one of ``"train"``/``"val"``/``"test"`` and each
            value is a ``torchmetrics.Metric`` (or a further ``{sub_key: Metric}``
            mapping for grouped metrics).
        model: The LightningModule the metrics are registered on (via
            ``model.metric_dict``) and through which scores are logged.
    """

    def __init__(self, metrics: dict[str, dict[str, Metric | dict[str, Metric]]], model: LightningModule) -> None:
        self.model = model
        self.metrics = {k: v for k, v in metrics.items()}

        self.losses = {stage: MeanMetric() for stage in self.metrics.keys()}
        self.best_val_metrics = {}

        # Only create best_val_metrics if "val" stage exists
        if "val" in self.metrics:
            for metric_name, metric_obj in self.metrics["val"].items():
                if isinstance(metric_obj, Mapping):
                    for sub_key in metric_obj:
                        full_key = f"{metric_name}/{sub_key}"
                        self.best_val_metrics[full_key] = MaxMetric()
                else:
                    self.best_val_metrics[metric_name] = MaxMetric()

        self.model.metric_dict = ModuleDict(
            {
                **{f"loss_{k}": v for k, v in self.losses.items()},
                **{f"train_{k}/{sk}" if isinstance(v, Mapping) else f"train_{k}": mv
                   for k, v in (metrics.get("train") or {}).items()
                   for sk, mv in (v.items() if isinstance(v, Mapping) else [(None, v)])},
                **{f"val_{k}/{sk}" if isinstance(v, Mapping) else f"val_{k}": mv
                   for k, v in (metrics.get("val") or {}).items()
                   for sk, mv in (v.items() if isinstance(v, Mapping) else [(None, v)])},
                **{f"test_{k}/{sk}" if isinstance(v, Mapping) else f"test_{k}": mv
                   for k, v in (metrics.get("test") or {}).items()
                   for sk, mv in (v.items() if isinstance(v, Mapping) else [(None, v)])},
                **{f"best_val_{k}": v for k, v in self.best_val_metrics.items()},
            }
        )

    def log_loss(self, stage: str, loss: torch.Tensor) -> None:
        """Log loss during step (loss can be accumulated incrementally)."""
        if loss is not None:
            self.losses[stage] = self.losses[stage].to(loss.device)
            self.losses[stage](loss)
            self.model.log(f"{stage}/loss", self.losses[stage], on_step=False, on_epoch=True, prog_bar=True)

    def update_metrics(self, stage: str, preds: torch.Tensor, targets: torch.Tensor) -> None:
        """Update metrics incrementally during epoch

        Args:
            stage: "train", "val", or "test"
            preds: (batch_size, num_labels) predictions/probabilities
            targets: (batch_size, num_labels) targets with -1 for ignored
        """
        if stage not in self.metrics:
            return

        # Test metrics are per-label (average=null) and accumulate all
        # predictions until epoch end.  Keeping them on GPU wastes VRAM
        # and can cause OOM, so we move everything to CPU for test stage.
        # Val metrics (even macro-averaged) with many labels (e.g. 5771)
        # also accumulate large state tensors, so offload them too.
        if stage in ("test", "val"):
            preds = preds.detach().cpu()
            targets = targets.detach().cpu()

        for metric_name, metric_obj in self.metrics[stage].items():
            if isinstance(metric_obj, Mapping):
                for sub_key, sub_metric in metric_obj.items():
                    sub_metric = sub_metric.to(preds.device)
                    sub_metric.update(preds, targets)
            else:
                metric_obj = metric_obj.to(preds.device)
                metric_obj.update(preds, targets)

    def _compute_single_metric(self, metric_obj: Metric, metric_name: str, stage: str,
                               per_species_results: dict) -> None:
        """Compute a single metric and add results appropriately.

        Handles both scalar metrics (already averaged) and per-label metrics.
        For per-label metrics, computes macro average for logging and stores per-species results.
        """
        try:
            try:
                scores = metric_obj.compute()
            except Exception as e:
                logger.warning(f"{stage}/{metric_name}: compute failed ({type(e).__name__}); logging NaN")
                self.model.log(f"{stage}/{metric_name}", float("nan"), prog_bar=True, sync_dist=True)
                return

            if scores.ndim == 0:
                # Scalar metric (already averaged internally)
                macro_value = scores.item()
                # Check if it's a percentage-based metric or 0-1 scale
                if macro_value <= 1.0:
                    macro_value = macro_value * 100
                # Track best for val stage
                if stage == "val" and metric_name in self.best_val_metrics:
                    self.best_val_metrics[metric_name](torch.tensor(macro_value))
                    best = self.best_val_metrics[metric_name].compute().item()
                    self.model.log(f"val/{metric_name}_best", best, prog_bar=False, sync_dist=True)
                    # Also push to WandB summary so it shows in the runs table
                    if (
                        hasattr(self.model, "logger")
                        and self.model.logger is not None
                        and hasattr(self.model.logger, "experiment")
                        and hasattr(self.model.logger.experiment, "summary")
                    ):
                        try:
                            self.model.logger.experiment.summary.update(
                                {f"val/{metric_name}_best": best}
                            )
                        except Exception:
                            pass
            else:
                # Per-label metric - compute macro average and store per-species
                scores_np = scores.detach().cpu().numpy().astype(np.float32)

                # Handle NaN values for macro averaging
                valid_scores = scores_np[np.isfinite(scores_np)]
                if len(valid_scores) > 0:
                    reduce = np.median if metric_name == "auprg" else np.mean
                    macro_value = float(reduce(valid_scores)) * 100
                else:
                    macro_value = float('nan')

                # Store per-species results
                per_species_results[metric_name] = scores_np

                if metric_name == "auprg" and len(valid_scores) > 0:
                    auprg_mean = float(np.mean(valid_scores)) * 100
                    self.model.log(f"{stage}/auprg_mean", auprg_mean,
                                   prog_bar=False, sync_dist=True)
                    if (
                        hasattr(self.model, "logger")
                        and self.model.logger is not None
                        and hasattr(self.model.logger, "experiment")
                        and hasattr(self.model.logger.experiment, "log")
                    ):
                        try:
                            self.model.logger.experiment.log(
                                {f"{stage}/auprg_mean": auprg_mean,
                                 "trainer/global_step": self.model.global_step,
                                 "epoch": self.model.current_epoch},
                            )
                        except Exception:
                            pass

            # Log via PL for callbacks (ModelCheckpoint, EarlyStopping, prog_bar)
            self.model.log(f"{stage}/{metric_name}", macro_value,
                          prog_bar=True, sync_dist=True)

            # Explicitly log to wandb to ensure metrics appear in plots
            if (
                hasattr(self.model, "logger")
                and self.model.logger is not None
                and hasattr(self.model.logger, "experiment")
                and not np.isnan(macro_value)
            ):
                try:
                    self.model.logger.experiment.log(
                        {
                            f"{stage}/{metric_name}": macro_value,
                            "trainer/global_step": self.model.global_step,
                            "epoch": self.model.current_epoch,
                        },
                    )
                except Exception:
                    pass  # wandb logging is best-effort

        except Exception as e:
            logger.warning(f"Could not compute {metric_name}: {e}")
        finally:
            # Always reset to free accumulated state, even if compute() failed
            metric_obj.reset()

    def compute_and_log_epoch_metrics(self, stage: str) -> dict[str, np.ndarray]:
        """Compute all metrics from accumulated state and log macro scores.

        Args:
            stage: "val" or "test"

        Returns:
            Dict[str, np.ndarray]: metric_name -> (num_species,) array of per-species scores
        """
        if stage not in self.metrics:
            return {}

        per_species_results = {}

        for metric_name, metric_obj in self.metrics[stage].items():
            if isinstance(metric_obj, Mapping):
                for sub_key, sub_metric in metric_obj.items():
                    full_name = f"{metric_name}/{sub_key}"
                    self._compute_single_metric(sub_metric, full_name, stage, per_species_results)
            else:
                self._compute_single_metric(metric_obj, metric_name, stage, per_species_results)

        return per_species_results

    def log_best_val_metrics(self, per_species_results: dict[str, np.ndarray]) -> None:
        """Update and log best validation metrics."""
        for metric_name, scores in per_species_results.items():
            valid_scores = scores[np.isfinite(scores)]
            if len(valid_scores) == 0:
                continue

            # match the val logging reduction: AUPRG uses the robust median.
            reduce = np.median if metric_name == "auprg" else np.mean
            current_value = float(reduce(valid_scores)) * 100

            if metric_name in self.best_val_metrics:
                self.best_val_metrics[metric_name](torch.tensor(current_value))
                self.model.log(
                    f"val/{metric_name}_best",
                    self.best_val_metrics[metric_name].compute(),
                    sync_dist=True,
                    prog_bar=True,
                )

    def reset_losses(self, stage: str) -> None:
        """Reset loss for a given stage."""
        if stage in self.losses:
            self.losses[stage].reset()

    def reset_metrics(self, stage: str) -> None:
        """Reset all metrics for a given stage."""
        if stage in self.losses:
            self.losses[stage].reset()

        # Also reset the actual metric objects
        if stage in self.metrics:
            for metric_name, metric_obj in self.metrics[stage].items():
                if isinstance(metric_obj, Mapping):
                    for sub_metric in metric_obj.values():
                        sub_metric.reset()
                else:
                    metric_obj.reset()

    def reset_val_metrics(self) -> None:
        """Reset best val metrics (called on train start)."""
        if "val" in self.losses:
            self.losses["val"].reset()
        for best_val_mm in self.best_val_metrics.values():
            best_val_mm.reset()
