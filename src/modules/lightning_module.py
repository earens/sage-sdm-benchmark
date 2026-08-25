"""LightningModule for multi-label species distribution modelling.

Trains on presence-only data with background points (weighted BCE), evaluates on
presence-absence plots with per-species range-masked metrics, and at test time
saves per-species metrics, per-location predictions, and continental maps to HDF5.
"""

import logging
from typing import Any

import pytorch_lightning as pl
import torch
from torch import nn
from torchmetrics.metric import Metric

from src.metrics.utils import MetricLogger

import os
import gc
import h5py
import hdf5plugin  # noqa: F401  — registers the Blosc filters the data is written with

logger = logging.getLogger(__name__)


class LightningModule(pl.LightningModule):
    """LightningModule for multi-label species distribution modelling.

    Trains on presence-only (PO) data with background points and evaluates on
    presence-absence (PA) data.  Per-species metrics and per-location predictions
    are saved to HDF5 at the end of each val/test epoch.

    Args:
        net: Model mapping ``(locations, predictors)`` to per-species logits.
        loss: Loss module (e.g. the weighted ``BCE``); receives logits, targets,
            an observation mask, and an optional range mask.
        metrics: Nested ``{stage: {name: torchmetrics.Metric}}`` for the
            ``train``/``val``/``test`` stages.
        species_weighting: Optional per-species loss-weighting config, e.g.
            ``{"method": "inversely_proportional_sqrt"}`` (see ``criteria.bce``).
        continent_map_configs: Optional list of dicts (each with ``continent``,
            ``species_ids`` and optional ``res``) driving continental map
            generation via ``src.visualizations.maps`` — static maps logged to
            wandb each validation epoch, interactive HTML maps saved at test time.
    """

    def __init__(
        self,
        net: nn.Module,
        loss: nn.Module,
        metrics: dict[str, dict[str, Metric]],
        species_weighting: dict[str, Any] | None = None,
        continent_map_configs: list[dict] = None,
    ) -> None:

        super().__init__()
        self.save_hyperparameters(ignore=["net", "loss", "metrics"])
        self.net = net
        self.loss = loss
        self.species_weighting = species_weighting

        self.metric_logger = MetricLogger(metrics, self)

        self.continent_map_configs = continent_map_configs or []

        # Accumulate per-location test predictions for saving
        self.test_all_probs: list[torch.Tensor] = []
        self.test_all_targets: list[torch.Tensor] = []
        self.test_all_locations: list[torch.Tensor] = []

    def forward(self, locations, predictors):
        return self.net(locations, predictors)

    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so we need to make sure val_acc_best doesn't store accuracy from these checks
        self.metric_logger.reset_val_metrics()

    def model_step(self, batch: Any):
        """Concatenate observations with background points, run the net, and
        return ``(logits, targets, range_mask)`` for the combined batch."""
        targets = batch["observation_targets"]

        if "bg_locations" in batch.keys():
            locations = torch.cat([batch["observation_locations"], batch["bg_locations"]], dim=0)
            predictors = torch.cat([batch["observation_predictors"], batch["bg_predictors"]], dim=0)
            bg_targets = torch.zeros(batch["bg_locations"].shape[0], targets.shape[1],
                                     device=targets.device, dtype=targets.dtype)
            targets = torch.cat([targets, bg_targets], dim=0)
        else:
            locations = batch["observation_locations"]
            predictors = batch["observation_predictors"]

        logits = self.forward(locations, predictors)
        range_mask = batch.get("range_mask", None)
        bg_range_mask = batch.get("bg_range_mask", None)
        if range_mask is not None and bg_range_mask is not None:
            range_mask = torch.cat([range_mask, bg_range_mask], dim=0)



        return logits, targets, range_mask

    def compute_loss_and_masked_targets(self, logits, targets, obs_mask=None, range_mask=None):
        """Compute loss and apply range mask to targets for metrics.
        """
        loss = self.loss(logits, targets, obs_mask=obs_mask, range_mask=range_mask)

        # Apply range mask to targets for metrics
        if range_mask is not None:
            targets = torch.where(
                range_mask.bool(),
                targets,
                torch.tensor(-1, device=targets.device)
            )
        return loss, targets, logits

    def training_step(self, batch: Any, batch_idx: int):
        """Presence-only training step: weighted BCE over observations + background."""
        logits, targets, range_mask = self.model_step(batch)

        if "bg_locations" in batch.keys():
            # Presence-only data with background points
            obs_size = batch["observation_targets"].shape[0]
            batch_size = logits.shape[0]
            obs_mask = torch.cat([
                torch.ones(obs_size, dtype=torch.bool, device=logits.device),
                torch.zeros(batch_size - obs_size, dtype=torch.bool, device=logits.device)
            ])
        else:
            # Presence-absence data
            obs_mask = None

        loss, targets, logits = self.compute_loss_and_masked_targets(
            logits, targets, obs_mask, range_mask
        )

        # Only log loss for training
        self.metric_logger.log_loss("train", loss)

        return {"loss": loss, "logits": logits, "targets": targets}

    def on_train_epoch_end(self):
        # Re-sample the overflow bg mapping for the next epoch so that
        # each epoch pairs observations with different background points
        # (deterministic data augmentation).
        dm = getattr(self.trainer, 'datamodule', None)
        if dm is not None:
            ds = getattr(dm, 'data_train', None)
            if ds is not None and hasattr(ds, 'set_epoch'):
                ds.set_epoch(self.current_epoch + 1)

    def on_validation_epoch_start(self):
        """Reset validation metrics at the start of each validation epoch."""
        self.metric_logger.reset_metrics("val")

    def validation_step(self, batch: Any, batch_idx: int):
        """Presence-absence validation step: accumulate range-masked per-species metrics."""
        logits, targets, range_mask = self.model_step(batch)

        obs_mask = torch.ones(logits.shape[0], dtype=torch.bool, device=logits.device)
        loss, targets, logits = self.compute_loss_and_masked_targets(
            logits, targets, obs_mask, range_mask
        )

        self.metric_logger.log_loss("val", loss)

        with torch.no_grad():
            probs = torch.sigmoid(logits)
        self.metric_logger.update_metrics("val", probs, targets)

        return {"loss": loss}

    def on_validation_epoch_end(self):
        # Skip metric computation during sanity check – too few batches
        # for AUROC (needs both positive and negative samples per label).
        if self.trainer.sanity_checking:
            self.metric_logger.reset_metrics("val")
            return

        per_species_results = self.metric_logger.compute_and_log_epoch_metrics("val")

        # Log best metrics
        self.metric_logger.log_best_val_metrics(per_species_results)

        # Save per-species metrics to HDF5
        if getattr(self.trainer, "global_rank", 0) == 0 and per_species_results:
            out_dir = os.path.join(
                self.trainer.logger.save_dir,
                self.trainer.logger.name,
                self.trainer.logger.version
            )
            os.makedirs(out_dir, exist_ok=True)

            metrics_h5_path = os.path.join(out_dir, "metrics_by_species.h5")
            mode = "a" if os.path.exists(metrics_h5_path) else "w"

            with h5py.File(metrics_h5_path, mode) as f:
                for metric_name, values in per_species_results.items():
                    dset_name = f"val_{metric_name}"
                    if dset_name in f:
                        del f[dset_name]
                    f.create_dataset(dset_name, data=values, dtype="float32")

        # Log continental prediction maps to wandb (paper palette, interactive)
        if (
            getattr(self.trainer, "global_rank", 0) == 0
            and self.continent_map_configs
            and self.trainer.datamodule is not None
        ):
            try:
                from src.visualizations.maps import generate_wandb_maps
                images = generate_wandb_maps(
                    self.net, self.trainer.datamodule, self.continent_map_configs
                )
                if images and self.logger and hasattr(self.logger.experiment, "log"):
                    self.logger.experiment.log(images, step=self.global_step)
            except Exception as e:
                logger.warning(f"Failed to generate continent maps: {e}")


        gc.collect()
        try:
            import ctypes
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except Exception:
            pass

    def on_test_epoch_start(self):
        self.metric_logger.reset_metrics("test")
        self.test_all_probs = []
        self.test_all_targets = []
        self.test_all_locations = []
        self.test_all_areas = []
        self._test_areas_available = True

    def test_step(self, batch: Any, batch_idx: int):
        """Presence-absence test step: metrics plus accumulation of per-location
        probabilities/targets/locations for saving at epoch end."""
        logits, targets, range_mask = self.model_step(batch)

        observation_mask = torch.ones(logits.shape[0], dtype=torch.bool, device=logits.device)
        loss, masked_targets, observation_logits = self.compute_loss_and_masked_targets(
            logits, targets, observation_mask, range_mask
        )

        self.metric_logger.log_loss("test", loss)

        with torch.no_grad():
            probs = torch.sigmoid(observation_logits)
        self.metric_logger.update_metrics("test", probs, masked_targets)

        # Accumulate per-location predictions for saving
        self.test_all_probs.append(probs.detach().cpu())
        self.test_all_targets.append(masked_targets.detach().cpu())
        self.test_all_locations.append(batch["observation_locations"].detach().cpu())
        area = batch.get("sampled_area_m2")
        if area is not None:
            self.test_all_areas.append(area.detach().cpu())
        else:
            self._test_areas_available = False

        return {"loss": loss}

    def on_test_epoch_end(self):
        per_species_results = self.metric_logger.compute_and_log_epoch_metrics("test")

        # Concatenate all accumulated per-location data
        all_probs = torch.cat(self.test_all_probs, dim=0)       # (num_test_locs, num_species)
        all_targets = torch.cat(self.test_all_targets, dim=0)   # (num_test_locs, num_species)
        all_locations = torch.cat(self.test_all_locations, dim=0)  # (num_test_locs, 2)
        all_areas = (
            torch.cat(self.test_all_areas, dim=0).reshape(-1)  # (num_test_locs,)
            if self._test_areas_available and self.test_all_areas
            else None
        )

        torch.cuda.empty_cache()

        # Save per-species metrics and per-location predictions to HDF5.
        # Under fast_dev_run (or any run with no real logger) save_dir is None —
        # skip the on-disk artifacts (predictions / metrics / maps) that need a
        # run directory.
        save_dir = getattr(getattr(self.trainer, "logger", None), "save_dir", None)
        if getattr(self.trainer, "global_rank", 0) == 0 and save_dir is not None:
            out_dir = os.path.join(
                save_dir,
                self.trainer.logger.name,
                self.trainer.logger.version
            )
            os.makedirs(out_dir, exist_ok=True)

            # Save per-species metrics
            if per_species_results:
                metrics_h5_path = os.path.join(out_dir, "metrics_by_species.h5")
                mode = "a" if os.path.exists(metrics_h5_path) else "w"

                with h5py.File(metrics_h5_path, mode) as f:
                    for metric_name, values in per_species_results.items():
                        dset_name = f"test_{metric_name}"
                        if dset_name in f:
                            del f[dset_name]
                        f.create_dataset(dset_name, data=values, dtype="float32")

            # --- Save per-location prediction matrix ---
            preds_h5_path = os.path.join(out_dir, "test_predictions.h5")
            with h5py.File(preds_h5_path, "w") as f:
                f.create_dataset("probabilities", data=all_probs.numpy(), dtype="float32")
                f.create_dataset("targets", data=all_targets.numpy(), dtype="float32")
                f.create_dataset("locations", data=all_locations.numpy(), dtype="float32")
                if all_areas is not None:
                    f.create_dataset("sampled_area_m2", data=all_areas.numpy(), dtype="float32")
            logger.info(f"Saved test predictions {tuple(all_probs.shape)} to {preds_h5_path}")

            # Generate interactive continental maps (threshold slider) as HTML
            if self.continent_map_configs:
                try:
                    from src.visualizations.maps import generate_test_maps
                    generate_test_maps(
                        self.net, self.trainer.datamodule, self.continent_map_configs,
                        obs_locations=all_locations.numpy(),
                        obs_targets=all_targets.numpy(),
                        obs_probs=all_probs.numpy(),
                        output_dir=os.path.join(out_dir, "continent_maps"),
                    )
                except Exception as e:
                    logger.warning(f"Failed to generate continent maps: {e}")

        # Free memory
        self.test_all_probs = []
        self.test_all_targets = []
        self.test_all_locations = []
        self.test_all_areas = []

    def on_load_checkpoint(self, checkpoint):
        # Remove unexpected keys before loading
        state_dict = checkpoint.get('state_dict', {})
        keys_to_remove = [k for k in state_dict if 'loss.species_weight' in k]
        for k in keys_to_remove:
            del state_dict[k]


