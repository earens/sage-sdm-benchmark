"""Lightning callback that logs per-epoch time, total training time, and peak GPU memory to WandB."""

import time
import torch
import pytorch_lightning as pl


class TimingCallback(pl.Callback):
    """Logs per-epoch wall-clock time, total training time, and peak GPU memory to WandB."""

    def __init__(self):
        # Initialize here so standalone validate/test runs (no on_fit_start) still
        # have these attributes; on_validation_epoch_end early-returns when None.
        self._fit_start = None
        self._train_epoch_start = None
        self._epoch_elapsed = None
        self._epoch_peak_gpu_gb = None

    def on_fit_start(self, trainer, pl_module):
        self._fit_start = time.perf_counter()
        self._epoch_elapsed = None
        self._epoch_peak_gpu_gb = None

    def on_train_epoch_start(self, trainer, pl_module):
        self._train_epoch_start = time.perf_counter()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def on_train_epoch_end(self, trainer, pl_module):
        # Store values; log after validation so WandB step is aligned
        self._epoch_elapsed = time.perf_counter() - self._train_epoch_start
        if torch.cuda.is_available():
            self._epoch_peak_gpu_gb = torch.cuda.max_memory_allocated() / 1e9

    def on_validation_epoch_end(self, trainer, pl_module):
        if self._epoch_elapsed is None:
            return
        exp = getattr(getattr(trainer, "logger", None), "experiment", None)
        if exp is None:
            return
        metrics = {
            "perf/train_epoch_time_s": self._epoch_elapsed,
            "trainer/global_step": trainer.global_step,
        }
        if self._epoch_peak_gpu_gb is not None:
            metrics["perf/gpu_peak_memory_gb"] = self._epoch_peak_gpu_gb
        try:
            exp.log(metrics)
        except Exception:
            pass
        self._epoch_elapsed = None
        self._epoch_peak_gpu_gb = None

    def on_fit_end(self, trainer, pl_module):
        total_s = time.perf_counter() - self._fit_start
        exp = getattr(getattr(trainer, "logger", None), "experiment", None)
        if exp is not None and hasattr(exp, "summary"):
            try:
                exp.summary["perf/total_train_time_s"] = total_s
                exp.summary["perf/total_train_time_h"] = total_s / 3600
            except (TypeError, AttributeError):
                pass  # non-wandb / dummy logger (e.g. under fast_dev_run)
