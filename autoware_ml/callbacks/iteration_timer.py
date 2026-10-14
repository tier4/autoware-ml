# Copyright 2026 TIER IV, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Per-iteration wall-clock timing for the train, validation, and test loops."""

from __future__ import annotations

import statistics
import time
from typing import Any, NamedTuple

import torch
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback

from autoware_ml.dataclasses.models.model_batch_inputs import ModelBatchInputs
from autoware_ml.types.dataset import SplitType


class IterationTiming(NamedTuple):
    """One iteration's compute duration and the wall-clock cost it belongs to.

    Attributes:
        batch_forward_time: Seconds between the batch-start and batch-end hooks.
        total_iter_time: ``batch_forward_time`` plus the fetch that preceded it,
            or ``None`` for the first iteration of an epoch, which has no
            measurable fetch.
    """

    batch_forward_time: float
    total_iter_time: float | None


class _StageTimer:
    """Accumulate per-iteration timings for a single loop stage.

    Attributes:
        batch_forward_times: Iteration compute durations collected in the current epoch.
        data_times: Batch-fetch durations collected in the current epoch.
        total_times: Fetch plus compute durations collected in the current epoch.
    """

    def __init__(self) -> None:
        self.batch_forward_times: list[float] = []
        self.data_times: list[float] = []
        self.total_times: list[float] = []
        self._batch_start: float | None = None
        self._batch_end: float | None = None
        self._pending_data_time: float | None = None

    def reset(self) -> None:
        """Drop all timings and pending timestamps, ready for a new epoch."""
        self.batch_forward_times.clear()
        self.data_times.clear()
        self.total_times.clear()
        self._batch_start = None
        self._batch_end = None
        self._pending_data_time = None

    def start_batch(self, now: float) -> float | None:
        """Mark the beginning of an iteration.

        Args:
            now: Current timestamp, in seconds.

        Returns:
            Seconds spent fetching this batch, or ``None`` for the first
            iteration of an epoch, where there is no previous batch to
            measure against.
        """
        self._batch_start = now
        if self._batch_end is None:
            self._pending_data_time = None
            return None
        data_time = now - self._batch_end
        self.data_times.append(data_time)
        self._pending_data_time = data_time
        return data_time

    def end_batch(self, now: float) -> IterationTiming | None:
        """Mark the end of an iteration.

        Args:
            now: Current timestamp, in seconds.

        Returns:
            The iteration's timings, or ``None`` when no matching
            :meth:`start_batch` was recorded.
        """
        self._batch_end = now
        if self._batch_start is None:
            return None
        batch_forward_time = now - self._batch_start
        self._batch_start = None
        self.batch_forward_times.append(batch_forward_time)
        total_iter_time = None
        if self._pending_data_time is not None:
            total_iter_time = self._pending_data_time + batch_forward_time
            self.total_times.append(total_iter_time)
        self._pending_data_time = None
        return IterationTiming(
            batch_forward_time=batch_forward_time, total_iter_time=total_iter_time
        )


class IterationTimer(Callback):
    """Record how long each train, validation, and test iteration takes.

    Every training iteration contributes three step metrics:
    ``train/batch_forward_time`` (the time spent between the batch-start and
    batch-end hooks), ``train/data_waiting_time`` (the gap since the previous
    iteration ended, which is dominated by waiting on the dataloader), and
    ``train/total_iter_time`` (their sum, the wall-clock cost of the iteration).
    The validation and test loops contribute epoch summaries only.

    At the end of each epoch the callback logs, for every stage,
    ``{stage}/batch_forward_time_mean``, ``{stage}/batch_forward_time_max``,
    ``{stage}/data_waiting_time_mean``, ``{stage}/data_waiting_time_max``,
    ``{stage}/total_iter_time_mean`` and ``{stage}/total_iter_time_max``, all in
    seconds.

    The first iteration of an epoch has no measurable fetch, so it contributes
    neither ``data_waiting_time`` nor ``total_iter_time``.

    Because CUDA kernels are launched asynchronously, a batch-end hook can be
    reached while the GPU is still busy; ``sync_cuda`` inserts a device
    synchronization before each timestamp so the measurement reflects real
    device work, at the cost of removing CPU/GPU overlap.

    Args:
        sync_cuda: Synchronize CUDA before taking a timestamp. Timings become
            accurate per iteration, but the loop loses CPU/GPU overlap.
        warmup_iters: Number of leading iterations per epoch excluded from the
            epoch summary. The first iterations pay for dataloader worker
            startup and kernel autotuning and are not representative.
        log_interval: Log the per-iteration training metrics every this many
            iterations. The batch index drives the decision, so every rank logs
            the same iterations. The per-epoch summaries cover every iteration
            of every stage regardless.
    """

    def __init__(
        self,
        sync_cuda: bool = False,
        warmup_iters: int = 1,
        log_interval: int = 1,
    ) -> None:
        if warmup_iters < 0:
            raise ValueError(f"warmup_iters must be non-negative, got {warmup_iters}.")
        if log_interval < 1:
            raise ValueError(f"log_interval must be positive, got {log_interval}.")
        self.sync_cuda = sync_cuda
        self.warmup_iters = warmup_iters
        self.log_interval = log_interval
        self._timers = {
            stage.value: _StageTimer() for stage in (SplitType.TRAIN, SplitType.VAL, SplitType.TEST)
        }

    def _now(self) -> float:
        """Return the current time, optionally after draining CUDA work.

        Returns:
            Monotonic timestamp in seconds.
        """
        if self.sync_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()
        return time.perf_counter()

    def _batch_start(
        self, stage: str, trainer: Trainer, pl_module: LightningModule, batch_idx: int
    ) -> None:
        """Time the batch fetch and open a new iteration for ``stage``.

        Every iteration is timed so that the epoch summary stays complete. Only
        training iterations landing on ``log_interval`` are logged as a step
        metric; the class docstring explains why the other stages are excluded.

        Args:
            stage: Loop stage being timed.
            trainer: Active trainer.
            pl_module: Module used to log metrics.
            batch_idx: Index of the current batch.
        """
        if trainer.sanity_checking:
            return
        data_time = self._timers[stage].start_batch(self._now())
        if stage != SplitType.TRAIN.value or batch_idx % self.log_interval != 0:
            return
        if data_time is not None:
            self._log(pl_module, f"{stage}/data_waiting_time", data_time)

    def _batch_end(
        self, stage: str, trainer: Trainer, pl_module: LightningModule, batch_idx: int
    ) -> None:
        """Close the open iteration for ``stage`` and log its duration.

        Every iteration is timed so that the epoch summary stays complete. Only
        training iterations landing on ``log_interval`` are logged as a step
        metric; the class docstring explains why the other stages are excluded.

        Args:
            stage: Loop stage being timed.
            trainer: Active trainer.
            pl_module: Module used to log metrics.
            batch_idx: Index of the current batch.
        """
        if trainer.sanity_checking:
            return
        timer = self._timers[stage]
        timing = timer.end_batch(self._now())
        if timing is None or stage != SplitType.TRAIN.value or batch_idx % self.log_interval != 0:
            return
        self._log(pl_module, f"{stage}/batch_forward_time", timing.batch_forward_time)
        if timing.total_iter_time is not None:
            self._log(pl_module, f"{stage}/total_iter_time", timing.total_iter_time)

    def _epoch_end(self, stage: str, trainer: Trainer, pl_module: LightningModule) -> None:
        """Summarize the finished epoch for ``stage`` and reset its timer.

        Args:
            stage: Loop stage being timed.
            trainer: Active trainer.
            pl_module: Module used to log metrics.
        """
        if trainer.sanity_checking:
            self._timers[stage].reset()
            return
        timer = self._timers[stage]
        batch_forward_times = timer.batch_forward_times[self.warmup_iters :]
        data_times = timer.data_times[self.warmup_iters :]
        total_times = timer.total_times[self.warmup_iters :]
        if batch_forward_times:
            self._log(
                pl_module,
                f"{stage}/batch_forward_time_mean",
                statistics.fmean(batch_forward_times),
                on_step=False,
            )
            self._log(
                pl_module,
                f"{stage}/batch_forward_time_max",
                max(batch_forward_times),
                on_step=False,
            )
        if data_times:
            self._log(
                pl_module,
                f"{stage}/data_waiting_time_mean",
                statistics.fmean(data_times),
                on_step=False,
            )
            self._log(
                pl_module,
                f"{stage}/data_waiting_time_max",
                max(data_times),
                on_step=False,
            )
        if total_times:
            self._log(
                pl_module,
                f"{stage}/total_iter_time_mean",
                statistics.fmean(total_times),
                on_step=False,
            )
            self._log(pl_module, f"{stage}/total_iter_time_max", max(total_times), on_step=False)
        timer.reset()

    @staticmethod
    def _log(pl_module: LightningModule, name: str, value: float, on_step: bool = True) -> None:
        """Log a timing metric through the module's loggers.

        The fixed ``batch_size`` keeps Lightning from inferring a reduction
        weight from the batch: a timing is a property of the iteration, not of
        the samples in it.

        Args:
            pl_module: Module used to log metrics.
            name: Metric name.
            value: Metric value, in seconds.
            on_step: Log per step rather than per epoch.
        """
        pl_module.log(
            name,
            value,
            on_step=on_step,
            on_epoch=not on_step,
            prog_bar=False,
            batch_size=1,
        )

    def on_train_batch_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        batch: ModelBatchInputs,
        batch_idx: int,
    ) -> None:
        """Open the timing window for a training iteration."""
        self._batch_start(SplitType.TRAIN.value, trainer, pl_module, batch_idx)

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: ModelBatchInputs,
        batch_idx: int,
    ) -> None:
        """Close the timing window for a training iteration."""
        self._batch_end(SplitType.TRAIN.value, trainer, pl_module, batch_idx)

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Log the training epoch timing summary."""
        self._epoch_end(SplitType.TRAIN.value, trainer, pl_module)

    def on_validation_batch_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        batch: ModelBatchInputs,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Open the timing window for a validation iteration."""
        self._batch_start(SplitType.VAL.value, trainer, pl_module, batch_idx)

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: ModelBatchInputs,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Close the timing window for a validation iteration."""
        self._batch_end(SplitType.VAL.value, trainer, pl_module, batch_idx)

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Log the validation epoch timing summary."""
        self._epoch_end(SplitType.VAL.value, trainer, pl_module)

    def on_test_batch_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        batch: ModelBatchInputs,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Open the timing window for a test iteration."""
        self._batch_start(SplitType.TEST.value, trainer, pl_module, batch_idx)

    def on_test_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: ModelBatchInputs,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Close the timing window for a test iteration."""
        self._batch_end(SplitType.TEST.value, trainer, pl_module, batch_idx)

    def on_test_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Log the test epoch timing summary."""
        self._epoch_end(SplitType.TEST.value, trainer, pl_module)
