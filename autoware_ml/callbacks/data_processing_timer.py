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

"""Sample loading and transform timing carried on the batch by the dataloader."""

from __future__ import annotations

import statistics
from typing import Any

from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback

from autoware_ml.dataclasses.models.model_batch_inputs import ModelBatchInputs
from autoware_ml.types.dataset import SplitType

METRIC_NAME = "data_processing_total_time"


class DataProcessingTimer(Callback):
    """Record the sample loading and transform time carried on each batch.

    ``BaseDataset`` measures loading plus transform time per sample
    inside the dataloader worker, and the collate function sums those per-sample
    values into ``ModelGTBatch.io_processing_time``. This callback reads that
    field and logs it, which is why the timing lives on the batch at all:
    Lightning forbids ``self.log()`` inside ``on_after_batch_transfer()``, where
    the batch first becomes available to the module.

    Because the value is a sum over the samples collated into one batch, it is
    logged as ``train/data_processing_total_time`` (in seconds). With
    ``num_workers > 0`` this time overlaps with compute, so it is a measure of IO
    work performed, not of time the training loop was blocked. Compare it
    against ``train/data_waiting_time`` from
    :class:`~autoware_ml.callbacks.iteration_timer.IterationTimer` to see how
    much of it the loop actually waited for.

    The step metric is logged on every training batch and Lightning writes it
    out every ``log_every_n_steps`` of the trainer. Logging only on a
    callback-side interval would leave Lightning flushing stale values whenever
    the epoch length is not a multiple of that interval.

    The validation and test loops contribute epoch summaries only.
    At the end of each epoch the per-batch mean and the slowest batch are
    logged, for every stage, as ``{stage}/data_processing_total_time_mean`` and
    ``{stage}/data_processing_total_time_max``.
    """

    def __init__(self) -> None:
        self._batch_times: dict[str, list[float]] = {
            stage: []
            for stage in (SplitType.TRAIN.value, SplitType.VAL.value, SplitType.TEST.value)
        }

    def _record_batch(
        self,
        stage: str,
        trainer: Trainer,
        pl_module: LightningModule,
        batch: ModelBatchInputs,
    ) -> None:
        """Accumulate one batch's IO processing time and log it.

        Every batch is accumulated so that the epoch summary stays complete.
        Only training batches are logged as a step metric. The class docstring
        explains why the other stages are excluded.

        Args:
            stage: Loop stage being recorded.
            trainer: Active trainer.
            pl_module: Module used to log metrics.
            batch: Batch handed to the loop after transfer to the device.
        """
        if trainer.sanity_checking:
            return
        io_time = batch.multi_task_gt_batch.io_processing_time
        if io_time is None:
            return
        batch_times = self._batch_times[stage]
        batch_times.append(io_time)
        if stage == SplitType.TRAIN.value:
            self._log(pl_module, f"{stage}/{METRIC_NAME}", io_time)

    def _record_epoch(self, stage: str, trainer: Trainer, pl_module: LightningModule) -> None:
        """Log the per-batch mean and slowest batch, then reset the stage.

        Args:
            stage: Loop stage being recorded.
            trainer: Active trainer.
            pl_module: Module used to log metrics.
        """
        batch_times = self._batch_times[stage]
        if trainer.sanity_checking:
            batch_times.clear()
            return
        if batch_times:
            mean = statistics.fmean(batch_times)
            max_batch_time = max(batch_times)
            self._log(pl_module, f"{stage}/{METRIC_NAME}_mean", mean, on_step=False)
            self._log(pl_module, f"{stage}/{METRIC_NAME}_max", max_batch_time, on_step=False)
        batch_times.clear()

    @staticmethod
    def _log(pl_module: LightningModule, name: str, value: float, on_step: bool = True) -> None:
        """Log an IO timing metric through the module's loggers.

        The fixed ``batch_size`` keeps Lightning from inferring a reduction
        weight from the batch: the timing is a property of the batch as a whole,
        not of the samples in it. Only one of ``on_step``/``on_epoch`` is ever
        set, because enabling both makes Lightning suffix the key with
        ``_step``/``_epoch``.

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
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Record the IO processing time of a training batch."""
        self._record_batch(SplitType.TRAIN.value, trainer, pl_module, batch)

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Log the training epoch IO processing summary."""
        self._record_epoch(SplitType.TRAIN.value, trainer, pl_module)

    def on_validation_batch_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Record the IO processing time of a validation batch."""
        self._record_batch(SplitType.VAL.value, trainer, pl_module, batch)

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Log the validation epoch IO processing summary."""
        self._record_epoch(SplitType.VAL.value, trainer, pl_module)

    def on_test_batch_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Record the IO processing time of a test batch."""
        self._record_batch(SplitType.TEST.value, trainer, pl_module, batch)

    def on_test_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Log the test epoch IO processing summary."""
        self._record_epoch(SplitType.TEST.value, trainer, pl_module)
