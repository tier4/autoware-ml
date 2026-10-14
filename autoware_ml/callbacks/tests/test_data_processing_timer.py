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

"""Unit tests for the data processing timing callback: step metric, epoch totals,
sanity-check suppression, and the logging interval."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock

from autoware_ml.callbacks.data_processing_timer import DataProcessingTimer


def _module() -> MagicMock:
    """Build a stand-in LightningModule that records ``log()`` calls."""
    module = MagicMock()
    module.log = MagicMock()
    return module


def _trainer(sanity_checking: bool = False) -> MagicMock:
    """Build a stand-in Trainer with the sanity-checking flag set."""
    trainer = MagicMock()
    trainer.sanity_checking = sanity_checking
    return trainer


def _batch(io_processing_time: float) -> MagicMock:
    """Build a stand-in for ModelBatchInputs carrying an IO timing."""
    batch = MagicMock(spec=["multi_task_gt_batch"])
    batch.multi_task_gt_batch.io_processing_time = io_processing_time
    return batch


def _logged(module: MagicMock) -> dict[str, float]:
    """Return the metric name to value mapping the callback logged."""
    return {call.args[0]: call.args[1] for call in module.log.call_args_list}


class TestStepMetric(unittest.TestCase):
    """The per-batch IO time is a step metric of the training loop only."""

    def test_batch_io_time_is_logged(self) -> None:
        """A training batch logs its IO time under the train-prefixed metric name."""
        callback = DataProcessingTimer()
        module = _module()

        callback.on_train_batch_start(_trainer(), module, batch=_batch(0.5), batch_idx=0)

        self.assertAlmostEqual(_logged(module)["train/data_processing_total_time"], 0.5)

    def test_evaluation_stages_log_no_step_metric(self) -> None:
        """Lightning re-emits eval step metrics every batch, so they are not logged."""
        hooks = (("val", "on_validation_batch_start"), ("test", "on_test_batch_start"))
        for stage, hook_name in hooks:
            with self.subTest(stage=stage):
                callback = DataProcessingTimer()
                module = _module()

                getattr(callback, hook_name)(_trainer(), module, batch=_batch(0.25), batch_idx=0)

                self.assertEqual(module.log.call_args_list, [])


class TestEpochSummary(unittest.TestCase):
    """The epoch hooks summarise and then reset the accumulated batch timings."""

    def test_epoch_mean_and_max(self) -> None:
        """The summary reports the per-batch mean and the slowest batch."""
        callback = DataProcessingTimer()
        module = _module()
        trainer = _trainer()

        for io_time in (1.0, 2.0, 3.0):
            callback.on_train_batch_start(trainer, module, batch=_batch(io_time), batch_idx=0)
        callback.on_train_epoch_end(trainer, module)

        logged = _logged(module)
        self.assertAlmostEqual(logged["train/data_processing_total_time_mean"], 2.0)
        self.assertAlmostEqual(logged["train/data_processing_total_time_max"], 3.0)

    def test_epoch_end_resets_the_accumulator(self) -> None:
        """Timings from a finished epoch do not leak into the next one."""
        callback = DataProcessingTimer()
        module = _module()
        trainer = _trainer()

        callback.on_train_batch_start(trainer, module, batch=_batch(10.0), batch_idx=0)
        callback.on_train_epoch_end(trainer, module)
        module.log.reset_mock()

        callback.on_train_batch_start(trainer, module, batch=_batch(1.0), batch_idx=0)
        callback.on_train_epoch_end(trainer, module)

        self.assertAlmostEqual(_logged(module)["train/data_processing_total_time_mean"], 1.0)

    def test_stages_accumulate_independently(self) -> None:
        """A validation epoch summary does not consume the training timings."""
        callback = DataProcessingTimer()
        module = _module()
        trainer = _trainer()

        callback.on_train_batch_start(trainer, module, batch=_batch(4.0), batch_idx=0)
        callback.on_validation_batch_start(trainer, module, batch=_batch(1.0), batch_idx=0)
        callback.on_validation_epoch_end(trainer, module)
        callback.on_train_epoch_end(trainer, module)

        logged = _logged(module)
        self.assertAlmostEqual(logged["val/data_processing_total_time_mean"], 1.0)
        self.assertAlmostEqual(logged["train/data_processing_total_time_mean"], 4.0)

    def test_summary_is_skipped_without_batches(self) -> None:
        """An epoch that recorded no batches logs nothing."""
        callback = DataProcessingTimer()
        module = _module()

        callback.on_test_epoch_end(_trainer(), module)

        self.assertEqual(module.log.call_args_list, [])


class TestSanityCheck(unittest.TestCase):
    """Sanity-check batches are excluded from the metrics."""

    def test_sanity_check_batches_are_not_recorded(self) -> None:
        """Neither the batch metric nor the epoch summary is logged while sanity checking."""
        callback = DataProcessingTimer()
        module = _module()
        sanity = _trainer(sanity_checking=True)

        callback.on_validation_batch_start(sanity, module, batch=_batch(1.0), batch_idx=0)
        callback.on_validation_epoch_end(sanity, module)

        self.assertEqual(module.log.call_args_list, [])


class TestLogInterval(unittest.TestCase):
    """The batch metric is logged every ``log_interval`` batches."""

    def test_only_matching_batches_are_logged(self) -> None:
        """With an interval of two, batch 1 contributes no metric."""
        callback = DataProcessingTimer(log_interval=2)
        trainer, module = _trainer(), _module()

        callback.on_train_batch_start(trainer, module, batch=_batch(0.5), batch_idx=1)

        self.assertEqual(module.log.call_args_list, [])

    def test_matching_batches_are_logged(self) -> None:
        """Batch 2 lands on an interval of two and is logged."""
        callback = DataProcessingTimer(log_interval=2)
        trainer, module = _trainer(), _module()

        callback.on_train_batch_start(trainer, module, batch=_batch(0.5), batch_idx=2)

        self.assertAlmostEqual(_logged(module)["train/data_processing_total_time"], 0.5)

    def test_skipped_batches_still_reach_the_epoch_summary(self) -> None:
        """Accumulation is independent of the logging interval."""
        callback = DataProcessingTimer(log_interval=10)
        trainer, module = _trainer(), _module()

        for batch_idx, io_time in enumerate((1.0, 2.0, 3.0)):
            callback.on_train_batch_start(
                trainer, module, batch=_batch(io_time), batch_idx=batch_idx
            )
        callback.on_train_epoch_end(trainer, module)

        self.assertAlmostEqual(_logged(module)["train/data_processing_total_time_mean"], 2.0)

    def test_non_positive_interval_is_rejected(self) -> None:
        """A zero interval would divide by zero, so it is refused."""
        with self.assertRaisesRegex(ValueError, "log_interval"):
            DataProcessingTimer(log_interval=0)


if __name__ == "__main__":
    unittest.main()
