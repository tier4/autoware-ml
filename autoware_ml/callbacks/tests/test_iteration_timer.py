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

"""Unit tests for the iteration timing callback: step metrics, accumulation over batches,
warmup exclusion, the logging interval, sanity-check suppression, and CUDA synchronization."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
import unittest
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import torch

from autoware_ml.callbacks.iteration_timer import IterationTimer


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


@contextmanager
def _clock(timestamps: Sequence[float]) -> Iterator[MagicMock]:
    """Replace the wall clock with one returning ``timestamps`` in order.

    The callback reads the clock exactly once per batch hook, so a test lists one timestamp
    per hook it drives. An extra read raises ``StopIteration`` and fails the test.
    """
    with patch("time.perf_counter", side_effect=list(timestamps)) as clock:
        yield clock


def _logged(module: MagicMock) -> dict[str, float]:
    """Return the metric name to last value mapping the callback logged."""
    return {call.args[0]: call.args[1] for call in module.log.call_args_list}


def _logged_values(module: MagicMock, name: str) -> list[float]:
    """Return every value the callback logged under ``name``, in order."""
    return [call.args[1] for call in module.log.call_args_list if call.args[0] == name]


def _run_train_batch(
    callback: IterationTimer, trainer: MagicMock, module: MagicMock, batch_idx: int
) -> None:
    """Drive the batch-start and batch-end hooks of one training iteration."""
    callback.on_train_batch_start(trainer, module, batch=MagicMock(), batch_idx=batch_idx)
    callback.on_train_batch_end(
        trainer, module, outputs=None, batch=MagicMock(), batch_idx=batch_idx
    )


class TestStepMetrics(unittest.TestCase):
    """The per-iteration timings are step metrics of the training loop only."""

    def test_first_iteration_logs_the_forward_time_only(self) -> None:
        """With no previous batch there is no fetch to measure, so only the forward time appears."""
        callback = IterationTimer()
        trainer, module = _trainer(), _module()

        with _clock([0.0, 0.5]):
            _run_train_batch(callback, trainer, module, batch_idx=0)

        logged = _logged(module)
        self.assertAlmostEqual(logged["train/batch_forward_time"], 0.5)
        self.assertNotIn("train/data_waiting_time", logged)
        self.assertNotIn("train/total_iter_time", logged)

    def test_later_iterations_log_the_fetch_and_the_total_time(self) -> None:
        """The gap since the previous batch ended is the fetch. Fetch plus forward is the total."""
        callback = IterationTimer()
        trainer, module = _trainer(), _module()

        with _clock([0.0, 0.5, 1.5, 2.0]):
            _run_train_batch(callback, trainer, module, batch_idx=0)
            _run_train_batch(callback, trainer, module, batch_idx=1)

        logged = _logged(module)
        self.assertAlmostEqual(logged["train/data_waiting_time"], 1.0)
        self.assertAlmostEqual(logged["train/batch_forward_time"], 0.5)
        self.assertAlmostEqual(logged["train/total_iter_time"], 1.5)

    def test_evaluation_stages_log_no_step_metric(self) -> None:
        """Lightning re-emits eval step metrics every batch, so they are not logged."""
        hooks = (
            ("val", "on_validation_batch_start", "on_validation_batch_end"),
            ("test", "on_test_batch_start", "on_test_batch_end"),
        )
        for stage, start_hook, end_hook in hooks:
            with self.subTest(stage=stage):
                callback = IterationTimer()
                trainer, module = _trainer(), _module()

                with _clock([0.0, 0.5, 1.5, 2.0]):
                    for batch_idx in (0, 1):
                        getattr(callback, start_hook)(
                            trainer, module, batch=MagicMock(), batch_idx=batch_idx
                        )
                        getattr(callback, end_hook)(
                            trainer, module, outputs=None, batch=MagicMock(), batch_idx=batch_idx
                        )

                self.assertEqual(module.log.call_args_list, [])

    def test_batch_end_without_a_start_logs_nothing(self) -> None:
        """A batch-end hook with no open iteration has nothing to measure."""
        callback = IterationTimer()
        trainer, module = _trainer(), _module()

        with _clock([1.0]):
            callback.on_train_batch_end(
                trainer, module, outputs=None, batch=MagicMock(), batch_idx=0
            )

        self.assertEqual(module.log.call_args_list, [])

    def test_step_metrics_are_logged_per_step_with_a_unit_batch_size(self) -> None:
        """A timing is a property of the iteration, so it is logged per step with batch size one."""
        callback = IterationTimer()
        trainer, module = _trainer(), _module()

        with _clock([0.0, 0.5]):
            _run_train_batch(callback, trainer, module, batch_idx=0)

        kwargs = module.log.call_args.kwargs
        self.assertTrue(kwargs["on_step"])
        self.assertFalse(kwargs["on_epoch"])
        self.assertFalse(kwargs["prog_bar"])
        self.assertEqual(kwargs["batch_size"], 1)


# One epoch of three iterations: forward times 1, 2 and 3 seconds, with fetch gaps of 1 and 3
# seconds between them. The first iteration has no fetch, so it contributes no total time.
_THREE_BATCH_CLOCK = (0.0, 1.0, 2.0, 4.0, 7.0, 10.0)


class TestEpochSummary(unittest.TestCase):
    """The epoch hooks summarise the accumulated iteration timings and then reset them."""

    def test_timings_accumulate_over_the_batches_of_an_epoch(self) -> None:
        """The summary reports the mean and the slowest of every timing kind over the epoch."""
        callback = IterationTimer(warmup_iters=0)
        trainer, module = _trainer(), _module()

        with _clock(_THREE_BATCH_CLOCK):
            for batch_idx in range(3):
                _run_train_batch(callback, trainer, module, batch_idx=batch_idx)
        callback.on_train_epoch_end(trainer, module)

        logged = _logged(module)
        self.assertAlmostEqual(logged["train/batch_forward_time_mean"], 2.0)
        self.assertAlmostEqual(logged["train/batch_forward_time_max"], 3.0)
        self.assertAlmostEqual(logged["train/data_waiting_time_mean"], 2.0)
        self.assertAlmostEqual(logged["train/data_waiting_time_max"], 3.0)
        self.assertAlmostEqual(logged["train/total_iter_time_mean"], 4.5)
        self.assertAlmostEqual(logged["train/total_iter_time_max"], 6.0)

    def test_warmup_iterations_are_excluded_from_the_summary(self) -> None:
        """The leading ``warmup_iters`` entries of every timing kind are dropped."""
        callback = IterationTimer(warmup_iters=1)
        trainer, module = _trainer(), _module()

        with _clock(_THREE_BATCH_CLOCK):
            for batch_idx in range(3):
                _run_train_batch(callback, trainer, module, batch_idx=batch_idx)
        callback.on_train_epoch_end(trainer, module)

        logged = _logged(module)
        self.assertAlmostEqual(logged["train/batch_forward_time_mean"], 2.5)
        self.assertAlmostEqual(logged["train/batch_forward_time_max"], 3.0)
        self.assertAlmostEqual(logged["train/data_waiting_time_mean"], 3.0)
        self.assertAlmostEqual(logged["train/data_waiting_time_max"], 3.0)
        self.assertAlmostEqual(logged["train/total_iter_time_mean"], 6.0)
        self.assertAlmostEqual(logged["train/total_iter_time_max"], 6.0)

    def test_warmup_does_not_affect_the_step_metrics(self) -> None:
        """Warmup only trims the summary. Every iteration still logs its own step metric."""
        callback = IterationTimer(warmup_iters=2)
        trainer, module = _trainer(), _module()

        with _clock(_THREE_BATCH_CLOCK):
            for batch_idx in range(3):
                _run_train_batch(callback, trainer, module, batch_idx=batch_idx)

        self.assertEqual(_logged_values(module, "train/batch_forward_time"), [1.0, 2.0, 3.0])

    def test_summary_is_skipped_when_warmup_covers_the_epoch(self) -> None:
        """An epoch no longer than the warmup leaves nothing to summarise."""
        callback = IterationTimer(warmup_iters=3)
        trainer, module = _trainer(), _module()

        with _clock(_THREE_BATCH_CLOCK):
            for batch_idx in range(3):
                _run_train_batch(callback, trainer, module, batch_idx=batch_idx)
        module.log.reset_mock()
        callback.on_train_epoch_end(trainer, module)

        self.assertEqual(module.log.call_args_list, [])

    def test_epoch_end_resets_the_accumulator(self) -> None:
        """Timings from a finished epoch do not leak into the next one, nor does its end time."""
        callback = IterationTimer(warmup_iters=0)
        trainer, module = _trainer(), _module()

        with _clock([0.0, 10.0]):
            _run_train_batch(callback, trainer, module, batch_idx=0)
        callback.on_train_epoch_end(trainer, module)
        module.log.reset_mock()

        with _clock([20.0, 22.0]):
            _run_train_batch(callback, trainer, module, batch_idx=0)
        callback.on_train_epoch_end(trainer, module)

        logged = _logged(module)
        self.assertAlmostEqual(logged["train/batch_forward_time_mean"], 2.0)
        self.assertNotIn("train/data_waiting_time", logged)
        self.assertNotIn("train/data_waiting_time_mean", logged)
        self.assertNotIn("train/total_iter_time_mean", logged)

    def test_stages_accumulate_independently(self) -> None:
        """A validation epoch summary does not consume the training timings."""
        callback = IterationTimer(warmup_iters=0)
        trainer, module = _trainer(), _module()

        with _clock([0.0, 4.0]):
            _run_train_batch(callback, trainer, module, batch_idx=0)
        with _clock([10.0, 11.0]):
            callback.on_validation_batch_start(trainer, module, batch=MagicMock(), batch_idx=0)
            callback.on_validation_batch_end(
                trainer, module, outputs=None, batch=MagicMock(), batch_idx=0
            )
        callback.on_validation_epoch_end(trainer, module)
        callback.on_train_epoch_end(trainer, module)

        logged = _logged(module)
        self.assertAlmostEqual(logged["val/batch_forward_time_mean"], 1.0)
        self.assertAlmostEqual(logged["train/batch_forward_time_mean"], 4.0)

    def test_summary_is_skipped_without_batches(self) -> None:
        """An epoch that recorded no batches logs nothing."""
        callback = IterationTimer()
        module = _module()

        callback.on_test_epoch_end(_trainer(), module)

        self.assertEqual(module.log.call_args_list, [])

    def test_summary_metrics_are_logged_per_epoch(self) -> None:
        """The summary is an epoch metric, not a step metric."""
        callback = IterationTimer(warmup_iters=0)
        trainer, module = _trainer(), _module()

        with _clock([0.0, 1.0]):
            _run_train_batch(callback, trainer, module, batch_idx=0)
        module.log.reset_mock()
        callback.on_train_epoch_end(trainer, module)

        kwargs = module.log.call_args.kwargs
        self.assertFalse(kwargs["on_step"])
        self.assertTrue(kwargs["on_epoch"])
        self.assertEqual(kwargs["batch_size"], 1)


class TestSanityCheck(unittest.TestCase):
    """Sanity-check batches are excluded from the metrics."""

    def test_sanity_check_batches_are_not_timed(self) -> None:
        """Neither the clock nor the logger is touched while sanity checking."""
        callback = IterationTimer(warmup_iters=0)
        module = _module()
        sanity = _trainer(sanity_checking=True)

        with _clock([]):
            callback.on_validation_batch_start(sanity, module, batch=MagicMock(), batch_idx=0)
            callback.on_validation_batch_end(
                sanity, module, outputs=None, batch=MagicMock(), batch_idx=0
            )
            callback.on_validation_epoch_end(sanity, module)

        self.assertEqual(module.log.call_args_list, [])

    def test_sanity_check_epoch_end_drops_recorded_timings(self) -> None:
        """A sanity-check epoch end resets the stage so stale timings are not summarised later."""
        callback = IterationTimer(warmup_iters=0)
        module = _module()

        with _clock([0.0, 1.0]):
            callback.on_validation_batch_start(_trainer(), module, batch=MagicMock(), batch_idx=0)
            callback.on_validation_batch_end(
                _trainer(), module, outputs=None, batch=MagicMock(), batch_idx=0
            )
        callback.on_validation_epoch_end(_trainer(sanity_checking=True), module)
        callback.on_validation_epoch_end(_trainer(), module)

        self.assertEqual(module.log.call_args_list, [])


class TestConstructorValidation(unittest.TestCase):
    """Invalid timer settings are refused at construction."""

    def test_negative_warmup_is_rejected(self) -> None:
        """A negative warmup would slice from the end of the timings, so it is refused."""
        with self.assertRaisesRegex(ValueError, "warmup_iters"):
            IterationTimer(warmup_iters=-1)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA synchronization needs a CUDA device")
class TestSyncCuda(unittest.TestCase):
    """``sync_cuda`` drains the device before every timestamp."""

    def test_each_timestamp_follows_a_device_synchronization(self) -> None:
        """One training iteration synchronizes twice, once per hook, and still logs its timing."""
        callback = IterationTimer(sync_cuda=True)
        trainer, module = _trainer(), _module()

        with patch.object(torch.cuda, "synchronize", wraps=torch.cuda.synchronize) as sync:
            _run_train_batch(callback, trainer, module, batch_idx=0)

        self.assertEqual(sync.call_count, 2)
        self.assertGreaterEqual(_logged(module)["train/batch_forward_time"], 0.0)

    def test_synchronization_is_off_by_default(self) -> None:
        """Without ``sync_cuda`` the timestamps are taken without draining the device."""
        callback = IterationTimer()
        trainer, module = _trainer(), _module()

        with patch.object(torch.cuda, "synchronize", wraps=torch.cuda.synchronize) as sync:
            _run_train_batch(callback, trainer, module, batch_idx=0)

        self.assertEqual(sync.call_count, 0)
        self.assertIn("train/batch_forward_time", _logged(module))


if __name__ == "__main__":
    unittest.main()
