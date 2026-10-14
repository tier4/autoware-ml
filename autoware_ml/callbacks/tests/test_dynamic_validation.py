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

"""Unit tests for the DynamicValidation callback: schedule validation, interval lookup, the
trainer precondition, and the per-epoch update of ``check_val_every_n_epoch``."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock

from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.demos.boring_classes import BoringModel
from lightning.pytorch.utilities.exceptions import MisconfigurationException

from autoware_ml.callbacks.dynamic_validation import DynamicValidation


class DynamicValidationTestCase(unittest.TestCase):
    """Shared fixtures for the DynamicValidation tests."""

    LOG_EVERY_N_STEPS = 50

    def setUp(self) -> None:
        """Build the example schedule and a trainer stand-in configured for epoch validation."""
        self.epoch_intervals = [[0, 5], [20, 1]]
        self.callback = DynamicValidation(self.epoch_intervals)
        self.trainer = self.make_trainer()
        self.module = MagicMock(spec=LightningModule)

    def make_trainer(
        self,
        current_epoch: int = 0,
        check_val_every_n_epoch: int | None = 5,
        val_check_interval: float | int = 1.0,
    ) -> MagicMock:
        """Build a stand-in Trainer carrying only the attributes the callback reads."""
        trainer = MagicMock(spec=Trainer)
        trainer.current_epoch = current_epoch
        trainer.check_val_every_n_epoch = check_val_every_n_epoch
        trainer.val_check_interval = val_check_interval
        trainer.log_every_n_steps = self.LOG_EVERY_N_STEPS
        trainer._val_check_time_interval = None
        return trainer


class TestEpochIntervalsValidation(DynamicValidationTestCase):
    """The schedule is checked at construction time."""

    def test_accepts_example_schedule(self) -> None:
        """
        Input: the schedule ``[[0, 5], [20, 1]]``.
        Expected: it is stored as ``(start_epoch, interval)`` tuples.
        Check: compare the normalized schedule.
        """
        self.assertEqual(self.callback.epoch_intervals, [(0, 5), (20, 1)])

    def test_accepts_single_segment(self) -> None:
        """
        Input: a schedule with one entry at epoch 0.
        Expected: a single segment is a valid, constant schedule.
        Check: the callback constructs and stores the one tuple.
        """
        callback = DynamicValidation([[0, 3]])

        self.assertEqual(callback.epoch_intervals, [(0, 3)])

    def test_rejects_empty_schedule(self) -> None:
        """
        Input: an empty list.
        Expected: at least one segment is required.
        Check: a ValueError mentioning "at least one" is raised.
        """
        with self.assertRaisesRegex(ValueError, "at least one"):
            DynamicValidation([])

    def test_rejects_schedule_not_starting_at_epoch_zero(self) -> None:
        """
        Input: a schedule whose first segment starts at epoch 5.
        Expected: epochs 0 to 4 would have no interval, so the schedule is rejected.
        Check: a ValueError mentioning "start at epoch 0" is raised.
        """
        with self.assertRaisesRegex(ValueError, "start at epoch 0"):
            DynamicValidation([[5, 1]])

    def test_rejects_non_increasing_start_epochs(self) -> None:
        """
        Input: two schedules, one with a repeated start epoch and one with a decreasing one.
        Expected: segment starts must strictly increase so the lookup is unambiguous.
        Check: both raise a ValueError mentioning "strictly increasing".
        """
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            DynamicValidation([[0, 5], [10, 2], [10, 1]])
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            DynamicValidation([[0, 5], [20, 2], [10, 1]])

    def test_rejects_interval_below_one(self) -> None:
        """
        Input: schedules with an interval of 0 and of -1.
        Expected: Lightning divides by ``check_val_every_n_epoch``, so it must be at least 1.
        Check: both raise a ValueError mentioning "at least 1".
        """
        with self.assertRaisesRegex(ValueError, "at least 1"):
            DynamicValidation([[0, 0]])
        with self.assertRaisesRegex(ValueError, "at least 1"):
            DynamicValidation([[0, 5], [20, -1]])

    def test_rejects_negative_start_epoch(self) -> None:
        """
        Input: a schedule with a negative start epoch after a valid first segment.
        Expected: epochs are zero-based, so negative starts are rejected.
        Check: a ValueError mentioning "non-negative" is raised.
        """
        with self.assertRaisesRegex(ValueError, "non-negative"):
            DynamicValidation([[0, 5], [-3, 1]])

    def test_rejects_malformed_entries(self) -> None:
        """
        Input: an entry with three values, one with a float, and one with a bool.
        Expected: every entry must be exactly two integers.
        Check: each raises a ValueError mentioning "pair of integers".
        """
        for bad_entry in ([0, 5, 1], [0, 2.5], [0, True]):
            with self.subTest(entry=bad_entry):
                with self.assertRaisesRegex(ValueError, "pair of integers"):
                    DynamicValidation([bad_entry])


class TestCheckValEveryNEpochAt(DynamicValidationTestCase):
    """The interval lookup by epoch."""

    def test_returns_first_segment_before_second_start(self) -> None:
        """
        Input: epochs 0, 1, 4 and 19 under ``[[0, 5], [20, 1]]``.
        Expected: every epoch before 20 uses the first interval of 5.
        Check: the lookup returns 5 for each.
        """
        for epoch in (0, 1, 4, 19):
            with self.subTest(epoch=epoch):
                self.assertEqual(self.callback.check_val_every_n_epoch_at(epoch), 5)

    def test_switches_at_segment_start_inclusive(self) -> None:
        """
        Input: epochs 20, 21 and 99 under ``[[0, 5], [20, 1]]``.
        Expected: the second segment applies from epoch 20 itself onwards, with no upper bound.
        Check: the lookup returns 1 for each.
        """
        for epoch in (20, 21, 99):
            with self.subTest(epoch=epoch):
                self.assertEqual(self.callback.check_val_every_n_epoch_at(epoch), 1)

    def test_three_segments(self) -> None:
        """
        Input: the schedule ``[[0, 10], [10, 5], [30, 1]]`` probed at each boundary.
        Expected: each epoch maps to the segment whose start is the largest one not above it.
        Check: compare the lookup at epochs 0, 9, 10, 29, 30 and 31.
        """
        callback = DynamicValidation([[0, 10], [10, 5], [30, 1]])

        expected = {0: 10, 9: 10, 10: 5, 29: 5, 30: 1, 31: 1}
        for epoch, interval in expected.items():
            with self.subTest(epoch=epoch):
                self.assertEqual(callback.check_val_every_n_epoch_at(epoch), interval)


class TestSetupPrecondition(DynamicValidationTestCase):
    """The trainer must validate at epoch ends for the schedule to make sense."""

    def test_accepts_epoch_based_validation(self) -> None:
        """
        Input: a trainer with ``check_val_every_n_epoch=5`` and ``val_check_interval=1.0``.
        Expected: this is the epoch-end configuration the callback supports.
        Check: ``setup`` returns without raising.
        """
        self.callback.setup(self.trainer, self.module, stage="fit")

    def test_rejects_check_val_every_n_epoch_none(self) -> None:
        """
        Input: a trainer with ``check_val_every_n_epoch=None``, meaning step-based validation.
        Expected: the epoch schedule cannot control step-based validation.
        Check: a MisconfigurationException mentioning ``check_val_every_n_epoch`` is raised.
        """
        trainer = self.make_trainer(check_val_every_n_epoch=None)

        with self.assertRaisesRegex(MisconfigurationException, "check_val_every_n_epoch"):
            self.callback.setup(trainer, self.module, stage="fit")

    def test_rejects_fractional_val_check_interval(self) -> None:
        """
        Input: a trainer with ``val_check_interval=0.5``, validating twice per epoch.
        Expected: validation within an epoch is outside the schedule's control.
        Check: a MisconfigurationException mentioning ``val_check_interval`` is raised.
        """
        trainer = self.make_trainer(val_check_interval=0.5)

        with self.assertRaisesRegex(MisconfigurationException, "val_check_interval"):
            self.callback.setup(trainer, self.module, stage="fit")

    def test_rejects_integer_val_check_interval(self) -> None:
        """
        Input: a trainer with the integer ``val_check_interval=1``, which means every batch.
        Expected: the integer form is batch-based even though ``1 == 1.0``, so it is rejected.
        Check: a MisconfigurationException mentioning ``val_check_interval`` is raised.
        """
        trainer = self.make_trainer(val_check_interval=1)

        with self.assertRaisesRegex(MisconfigurationException, "val_check_interval"):
            self.callback.setup(trainer, self.module, stage="fit")

    def test_rejects_time_based_validation(self) -> None:
        """
        Input: a trainer whose wall-time validation interval is set.
        Expected: time-based validation bypasses the epoch rule, so it is rejected.
        Check: a MisconfigurationException mentioning "wall-time" is raised.
        """
        trainer = self.make_trainer()
        trainer._val_check_time_interval = 60.0

        with self.assertRaisesRegex(MisconfigurationException, "wall-time"):
            self.callback.setup(trainer, self.module, stage="fit")

    def test_ignores_non_fit_stages(self) -> None:
        """
        Input: a step-based trainer, set up for the "test", "validate" and "predict" stages.
        Expected: the precondition only matters for training, so other stages pass.
        Check: ``setup`` returns without raising for each stage.
        """
        trainer = self.make_trainer(check_val_every_n_epoch=None)

        for stage in ("test", "validate", "predict"):
            with self.subTest(stage=stage):
                self.callback.setup(trainer, self.module, stage=stage)


class TestOnTrainEpochStart(DynamicValidationTestCase):
    """The per-epoch update of ``trainer.check_val_every_n_epoch``."""

    def test_keeps_first_interval_before_switch(self) -> None:
        """
        Input: the trainer at epoch 0 with ``check_val_every_n_epoch=5`` already set.
        Expected: the first segment matches, so the value is left at 5.
        Check: read ``check_val_every_n_epoch`` after the hook.
        """
        self.callback.on_train_epoch_start(self.trainer, self.module)

        self.assertEqual(self.trainer.check_val_every_n_epoch, 5)

    def test_switches_interval_at_segment_start(self) -> None:
        """
        Input: the trainer at epoch 20 still carrying the first interval of 5.
        Expected: the hook rewrites ``check_val_every_n_epoch`` to 1.
        Check: read the attribute after the hook.
        """
        trainer = self.make_trainer(current_epoch=20)

        self.callback.on_train_epoch_start(trainer, self.module)

        self.assertEqual(trainer.check_val_every_n_epoch, 1)

    def test_resume_mid_segment_applies_current_interval(self) -> None:
        """
        Input: a trainer restored at epoch 25 whose checkpoint carried ``check_val_every_n_epoch=5``.
        Expected: the hook applies the interval scheduled for epoch 25 regardless of the
        restored value, so resumed runs follow the schedule.
        Check: the attribute is 1 after the hook.
        """
        trainer = self.make_trainer(current_epoch=25, check_val_every_n_epoch=5)

        self.callback.on_train_epoch_start(trainer, self.module)

        self.assertEqual(trainer.check_val_every_n_epoch, 1)

    def test_logs_when_interval_changes(self) -> None:
        """
        Input: the trainer at epoch 20 with the first interval still set.
        Expected: the switch is reported at INFO level through the module logger.
        Check: capture the log and look for the new interval in the message.
        """
        trainer = self.make_trainer(current_epoch=20)

        with self.assertLogs("autoware_ml.callbacks.dynamic_validation", level="INFO") as logs:
            self.callback.on_train_epoch_start(trainer, self.module)

        self.assertTrue(any("every 1 epoch" in message for message in logs.output))

    def test_never_touches_log_every_n_steps(self) -> None:
        """
        Input: the trainer driven through ``setup`` and epoch starts 0, 19, 20 and 30.
        Expected: the callback rewrites only ``check_val_every_n_epoch``. Other callbacks read
        ``log_every_n_steps`` for their cadence, so it must be left exactly as configured.
        Check: ``log_every_n_steps`` still equals the configured value after every hook.
        """
        self.callback.setup(self.trainer, self.module, stage="fit")
        for epoch in (0, 19, 20, 30):
            self.trainer.current_epoch = epoch
            self.callback.on_train_epoch_start(self.trainer, self.module)

            self.assertEqual(self.trainer.log_every_n_steps, self.LOG_EVERY_N_STEPS)


class _ValidationEpochRecorder(Callback):
    """Record the training epoch at which every validation loop ran."""

    def __init__(self) -> None:
        super().__init__()
        self.epochs: list[int] = []

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if not trainer.sanity_checking:
            self.epochs.append(trainer.current_epoch)


class TestWithLightningTrainer(unittest.TestCase):
    """End-to-end check that Lightning honours the rewritten interval."""

    def test_validation_epochs_follow_schedule(self) -> None:
        """
        Input: a CPU trainer over 8 epochs of a BoringModel with the schedule
        ``[[0, 3], [5, 1]]`` and ``check_val_every_n_epoch=3`` as the initial value.
        Expected: Lightning validates after epoch 2 under the first segment, since
        ``(2 + 1) % 3 == 0``, skips epochs 3 and 4, then validates after every epoch from
        epoch 5 once the interval drops to 1, giving epochs ``[2, 5, 6, 7]``.
        Check: a recording callback collects the epoch of every validation loop.
        """
        recorder = _ValidationEpochRecorder()
        trainer = Trainer(
            max_epochs=8,
            limit_train_batches=2,
            limit_val_batches=1,
            num_sanity_val_steps=0,
            check_val_every_n_epoch=3,
            callbacks=[DynamicValidation([[0, 3], [5, 1]]), recorder],
            accelerator="cpu",
            devices=1,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
        )

        trainer.fit(BoringModel())

        self.assertEqual(recorder.epochs, [2, 5, 6, 7])


if __name__ == "__main__":
    unittest.main()
