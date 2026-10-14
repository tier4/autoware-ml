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

"""Epoch-dependent validation frequency."""

from __future__ import annotations

import logging
from typing import Sequence

from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities.exceptions import MisconfigurationException

logger = logging.getLogger(__name__)


class DynamicValidation(Callback):
    """Change how often validation runs as training progresses.

    Lightning validates every ``check_val_every_n_epoch`` epochs and reads that trainer
    attribute afresh at every epoch, so this callback rewrites it at the start of each
    training epoch according to a schedule. A typical use is to validate sparsely while the
    loss is still falling quickly and every epoch near the end of training, when the
    checkpoint selection needs the finer resolution.

    The schedule is a sequence of ``[start_epoch, check_val_every_n_epoch]`` pairs. Each pair
    applies from its start epoch (zero-based, inclusive) until the next pair's start epoch.
    ``[[0, 5], [20, 1]]`` therefore validates every 5 epochs during epochs 0 to 19 and every
    epoch from epoch 20 onwards. Within a segment Lightning keeps its own rule, validating
    after epoch ``e`` when ``(e + 1) % check_val_every_n_epoch == 0``, so the epochs at which
    validation runs are counted from epoch 0, not from the segment start.

    The callback only touches ``trainer.check_val_every_n_epoch``. In particular it never
    modifies ``trainer.log_every_n_steps``, which other callbacks read for their own logging
    cadence. It requires validation to be scheduled per epoch: ``val_check_interval`` must be
    ``1.0`` and ``check_val_every_n_epoch`` must not be ``None``, otherwise Lightning would
    validate on a step or time basis and the epoch schedule would be meaningless.

    Args:
        epoch_intervals: ``[start_epoch, check_val_every_n_epoch]`` pairs, sorted by strictly
            increasing start epoch. The first start epoch must be 0 so that every epoch is
            covered, and every interval must be at least 1.

    Raises:
        ValueError: If the schedule is empty, malformed, does not start at epoch 0, is not
            strictly increasing, or contains an interval below 1.
    """

    def __init__(self, epoch_intervals: Sequence[Sequence[int]]) -> None:
        super().__init__()
        self.epoch_intervals: list[tuple[int, int]] = self._validate_epoch_intervals(
            epoch_intervals
        )

    @staticmethod
    def _validate_epoch_intervals(
        epoch_intervals: Sequence[Sequence[int]],
    ) -> list[tuple[int, int]]:
        """Normalize the schedule to ``(start_epoch, interval)`` tuples and check its shape.

        Args:
            epoch_intervals: The schedule as passed to the constructor.

        Returns:
            The schedule as a list of ``(start_epoch, interval)`` tuples.

        Raises:
            ValueError: See the class docstring.
        """
        if len(epoch_intervals) == 0:
            raise ValueError("epoch_intervals must contain at least one [start_epoch, interval].")

        normalized: list[tuple[int, int]] = []
        for entry in epoch_intervals:
            if len(entry) != 2 or any(isinstance(v, bool) or not isinstance(v, int) for v in entry):
                raise ValueError(
                    "Every entry of epoch_intervals must be a [start_epoch, interval] pair of "
                    f"integers, got {list(entry)!r}."
                )
            start_epoch, interval = int(entry[0]), int(entry[1])
            if start_epoch < 0:
                raise ValueError(f"start_epoch must be non-negative, got {start_epoch}.")
            if interval < 1:
                raise ValueError(
                    f"check_val_every_n_epoch must be at least 1, got {interval} for epoch "
                    f"{start_epoch}."
                )
            normalized.append((start_epoch, interval))

        if normalized[0][0] != 0:
            raise ValueError(
                "The first entry of epoch_intervals must start at epoch 0 so that every epoch "
                f"has an interval, got start_epoch {normalized[0][0]}."
            )
        starts = [start for start, _ in normalized]
        if any(later <= earlier for earlier, later in zip(starts, starts[1:])):
            raise ValueError(
                f"epoch_intervals must have strictly increasing start epochs, got {starts}."
            )
        return normalized

    def check_val_every_n_epoch_at(self, epoch: int) -> int:
        """Return the validation interval in force at ``epoch``.

        Args:
            epoch: Zero-based training epoch.

        Returns:
            The ``check_val_every_n_epoch`` of the last schedule entry whose start epoch is at
            or before ``epoch``.
        """
        interval = self.epoch_intervals[0][1]
        for start_epoch, segment_interval in self.epoch_intervals:
            if start_epoch > epoch:
                break
            interval = segment_interval
        return interval

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        """Check that the trainer validates on an epoch basis before training starts.

        Args:
            trainer: The trainer the callback is attached to.
            pl_module: The module being trained. Unused.
            stage: The running stage. Only ``"fit"`` is checked.

        Raises:
            MisconfigurationException: If validation is driven by steps or wall time instead
                of epochs, which the epoch schedule cannot control.
        """
        if stage != "fit":
            return
        if trainer.check_val_every_n_epoch is None:
            raise MisconfigurationException(
                "DynamicValidation schedules validation per epoch, but "
                "`Trainer(check_val_every_n_epoch=None)` validates every `val_check_interval` "
                "training batches. Set `check_val_every_n_epoch` to an integer."
            )
        val_check_interval = trainer.val_check_interval
        if not (isinstance(val_check_interval, float) and val_check_interval == 1.0):
            raise MisconfigurationException(
                "DynamicValidation schedules validation per epoch, but "
                f"`Trainer(val_check_interval={val_check_interval!r})` validates within an "
                "epoch. Set `val_check_interval` to 1.0 so that validation runs at epoch ends."
            )
        if getattr(trainer, "_val_check_time_interval", None) is not None:
            raise MisconfigurationException(
                "DynamicValidation schedules validation per epoch, but the trainer validates on "
                "a wall-time interval. Use a float or integer `val_check_interval` of 1.0."
            )

    def on_train_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Apply the interval scheduled for the epoch that is about to run.

        Args:
            trainer: The trainer whose ``check_val_every_n_epoch`` is updated.
            pl_module: The module being trained. Unused.
        """
        interval = self.check_val_every_n_epoch_at(trainer.current_epoch)
        if trainer.check_val_every_n_epoch != interval:
            logger.info(
                "DynamicValidation: from epoch %d validation runs every %d epoch(s) "
                "(was every %s).",
                trainer.current_epoch,
                interval,
                trainer.check_val_every_n_epoch,
            )
        trainer.check_val_every_n_epoch = interval
