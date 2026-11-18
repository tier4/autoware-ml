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

"""Save a checkpoint and stop training when the process receives a signal."""

import os
from pathlib import Path
import signal
from types import FrameType
from typing import Any

from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities.rank_zero import rank_zero_info

CHECKPOINT_FILENAME = "stopped.ckpt"


class TrainingStopRequested(Exception):
    """Raised on every rank once ``StopOnSignal`` has saved its checkpoint."""

    def __init__(self, checkpoint_path: Path) -> None:
        """Initialize the exception.

        Args:
            checkpoint_path: Checkpoint the training resumes from.
        """
        super().__init__(f"Training stopped on request, checkpoint saved to {checkpoint_path}.")
        self.checkpoint_path = checkpoint_path


class StopOnSignal(Callback):
    """Save ``stopped.ckpt`` next to the other checkpoints and stop when a signal arrives.

    The handler only records the signal, so the forked dataloader workers that inherit it and
    receive the signal too do nothing. Every ``check_every_n_steps`` optimizer steps the ranks
    agree whether any of them received it. If one did, they save the checkpoint together and
    raise ``TrainingStopRequested``, which skips the validation pass Lightning runs before it
    honors ``trainer.should_stop``. The handler replaces the one Lightning registers for its
    auto-requeue, and Lightning's teardown restores the handler from before fit.
    """

    def __init__(self, signal_name: str = "SIGUSR1", check_every_n_steps: int = 50) -> None:
        """Initialize the callback.

        Args:
            signal_name: Signal that requests the stop. ``SIGINT`` and ``SIGTERM`` are rejected,
                Lightning stops on them before a checkpoint could be saved.
            check_every_n_steps: Optimizer steps between checks, each one an all-reduce.

        Raises:
            ValueError: If the signal is ``SIGINT`` or ``SIGTERM``, or the interval is below 1.
        """
        super().__init__()
        self.signum = signal.Signals[signal_name]
        if self.signum in (signal.SIGINT, signal.SIGTERM):
            raise ValueError(f"{signal_name} is handled by Lightning, use SIGUSR1 or SIGUSR2.")
        if check_every_n_steps < 1:
            raise ValueError(f"check_every_n_steps must be at least 1, got {check_every_n_steps}.")
        self.check_every_n_steps = check_every_n_steps
        self._requested = False
        self._owner_pid: int | None = None
        self._checked_step: int | None = None

    def _handle(self, signum: int, frame: FrameType | None) -> None:
        if os.getpid() == self._owner_pid:
            self._requested = True

    def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Install the signal handler after Lightning registered its own."""
        self._requested = False
        self._checked_step = None
        self._owner_pid = os.getpid()
        signal.signal(self.signum, self._handle)

    def on_train_batch_end(
        self, trainer: Trainer, pl_module: LightningModule, outputs: Any, batch: Any, batch_idx: int
    ) -> None:
        """Save the checkpoint and stop once any rank received the signal.

        Raises:
            TrainingStopRequested: After the checkpoint is saved.
        """
        step = trainer.global_step
        if step == self._checked_step or step % self.check_every_n_steps != 0:
            return
        self._checked_step = step
        if not trainer.strategy.reduce_boolean_decision(self._requested, all=False):
            return

        checkpoint_callback = trainer.checkpoint_callback
        if checkpoint_callback is not None and checkpoint_callback.dirpath is not None:
            checkpoint_dir = Path(checkpoint_callback.dirpath)
        else:
            checkpoint_dir = Path(trainer.default_root_dir) / "checkpoints"
        checkpoint_path = checkpoint_dir / CHECKPOINT_FILENAME
        trainer.save_checkpoint(checkpoint_path)
        rank_zero_info(f"{self.signum.name} received, saved {checkpoint_path} at step {step}.")
        raise TrainingStopRequested(checkpoint_path)
