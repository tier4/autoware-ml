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

"""Intra-epoch checkpoints that let a preempted run resume with little rework.

On a shared slurm cluster a job can be requeued at any moment with no grace
period, so an end-of-epoch checkpoint can cost a whole epoch of GPU time. This
callback saves the full training state a fixed number of times per epoch (plus
at the end of the epoch), keeps only the newest few files, and points ``last.ckpt``
at the newest one. A relaunch resumes from ``last.ckpt``; together with the
resumable training dataloader the interrupted epoch continues with the batch after
the one that was saved.
"""

from __future__ import annotations

import logging
import math
import os
import shutil
from pathlib import Path
from typing import Any

import lightning as L
from lightning.pytorch.utilities.exceptions import SIGTERMException
from lightning.pytorch.utilities.rank_zero import rank_zero_only

logger = logging.getLogger(__name__)


class PeriodicCheckpoint(L.Callback):
    """Save rotating full checkpoints several times per epoch.

    The save period is derived at the start of training from the number of
    optimizer steps in an epoch, so ``saves_per_epoch`` reads the same whatever
    the corpus, the device count or the gradient accumulation. Saves happen from
    ``on_train_batch_end`` right after an optimizer step, at the end of every
    training epoch, and when the job is asked to stop.

    A stop request is a file: the launcher creates the path named by the
    ``AUTOWARE_ML_STOP_FILE`` environment variable (or ``stop_file``) when the time
    limit approaches. Every rank checks for it at the end of every batch, the ranks
    agree on the decision through one all-reduce, save a checkpoint together and set
    ``trainer.should_stop``. Signals are not used for this: Lightning's SIGTERM
    handler broadcasts from inside the signal handler, which deadlocks against the
    collective the ranks are already running (stage 2 of the foundation model hung
    for its whole 15 minute grace period and was hard-killed without a requeue).
    """

    LAST_NAME = "last.ckpt"

    def __init__(
        self,
        dirpath: str | None = None,
        saves_per_epoch: int = 4,
        keep_last: int = 2,
        save_on_train_epoch_end: bool = True,
        filename: str = "periodic-epoch{epoch:03d}-step{step:08d}.ckpt",
        stop_file: str | None = None,
    ) -> None:
        """Initialize the callback.

        Args:
            dirpath: Directory receiving the checkpoints. Defaults to
                ``<default_root_dir>/checkpoints`` at setup.
            saves_per_epoch: Number of intra-epoch saves.
            keep_last: Number of periodic files kept; older ones are deleted.
            save_on_train_epoch_end: Whether to also save at the end of every epoch.
            filename: Format of the periodic files, with ``epoch`` and ``step``.
            stop_file: Path whose existence asks the run to save and stop; defaults to
                the ``AUTOWARE_ML_STOP_FILE`` environment variable.

        Raises:
            ValueError: Raised for a non-positive count.
        """
        super().__init__()
        if saves_per_epoch < 1:
            raise ValueError(f"saves_per_epoch must be at least 1, got {saves_per_epoch}.")
        if keep_last < 1:
            raise ValueError(f"keep_last must be at least 1, got {keep_last}.")
        self.dirpath = dirpath
        self.saves_per_epoch = int(saves_per_epoch)
        self.keep_last = int(keep_last)
        self.save_on_train_epoch_end = bool(save_on_train_epoch_end)
        self.filename = filename
        self.stop_file = stop_file or os.environ.get("AUTOWARE_ML_STOP_FILE") or None
        self.stop_requested = False
        self.every_n_steps: int | None = None
        self.saved_paths: list[str] = []
        self._last_saved_step = -1
        self._last_seen_step = -1
        self._saves_this_run = 0

    @property
    def last_path(self) -> Path | None:
        """Return the path of ``last.ckpt`` once the directory is known."""
        return Path(self.dirpath) / self.LAST_NAME if self.dirpath is not None else None

    def setup(self, trainer: L.Trainer, pl_module: L.LightningModule, stage: str) -> None:
        """Resolve the checkpoint directory on every rank identically."""
        if self.dirpath is None:
            self.dirpath = os.path.join(trainer.default_root_dir, "checkpoints")
        self.dirpath = trainer.strategy.broadcast(self.dirpath)

    def on_train_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Derive the save period from the length of an epoch in optimizer steps."""
        num_batches = trainer.num_training_batches
        if num_batches == float("inf"):
            raise ValueError(
                "PeriodicCheckpoint needs a sized training dataloader to derive its period; "
                "use ModelCheckpoint(every_n_train_steps=...) with an iterable dataset."
            )
        steps_per_epoch = math.ceil(num_batches / trainer.accumulate_grad_batches)
        self.every_n_steps = max(1, math.ceil(steps_per_epoch / self.saves_per_epoch))
        self._last_seen_step = trainer.global_step
        if trainer.is_global_zero:
            logger.info(
                "PeriodicCheckpoint: %d optimizer steps per epoch, saving every %d steps into %s.",
                steps_per_epoch,
                self.every_n_steps,
                self.dirpath,
            )

    def on_train_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Save right after every ``every_n_steps``-th optimizer step."""
        if self._stop_requested(trainer):
            if trainer.global_step > 0 and trainer.global_step != self._last_saved_step:
                self._save(trainer)
            trainer.should_stop = True
            return
        step = trainer.global_step
        if step == self._last_seen_step:
            return  # gradient accumulation, no optimizer step happened
        self._last_seen_step = step
        assert self.every_n_steps is not None
        if step % self.every_n_steps == 0 and step != self._last_saved_step:
            self._save(trainer)

    def _stop_requested(self, trainer: L.Trainer) -> bool:
        """Return whether the launcher asked the job to stop, agreed on by every rank.

        The file check runs on every rank and the answers are combined with one
        all-reduce, so the ranks take the same branch at the same batch boundary even
        when the file appears between two ranks' checks.
        """
        if self.stop_requested:
            return True
        if self.stop_file is None:
            return False
        seen = os.path.exists(self.stop_file)
        if trainer.world_size > 1:
            seen = bool(trainer.strategy.reduce_boolean_decision(seen, all=False))
        if seen:
            self.stop_requested = True
            if trainer.is_global_zero:
                logger.info(
                    "PeriodicCheckpoint: stop requested through %s, saving and ending the run.",
                    self.stop_file,
                )
        return seen

    def on_train_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Save at the end of the epoch so the next one resumes from a clean boundary."""
        if self.save_on_train_epoch_end and trainer.global_step != self._last_saved_step:
            self._save(trainer)

    def on_exception(
        self, trainer: L.Trainer, pl_module: L.LightningModule, exception: BaseException
    ) -> None:
        """Save when the job is told to stop; other exceptions are left alone.

        SIGTERM is raised by Lightning at a batch boundary on every rank, so the
        collective save is safe there. An arbitrary exception may have occurred on
        one rank only, where a collective save would deadlock.
        """
        if (
            isinstance(exception, (SIGTERMException, KeyboardInterrupt))
            and trainer.global_step > 0
            and trainer.global_step != self._last_saved_step
        ):
            self._save(trainer)

    def state_dict(self) -> dict[str, Any]:
        """Return the rotation bookkeeping."""
        return {
            "saved_paths": list(self.saved_paths),
            "every_n_steps": self.every_n_steps,
            "last_saved_step": self._last_saved_step,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore the rotation bookkeeping."""
        self.saved_paths = [path for path in state_dict.get("saved_paths", []) if path]
        self._last_saved_step = int(state_dict.get("last_saved_step", -1))

    def _save(self, trainer: L.Trainer) -> None:
        """Write the checkpoint on every rank's strategy, then rotate on rank zero."""
        assert self.dirpath is not None
        filepath = os.path.join(
            self.dirpath,
            self.filename.format(epoch=trainer.current_epoch, step=trainer.global_step),
        )
        # Bookkeeping first, so the checkpoint being written lists itself and the
        # step it was written at; a failed write leaves a path the rotation skips.
        self._last_saved_step = trainer.global_step
        self.saved_paths.append(filepath)
        trainer.save_checkpoint(filepath)
        self._saves_this_run += 1
        self._rotate(trainer)
        trainer.strategy.barrier("PeriodicCheckpoint.save")

    @rank_zero_only
    def _rotate(self, trainer: L.Trainer) -> None:
        """Point ``last.ckpt`` at the newest file and delete files beyond ``keep_last``.

        The file this run resumed from is protected until ``keep_last`` newer saves
        have completed: before that it is the only state proven to load, after that
        the rotation treats it like any other file.
        """
        self._link_last(self.saved_paths[-1])
        resume_path = None
        if trainer.ckpt_path is not None and self._saves_this_run < self.keep_last:
            resume_path = os.path.realpath(trainer.ckpt_path)
        excess = len(self.saved_paths) - self.keep_last
        kept: list[str] = []
        for path in self.saved_paths:
            if excess > 0 and os.path.realpath(path) != resume_path:
                if os.path.exists(path):
                    os.remove(path)
                excess -= 1
            else:
                kept.append(path)
        self.saved_paths = kept

    def _link_last(self, target: str) -> None:
        """Make ``last.ckpt`` a relative symlink to ``target``, copying if links fail."""
        last = str(self.last_path)
        if os.path.lexists(last):
            os.remove(last)
        try:
            os.symlink(os.path.relpath(target, os.path.dirname(last)), last)
        except OSError:
            shutil.copy(target, last)
