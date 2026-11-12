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

"""Refit step schedulers sized from the run length when a run resumes in another layout.

The framework fills ``total_steps`` of a scheduler such as ``OneCycleLR`` from the
trainer's estimate of the run's optimizer steps, which depends on the number of GPUs, the
batch size and the gradient accumulation. A resumed run restores the scheduler state of the
checkpoint, ``total_steps`` included. When the relaunch uses another global batch, the
restored horizon no longer matches the steps the run will take: with fewer steps than before
the schedule never reaches its end, with more ``OneCycleLR`` raises once it steps past its
horizon.
"""

from __future__ import annotations

import logging

import lightning as L
from lightning.pytorch.core.optimizer import _init_optimizers_and_lr_schedulers

from autoware_ml.resume.data_loader import LAYOUT_CHANGE_HINT, ResumableDataLoader

logger = logging.getLogger(__name__)


class SchedulerHorizon(L.Callback):
    """Refit the restored step schedulers of a resumed run to its optimizer steps.

    For every step-interval scheduler whose restored ``total_steps`` differs from the one the
    module sizes for this run, the scheduler state is rebuilt for this run and moved to the
    step that leaves exactly the optimizer steps the resumed run still takes: the rest of the
    interrupted epoch from the dataloader's resume position, and every later epoch. The
    schedule then ends on the run's last step, as an uninterrupted run in the new layout
    would. Include it through ``/defaults/preemption``.

    A resume in another layout is refused unless ``allow_layout_change`` is set.
    """

    def __init__(self, allow_layout_change: bool = False) -> None:
        """Initialize the callback.

        Args:
            allow_layout_change: Refit a scheduler sized for another number of optimizer
                steps instead of failing the resume.
        """
        self.allow_layout_change = allow_layout_change

    def on_train_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Refit the schedulers after the checkpoint restored them.

        Args:
            trainer: Trainer of the run.
            pl_module: Trained module, asked for schedulers sized for this run.
        """
        if trainer.ckpt_path is None:
            return
        sized = [
            index
            for index, config in enumerate(trainer.lr_scheduler_configs)
            if config.interval == "step" and hasattr(config.scheduler, "total_steps")
        ]
        if not sized:
            return
        _, fresh_configs = _init_optimizers_and_lr_schedulers(pl_module)
        stale = [
            index
            for index in sized
            if trainer.lr_scheduler_configs[index].scheduler.total_steps
            != fresh_configs[index].scheduler.total_steps
        ]
        if not stale:
            return
        if not self.allow_layout_change:
            sizes = [trainer.lr_scheduler_configs[index].scheduler.total_steps for index in stale]
            raise RuntimeError(
                f"The checkpoint's schedulers were sized for {sizes} optimizer steps, this run "
                f"takes {[fresh_configs[index].scheduler.total_steps for index in stale]}. "
                f"{LAYOUT_CHANGE_HINT}"
            )
        remaining = self._remaining_steps(trainer)
        if remaining is None:
            logger.warning(
                "The restored schedulers were sized for another number of optimizer steps "
                "and the remaining steps of this run are unknown; keeping them."
            )
            return
        for index in stale:
            scheduler = trainer.lr_scheduler_configs[index].scheduler
            fresh = fresh_configs[index].scheduler
            restored_total, restored_step = scheduler.total_steps, scheduler.last_epoch
            position = max(fresh.total_steps - remaining, 0)
            state = fresh.state_dict()
            # step() below advances to the position and writes its learning rate
            state["last_epoch"] = position - 1
            state["_step_count"] = position
            scheduler.load_state_dict(state)
            scheduler.step()
            logger.warning(
                "Scheduler %s restored at step %d of %d steps, refit to step %d of %d: the "
                "resumed run takes %d more optimizer steps.",
                type(scheduler).__name__,
                restored_step,
                restored_total,
                position,
                fresh.total_steps,
                remaining,
            )

    @staticmethod
    def _remaining_steps(trainer: L.Trainer) -> int | None:
        """Return the optimizer steps the resumed run still takes, None when unknown.

        Args:
            trainer: Trainer of the run, with its loop state restored.

        Returns:
            Steps of the rest of the current epoch plus those of every later epoch.
        """
        num_batches = trainer.num_training_batches
        max_epochs = trainer.max_epochs
        if trainer.max_steps != -1 or max_epochs is None or max_epochs < 0:
            return None
        if num_batches == float("inf"):
            return None
        num_batches = int(num_batches)
        accumulate = trainer.accumulate_grad_batches
        epoch = trainer.current_epoch
        loader = trainer.train_dataloader
        if isinstance(loader, ResumableDataLoader):
            start = loader.pending_start_batch(epoch)
        else:
            start = trainer.fit_loop.epoch_loop.batch_progress.current.ready
        # A step follows every accumulate-th batch and the epoch's last one.
        steps_per_epoch = -(-num_batches // accumulate)
        in_epoch = steps_per_epoch - min(start, num_batches) // accumulate
        return in_epoch + max(max_epochs - epoch - 1, 0) * steps_per_epoch
