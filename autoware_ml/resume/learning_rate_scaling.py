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

"""Scale the learning rate when a resumed run trains with another global batch.

The learning rates of a config are tuned for one global batch: GPUs times batch size times
gradient accumulation. A relaunch on fewer or more GPUs without a matching accumulation trains
with another global batch, and the same learning rate then takes steps of another size per
sample. This callback remembers the global batch of the run's first launch in the checkpoint
and, in every later launch, multiplies the learning rate of every optimizer step by a factor of
the ratio of the two.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable
from enum import StrEnum, auto
from typing import Any

import lightning as L
from torch.optim import Optimizer
from torch.utils.hooks import RemovableHandle

from autoware_ml.resume.data_loader import LAYOUT_CHANGE_HINT

logger = logging.getLogger(__name__)


class LearningRateScalingRule(StrEnum):
    """
    How the learning rate follows the ratio of the global batch to the tuned one.

    Attributes:
      SQRT: Factor of the square root of the ratio, the rule for adaptive optimizers such as
        Adam and AdamW.
      LINEAR: Factor of the ratio, the rule for SGD.
      NONE: Keep the learning rate.
    """

    SQRT = auto()
    LINEAR = auto()
    NONE = auto()

    def factor(self, ratio: float) -> float:
        """
        Return the learning-rate factor for a ratio of global batches.

        Args:
          ratio: Global batch of the run divided by the tuned global batch.

        Returns:
          float: Factor every learning rate is multiplied with.
        """

        if self is LearningRateScalingRule.SQRT:
            return math.sqrt(ratio)
        if self is LearningRateScalingRule.LINEAR:
            return ratio
        return 1.0


class BatchSizeLearningRateScaling(L.Callback):
    """
    Multiply the learning rate of every optimizer step after a change of the global batch.

    The factor is applied around each optimizer step only: the param groups hold the tuned
    learning rate at every other moment, so schedulers keep computing from it, checkpoints
    store it and the logged learning rate is the tuned one. The tuned global batch is the
    one of the run's first launch and travels in the checkpoint, so successive relaunches do
    not compound their factors. Include it through ``/defaults/preemption``.

    A resume with another global batch is refused unless ``allow_layout_change`` is set.
    """

    def __init__(
        self,
        rule: str | LearningRateScalingRule = LearningRateScalingRule.SQRT,
        allow_layout_change: bool = False,
    ) -> None:
        """
        Initialize the callback.

        Args:
          rule: Scaling rule, ``sqrt``, ``linear`` or ``none``.
          allow_layout_change: Scale the learning rate of a resume with another global batch
            instead of failing it.
        """

        self.rule = LearningRateScalingRule(rule)
        self.allow_layout_change = allow_layout_change
        self.reference_global_batch: int | None = None
        self.factor = 1.0
        self._handles: list[RemovableHandle] = []

    @staticmethod
    def global_batch(trainer: L.Trainer) -> int | None:
        """
        Return the samples of one optimizer step over every rank, None when unknown.

        Args:
          trainer: Trainer with its training dataloader built.

        Returns:
          int | None: Ranks times batch size times gradient accumulation.
        """

        batch_size = getattr(trainer.train_dataloader, "batch_size", None)
        if batch_size is None:
            return None
        return int(trainer.world_size * batch_size * trainer.accumulate_grad_batches)

    def on_train_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """
        Compare the global batch with the tuned one and hook the optimizers when they differ.

        Args:
          trainer: Trainer of the run, with a resumed checkpoint already restored.
          pl_module: Trained module.
        """

        global_batch = self.global_batch(trainer)
        if global_batch is None:
            logger.warning("The global batch of the run is unknown; keeping the learning rate.")
            return
        if self.reference_global_batch is None:
            self.reference_global_batch = global_batch
        if global_batch != self.reference_global_batch and not self.allow_layout_change:
            raise RuntimeError(
                f"The run was launched with a global batch of {self.reference_global_batch}, "
                f"this run uses {global_batch}. {LAYOUT_CHANGE_HINT}"
            )
        ratio = global_batch / self.reference_global_batch
        self.factor = self.rule.factor(ratio)
        if self.factor == 1.0:
            return
        logger.warning(
            "Global batch %d instead of the tuned %d: every learning rate is multiplied by "
            "%.4g (%s rule) in the optimizer steps of this run.",
            global_batch,
            self.reference_global_batch,
            self.factor,
            self.rule,
        )
        for optimizer in trainer.optimizers:
            pre_hook, post_hook = self._step_hooks()
            self._handles.append(optimizer.register_step_pre_hook(pre_hook))
            self._handles.append(optimizer.register_step_post_hook(post_hook))

    def on_fit_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Remove the optimizer hooks."""
        self._remove_hooks()

    def on_exception(
        self, trainer: L.Trainer, pl_module: L.LightningModule, exception: BaseException
    ) -> None:
        """Remove the optimizer hooks when the run fails."""
        self._remove_hooks()

    def state_dict(self) -> dict[str, Any]:
        """Return the tuned global batch for the checkpoint."""
        return {"reference_global_batch": self.reference_global_batch}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """
        Restore the tuned global batch of the run's first launch.

        Args:
          state_dict: Mapping produced by :meth:`state_dict`.
        """

        self.reference_global_batch = state_dict.get("reference_global_batch")

    def _step_hooks(
        self,
    ) -> tuple[Callable[[Optimizer, Any, Any], None], Callable[[Optimizer, Any, Any], None]]:
        """
        Build the hooks that scale the learning rates of one optimizer around its step.

        Returns:
          tuple: The step pre-hook, which scales, and the step post-hook, which restores.
        """

        # An optimizer whose step calls its parent class's step runs the hooks once per
        # level; only the outermost call scales and restores.
        tuned: list[Any] = []
        depth = 0

        def scale(optimizer: Optimizer, args: Any, kwargs: Any) -> None:
            nonlocal depth
            depth += 1
            if depth > 1:
                return
            tuned[:] = [group["lr"] for group in optimizer.param_groups]
            for group in optimizer.param_groups:
                group["lr"] = group["lr"] * self.factor

        def restore(optimizer: Optimizer, args: Any, kwargs: Any) -> None:
            nonlocal depth
            depth -= 1
            if depth > 0:
                return
            for group, learning_rate in zip(optimizer.param_groups, tuned, strict=True):
                group["lr"] = learning_rate
            tuned.clear()

        return scale, restore

    def _remove_hooks(self) -> None:
        """Remove every registered optimizer hook."""
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
