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

"""Iteration-warmup, epoch-cosine learning-rate scheduler."""

from __future__ import annotations

import math

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


class IterWarmupEpochCosineLR(LRScheduler):
    """Linear warmup per optimization step multiplied by a cosine decay per epoch.

    Every param group starts at ``warmup_start_factor`` times its own learning rate and warms up
    linearly over ``warmup_iters`` steps. The cosine factor changes once per epoch of
    ``total_steps // max_epochs`` steps and decays towards ``eta_min_factor`` times the
    optimizer's default learning rate, one floor for every group. Under a
    ``trainer.max_steps`` cap the whole cosine runs inside the cap. Step it every optimization
    step (``scheduler_config: {interval: step}``).

    Args:
        optimizer: Wrapped optimizer.
        total_steps: Total optimization steps of the run, filled in by the framework.
        max_epochs: Number of epochs of the cosine; bind it to ``${trainer.max_epochs}``.
        warmup_iters: Number of warmup steps.
        warmup_start_factor: Learning-rate factor of every param group at step 0.
        eta_min_factor: Cosine floor as a fraction of the optimizer's default learning rate.
        last_epoch: Index of the last step.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        total_steps: int,
        max_epochs: int,
        warmup_iters: int,
        warmup_start_factor: float,
        eta_min_factor: float,
        last_epoch: int = -1,
    ) -> None:
        if max_epochs <= 0:
            raise ValueError(f"max_epochs must be positive, got {max_epochs}.")
        if total_steps < max_epochs:
            raise ValueError(
                f"total_steps ({total_steps}) must cover at least one step per epoch "
                f"({max_epochs} epochs)."
            )
        self.max_epochs = max_epochs
        self.steps_per_epoch = total_steps // max_epochs
        self.warmup_iters = warmup_iters
        self.warmup_start_factor = warmup_start_factor
        self.eta_min = eta_min_factor * float(optimizer.defaults["lr"])
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float]:
        """Compute the learning rate for the current optimization step."""
        step = self.last_epoch
        if self.warmup_iters > 0 and step < self.warmup_iters:
            progress = step / self.warmup_iters
            warmup_factor = self.warmup_start_factor + (1.0 - self.warmup_start_factor) * progress
        else:
            warmup_factor = 1.0

        # The total_steps % max_epochs trailing steps keep the last stage's factor.
        epoch_index = min(step // self.steps_per_epoch, self.max_epochs - 1)
        cosine = 0.5 * (1.0 + math.cos(math.pi * epoch_index / self.max_epochs))
        return [
            warmup_factor * (self.eta_min + (base_lr - self.eta_min) * cosine)
            for base_lr in self.base_lrs
        ]
