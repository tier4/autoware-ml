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

"""Iteration-warmup, epoch-cosine learning-rate schedule.

This reproduces the common mmengine recipe of chaining an iteration-based
``LinearLR`` warmup with an epoch-based ``CosineAnnealingLR``: the two factors
multiply, warmup resolving per optimization step while the cosine decay
resolves per epoch. Step it every optimization step
(``scheduler_config: {interval: step}``).
"""

from __future__ import annotations

import math

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


class IterWarmupEpochCosineLR(LRScheduler):
    """Linear per-iteration warmup multiplied by per-epoch cosine annealing.

    Args:
        optimizer: Wrapped optimizer.
        total_steps: Total optimization steps of the run (auto-filled by the
            framework from ``trainer.estimated_stepping_batches``).
        max_epochs: Total training epochs; sets both the epoch length
            (``total_steps / max_epochs``) and the cosine period.
        warmup_iters: Length of the linear warmup in optimization steps.
        warmup_start_factor: Learning-rate factor at step 0.
        eta_min_factor: Cosine floor as a fraction of the base learning rate.
        last_epoch: Index of the last step (scheduler steps once per
            optimization step).
    """

    def __init__(
        self,
        optimizer: Optimizer,
        total_steps: int,
        max_epochs: int,
        warmup_iters: int = 500,
        warmup_start_factor: float = 1.0 / 3.0,
        eta_min_factor: float = 1e-4,
        last_epoch: int = -1,
    ) -> None:
        if total_steps <= 0:
            raise ValueError(f"total_steps must be positive, got {total_steps}.")
        if max_epochs <= 0:
            raise ValueError(f"max_epochs must be positive, got {max_epochs}.")
        self.max_epochs = max_epochs
        self.steps_per_epoch = max(total_steps // max_epochs, 1)
        self.warmup_iters = warmup_iters
        self.warmup_start_factor = warmup_start_factor
        self.eta_min_factor = eta_min_factor
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float]:
        """Compute the learning rate for the current optimization step."""
        step = self.last_epoch
        if self.warmup_iters > 0 and step < self.warmup_iters:
            progress = step / self.warmup_iters
            warmup_factor = self.warmup_start_factor + (1.0 - self.warmup_start_factor) * progress
        else:
            warmup_factor = 1.0

        # Trailing steps from total_steps % max_epochs stay at the last
        # training epoch's factor rather than dropping to the eta_min floor.
        epoch_index = min(step // self.steps_per_epoch, self.max_epochs - 1)
        cosine_factor = self.eta_min_factor + (1.0 - self.eta_min_factor) * 0.5 * (
            1.0 + math.cos(math.pi * epoch_index / self.max_epochs)
        )
        return [base_lr * warmup_factor * cosine_factor for base_lr in self.base_lrs]
