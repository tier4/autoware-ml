# Copyright 2025 TIER IV, Inc.
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

"""Cosine annealing learning rate scheduler."""

import math
from typing import List

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


class CosineAnnealingLR(LRScheduler):
    """Cosine annealing scheduler with recursive formula.

    This implements a cosine annealing schedule that works well with PyTorch's
    SequentialLR for multi-phase training schedules.

    Formula:
        η_t = η_min + 0.5 * (η_max - η_min) * (1 + cos(π * T_cur / T_max))

    Where:
        - η_max is the initial LR (starting point at t=0)
        - η_min is the target LR (ending point at t=T_max)
        - T_cur is the current epoch
        - T_max is the total epochs for this phase

    Note on naming convention:
        The standard cosine formula goes from η_max -> η_min (start -> end).
        For warmup (increasing LR), set eta_min HIGHER than current LR.
        For decay (decreasing LR), set eta_min LOWER than current LR.

    Args:
        optimizer: Wrapped optimizer.
        T_max: Maximum number of epochs for this phase.
        eta_min: Target learning rate (can be higher or lower than initial).
        last_epoch: The index of last epoch. Default: -1.

    Example:
        ```python
        # Phase 1 (warmup): LR goes from 1e-4 to 1e-3
        scheduler1 = CosineAnnealingLR(optimizer, T_max=8, eta_min=1e-3)

        # Phase 2 (decay): LR goes from 1e-3 to 1e-8
        scheduler2 = CosineAnnealingLR(optimizer, T_max=12, eta_min=1e-8)

        # Combine with SequentialLR
        scheduler = SequentialLR(optimizer, [scheduler1, scheduler2], milestones=[8])
        ```
    """

    def __init__(
        self,
        optimizer: Optimizer,
        T_max: int,
        eta_min: float,
        last_epoch: int = -1,
    ) -> None:
        """Initialize cosine annealing scheduler.

        Args:
            optimizer: Wrapped optimizer.
            T_max: Maximum number of epochs for this phase.
            eta_min: Target learning rate.
            last_epoch: The index of last epoch.
        """
        self.T_max = T_max
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        """Calculate learning rate using recursive cosine annealing formula.

        At epoch 0 (called during __init__), we return the current LR without modification.
        This establishes the starting point (η_max) for the cosine schedule.

        We use the current LR from optimizer.param_groups instead of base_lrs to support
        SequentialLR where the base_lrs may not be updated correctly.
        """
        if self.last_epoch == 0:
            return [group["lr"] for group in self.optimizer.param_groups]
        elif (self.last_epoch - 1 - self.T_max) % (2 * self.T_max) == 0:
            return [
                group["lr"]
                + (group["lr"] - self.eta_min) * (1 - math.cos(math.pi / self.T_max)) / 2
                for group in self.optimizer.param_groups
            ]

        return [
            (1 + math.cos(math.pi * self.last_epoch / self.T_max))
            / (1 + math.cos(math.pi * (self.last_epoch - 1) / self.T_max))
            * (group["lr"] - self.eta_min)
            + self.eta_min
            for group in self.optimizer.param_groups
        ]
