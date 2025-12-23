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

"""Cyclic momentum scheduler for optimizer momentum adjustment."""

import math
from typing import List, Optional

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


class CyclicMomentumScheduler(LRScheduler):
    """Two-phase cosine momentum scheduler for cyclic training.

    Adjusts optimizer momentum (beta1 for AdamW) using cosine annealing across
    two phases. Works alongside learning rate schedulers.

    Phase 1 (warmup): Momentum decreases from base to minimum using cosine
    Phase 2 (decay): Momentum increases from minimum to maximum using cosine

    This matches the original cyclic-20e momentum schedule:
    - Phase 1 (epochs 0-8): 0.95 -> 0.895 (0.85/0.95)
    - Phase 2 (epochs 8-20): 0.895 -> 1.0

    Args:
        optimizer: Wrapped optimizer.
        warmup_epochs: Number of warmup epochs (phase 1).
        decay_epochs: Number of decay epochs (phase 2).
        min_momentum_factor: Minimum momentum factor (default: 0.85/0.95).
        max_momentum_factor: Maximum momentum factor (default: 1.0).
        last_epoch: The index of last epoch. Default: -1.

    Note:
        This is optional - momentum scheduling has minimal impact on final
        performance compared to learning rate scheduling. Only use if you
        need exact reproduction of original training behavior.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_epochs: int = 8,
        decay_epochs: int = 12,
        min_momentum_factor: float = 0.85 / 0.95,
        max_momentum_factor: float = 1.0,
        last_epoch: int = -1,
    ) -> None:
        """Initialize cyclic momentum scheduler.

        Args:
            optimizer: Wrapped optimizer.
            warmup_epochs: Number of warmup epochs (phase 1).
            decay_epochs: Number of decay epochs (phase 2).
            min_momentum_factor: Minimum momentum factor.
            max_momentum_factor: Maximum momentum factor.
            last_epoch: The index of last epoch.
        """
        self.warmup_epochs = warmup_epochs
        self.decay_epochs = decay_epochs
        self.min_momentum_factor = min_momentum_factor
        self.max_momentum_factor = max_momentum_factor

        # Store base momentum values
        self.base_momentums: List[float] = []
        for group in optimizer.param_groups:
            if "betas" in group:
                self.base_momentums.append(group["betas"][0])
            elif "momentum" in group:
                self.base_momentums.append(group["momentum"])
            else:
                self.base_momentums.append(0.9)

        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        """Dummy method required by LRScheduler - returns current LR unchanged."""
        return [group["lr"] for group in self.optimizer.param_groups]

    def get_momentum(self) -> List[float]:
        """Calculate momentum values using two-phase cosine annealing.

        Uses standard cosine formula: val = end + 0.5 * (start - end) * (1 + cos(π * t/T))

        Note on variable naming:
            - start_factor: Factor at beginning of phase (t=0)
            - end_factor: Factor at end of phase (t=T)

        The formula always goes from start_factor -> end_factor as t increases.
        """
        if self.last_epoch < self.warmup_epochs:
            # Warmup: momentum decreases from 1.0 (base) to min_momentum_factor
            start_factor = 1.0
            end_factor = self.min_momentum_factor
            t_cur = self.last_epoch
            t_max = self.warmup_epochs
        else:
            # Decay: momentum increases from min_momentum_factor to max_momentum_factor
            start_factor = self.min_momentum_factor
            end_factor = self.max_momentum_factor
            t_cur = self.last_epoch - self.warmup_epochs
            t_max = self.decay_epochs

        momentum_factor = end_factor + 0.5 * (start_factor - end_factor) * (
            1 + math.cos(math.pi * t_cur / t_max)
        )

        return [base_momentum * momentum_factor for base_momentum in self.base_momentums]

    def step(self, epoch: Optional[int] = None) -> None:
        """Update momentum values in optimizer.

        Args:
            epoch: Current epoch number. If None, increments last_epoch.
        """
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch

        momentums = self.get_momentum()

        for param_group, momentum in zip(self.optimizer.param_groups, momentums):
            if "betas" in param_group:
                param_group["betas"] = (momentum, param_group["betas"][1])
            elif "momentum" in param_group:
                param_group["momentum"] = momentum
