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

"""Cyclic cosine annealing learning rate scheduler."""

import math
from typing import List

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


class CyclicCosineAnnealingLR(LRScheduler):
    """Two-phase cosine annealing scheduler for cyclic training.

    This single-scheduler solution handles both warmup and decay phases using
    cosine annealing. It's simpler to configure than using SequentialLR with
    two separate schedulers.

    Phase 1 (warmup): LR increases from base_lr to max_lr using cosine
    Phase 2 (decay): LR decreases from max_lr to min_lr using cosine

    Example configuration for CenterPoint (cyclic-20e):
    - Warmup (epochs 0-8): 1e-4 -> 1e-3
    - Decay (epochs 8-20): 1e-3 -> 1e-8

    Args:
        optimizer: Wrapped optimizer.
        warmup_epochs: Number of warmup epochs (phase 1).
        decay_epochs: Number of decay epochs (phase 2).
        max_lr_factor: Max LR = base_lr * max_lr_factor.
        min_lr_factor: Min LR = base_lr * min_lr_factor.
        last_epoch: The index of last epoch. Default: -1.

    Example:
        ```python
        scheduler = CyclicCosineAnnealingLR(
            optimizer,
            warmup_epochs=8,
            decay_epochs=12,
            max_lr_factor=10.0,  # max_lr = 1e-4 * 10 = 1e-3
            min_lr_factor=1e-4   # min_lr = 1e-4 * 1e-4 = 1e-8
        )
        ```
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_epochs: int = 8,
        decay_epochs: int = 12,
        max_lr_factor: float = 10.0,
        min_lr_factor: float = 1e-4,
        last_epoch: int = -1,
    ) -> None:
        """Initialize cyclic cosine annealing scheduler.

        Args:
            optimizer: Wrapped optimizer.
            warmup_epochs: Number of warmup epochs (phase 1).
            decay_epochs: Number of decay epochs (phase 2).
            max_lr_factor: Max LR = base_lr * max_lr_factor.
            min_lr_factor: Min LR = base_lr * min_lr_factor.
            last_epoch: The index of last epoch.
        """
        self.warmup_epochs = warmup_epochs
        self.decay_epochs = decay_epochs
        self.total_epochs = warmup_epochs + decay_epochs
        self.max_lr_factor = max_lr_factor
        self.min_lr_factor = min_lr_factor
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        """Calculate learning rate using two-phase cosine annealing.

        Uses standard cosine formula: lr = end + 0.5 * (start - end) * (1 + cos(π * t/T))

        Note on variable naming (following PyTorch convention):
            - start_factor: Factor at beginning of phase (t=0)
            - end_factor: Factor at end of phase (t=T)

        The formula always goes from start_factor -> end_factor as t increases.
        """
        if self.last_epoch < self.warmup_epochs:
            # Warmup: factor increases from 1.0 (base_lr) to max_lr_factor (peak)
            start_factor = 1.0
            end_factor = self.max_lr_factor
            t_cur = self.last_epoch
            t_max = self.warmup_epochs
        else:
            # Decay: factor decreases from max_lr_factor (peak) to min_lr_factor
            start_factor = self.max_lr_factor
            end_factor = self.min_lr_factor
            t_cur = self.last_epoch - self.warmup_epochs
            t_max = self.decay_epochs

        lr_factor = end_factor + 0.5 * (start_factor - end_factor) * (
            1 + math.cos(math.pi * t_cur / t_max)
        )

        return [base_lr * lr_factor for base_lr in self.base_lrs]
