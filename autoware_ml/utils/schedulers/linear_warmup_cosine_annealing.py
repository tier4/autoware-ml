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

"""Linear warmup with cosine annealing learning rate scheduler."""

import math
from typing import List

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


class LinearWarmupCosineAnnealingLR(LRScheduler):
    """Two-phase scheduler: linear warmup followed by cosine annealing decay.

    This scheduler provides a simple and intuitive warmup + decay schedule:

    Phase 1 (warmup): LR increases linearly from warmup_start_lr to base_lr
    Phase 2 (decay): LR decreases from base_lr to eta_min using cosine annealing

    This is simpler to configure than CyclicCosineAnnealingLR as it uses:
    - Linear warmup (not cosine) which is more common in practice
    - Direct LR values instead of factors
    - Clearer naming conventions

    Args:
        optimizer: Wrapped optimizer.
        warmup_epochs: Number of warmup epochs (phase 1).
        max_epochs: Total number of training epochs.
        warmup_start_lr: Starting LR for warmup (default: 1e-6).
        eta_min: Minimum LR at end of cosine decay (default: 1e-8).
        last_epoch: The index of last epoch. Default: -1.

    Example:
        ```python
        # Warmup from 1e-6 to 1e-3 over 5 epochs, then decay to 1e-8
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer,
            warmup_epochs=5,
            max_epochs=20,
            warmup_start_lr=1e-6,
            eta_min=1e-8
        )
        ```
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_epochs: int = 5,
        max_epochs: int = 20,
        warmup_start_lr: float = 1e-6,
        eta_min: float = 1e-8,
        last_epoch: int = -1,
    ) -> None:
        """Initialize linear warmup cosine annealing scheduler.

        Args:
            optimizer: Wrapped optimizer.
            warmup_epochs: Number of warmup epochs (phase 1).
            max_epochs: Total number of training epochs.
            warmup_start_lr: Starting LR for warmup.
            eta_min: Minimum LR at end of cosine decay.
            last_epoch: The index of last epoch.
        """
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.decay_epochs = max_epochs - warmup_epochs
        self.warmup_start_lr = warmup_start_lr
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        """Calculate learning rate for current epoch.

        Phase 1 (warmup): Linear interpolation from warmup_start_lr to base_lr
        Phase 2 (decay): Cosine annealing from base_lr to eta_min
        """
        if self.last_epoch < self.warmup_epochs:
            # Linear warmup: start_lr -> base_lr
            alpha = self.last_epoch / max(1, self.warmup_epochs)
            return [
                self.warmup_start_lr + alpha * (base_lr - self.warmup_start_lr)
                for base_lr in self.base_lrs
            ]
        else:
            # Cosine annealing: base_lr -> eta_min
            t_cur = self.last_epoch - self.warmup_epochs
            t_max = max(1, self.decay_epochs)

            return [
                self.eta_min
                + 0.5 * (base_lr - self.eta_min) * (1 + math.cos(math.pi * t_cur / t_max))
                for base_lr in self.base_lrs
            ]
