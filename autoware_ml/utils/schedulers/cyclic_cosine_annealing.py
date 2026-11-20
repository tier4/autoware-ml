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

"""Cyclic cosine annealing learning-rate scheduler utilities.

This module provides cyclic cosine annealing learning-rate schedules used by
selected training configurations.
"""

import math

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


class CyclicCosineAnnealingLR(LRScheduler):
    """Two-phase cosine annealing scheduler for cyclic training.

    This single-scheduler solution handles both warmup and decay phases using
    cosine annealing. It's simpler to configure than using SequentialLR with
    two separate schedulers.

    Phase 1 (warmup): LR increases from base_lr to max_lr using cosine
    Phase 2 (decay): LR decreases from max_lr to min_lr using cosine

    Example cyclic-20e-style configuration:
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

    def get_lr(self) -> list[float]:
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
            # Past the decay the LR stays at its end instead of climbing back
            t_cur = min(self.last_epoch - self.warmup_epochs, self.decay_epochs)
            t_max = self.decay_epochs

        lr_factor = end_factor + 0.5 * (start_factor - end_factor) * (
            1 + math.cos(math.pi * t_cur / t_max)
        )

        return [base_lr * lr_factor for base_lr in self.base_lrs]


class StepCyclicCosineAnnealingLR(CyclicCosineAnnealingLR):
    """Cyclic cosine annealing stepped every optimizer step, its warmup given in epochs.

    The warmup and the decay over the remaining epochs are spread over the optimizer steps of the
    run, so it needs a Lightning scheduler config with ``interval: step``. The framework fills
    ``total_steps`` from the trainer.

    Args:
        optimizer: Wrapped optimizer.
        total_steps: Optimizer steps of the whole run.
        max_epochs: Epochs of the run, bind it to ``${trainer.max_epochs}``.
        warmup_epochs: Epochs of the warmup phase, the decay takes the remaining ones.
        max_lr_factor: Peak learning rate as a multiple of the base learning rate.
        min_lr_factor: Final learning rate as a multiple of the base learning rate.
        last_epoch: The index of the last step.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        total_steps: int,
        max_epochs: int,
        warmup_epochs: float,
        max_lr_factor: float = 10.0,
        min_lr_factor: float = 1e-4,
        last_epoch: int = -1,
    ) -> None:
        """Initialize the per-step cyclic cosine annealing scheduler.

        Raises:
            ValueError: If the warmup does not leave a decay inside the run, if the run has
                fewer steps than epochs, or if the decay gets no step.
        """
        if not 0 <= warmup_epochs < max_epochs:
            raise ValueError(
                f"warmup_epochs must be in [0, max_epochs), got {warmup_epochs} with "
                f"max_epochs={max_epochs}."
            )
        if total_steps < max_epochs:
            raise ValueError(
                f"total_steps ({total_steps}) must cover at least one step per epoch "
                f"({max_epochs} epochs)."
            )
        warmup_steps = round(total_steps * warmup_epochs / max_epochs)
        decay_steps = total_steps - warmup_steps
        if decay_steps < 1:
            raise ValueError(
                f"A warmup of {warmup_epochs} of {max_epochs} epochs leaves no decay step of "
                f"the {total_steps} steps."
            )
        super().__init__(
            optimizer,
            warmup_epochs=warmup_steps,
            decay_epochs=decay_steps,
            max_lr_factor=max_lr_factor,
            min_lr_factor=min_lr_factor,
            last_epoch=last_epoch,
        )
