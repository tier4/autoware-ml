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

"""Learning-rate schedulers shared by detection2d models."""

from __future__ import annotations

import math

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def _flat_cosine_scale(
    current_step: int,
    total_steps: int,
    warmup_steps: int,
    flat_steps: int,
    no_aug_steps: int,
    lr_gamma: float,
) -> float:
    if total_steps <= 0:
        return 1.0
    if warmup_steps > 0 and current_step <= warmup_steps:
        progress = current_step / float(warmup_steps)
        return progress * progress
    if current_step <= flat_steps:
        return 1.0
    if no_aug_steps > 0 and current_step >= total_steps - no_aug_steps:
        return lr_gamma

    cosine_denominator = max(total_steps - flat_steps - no_aug_steps, 1)
    cosine_progress = (current_step - flat_steps) / float(cosine_denominator)
    cosine_decay = 0.5 * (1.0 + math.cos(math.pi * cosine_progress))
    return lr_gamma + (1.0 - lr_gamma) * cosine_decay


class FlatCosineAnnealingLR(LambdaLR):
    """Warmup + flat + cosine decay schedule matching RT-DETR training."""

    def __init__(
        self,
        optimizer: Optimizer,
        total_steps: int,
        warmup_steps: int = 0,
        flat_steps: int | None = None,
        no_aug_steps: int | None = None,
        max_epochs: int | None = None,
        flat_epochs: int = 0,
        no_aug_epochs: int = 0,
        lr_gamma: float = 0.5,
        last_epoch: int = -1,
    ) -> None:
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        steps_per_epoch = (
            max(int(total_steps // max_epochs), 1) if max_epochs not in (None, 0) else None
        )
        self.flat_steps = flat_steps if flat_steps is not None else (flat_epochs * steps_per_epoch if steps_per_epoch is not None else 0)
        self.no_aug_steps = no_aug_steps if no_aug_steps is not None else (no_aug_epochs * steps_per_epoch if steps_per_epoch is not None else 0)
        self.lr_gamma = lr_gamma
        super().__init__(
            optimizer,
            lr_lambda=lambda step: _flat_cosine_scale(
                step,
                total_steps=self.total_steps,
                warmup_steps=self.warmup_steps,
                flat_steps=self.flat_steps,
                no_aug_steps=self.no_aug_steps,
                lr_gamma=self.lr_gamma,
            ),
            last_epoch=last_epoch,
        )
