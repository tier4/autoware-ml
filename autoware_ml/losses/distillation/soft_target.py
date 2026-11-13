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

"""Soft-target losses between the logits of a student and a teacher."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftTargetKLDivLoss(nn.Module):
    """KL divergence from the softened teacher distribution to the student's.

    Both logits are divided by the temperature before the softmax and the divergence is scaled
    by its square, so the gradient magnitude stays comparable across temperatures. The mean is
    taken over every position (points, voxels or cells).

    Args:
        temperature: Softening temperature, 1 keeps the raw distributions.
        class_dim: Dimension holding the classes, 1 for ``(N, C)`` logits and ``(B, C, H, W)``
            maps alike.

    Raises:
        ValueError: If the temperature is not positive.
    """

    def __init__(self, temperature: float = 1.0, class_dim: int = 1) -> None:
        super().__init__()
        if temperature <= 0.0:
            raise ValueError(f"temperature must be positive, got {temperature}.")
        self.temperature = float(temperature)
        self.class_dim = int(class_dim)

    def forward(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
        """Compute the temperature-scaled divergence.

        Args:
            student_logits: Logits of the student.
            teacher_logits: Logits of the teacher, same shape.

        Returns:
            Scalar loss.
        """
        student_log_probs = F.log_softmax(
            student_logits.float() / self.temperature, dim=self.class_dim
        )
        teacher_log_probs = F.log_softmax(
            teacher_logits.float() / self.temperature, dim=self.class_dim
        )
        divergence = (teacher_log_probs.exp() * (teacher_log_probs - student_log_probs)).sum(
            dim=self.class_dim
        )
        return divergence.mean() * self.temperature**2
