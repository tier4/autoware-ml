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

"""Soft-target loss for sigmoid outputs such as detection heatmaps."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SigmoidSoftTargetLoss(nn.Module):
    """Binary cross entropy of the student logits against the teacher probabilities.

    A heatmap is almost entirely background, so the sum over cells is divided by the teacher's
    total probability mass, the soft count of its positives, the way the ground-truth heatmap
    loss divides by the number of objects. ``focal_gamma`` above zero weighs every cell by the
    gap between the two probabilities, which focuses the term on the cells the student gets
    wrong.

    Args:
        focal_gamma: Exponent of the ``|p_student - p_teacher|`` cell weight, 0 for none.
        normalize: ``teacher_mass`` divides by the teacher's probability mass, ``mean`` by the
            number of cells.

    Raises:
        ValueError: If the gamma is negative or the normalization is unknown.
    """

    def __init__(self, focal_gamma: float = 0.0, normalize: str = "teacher_mass") -> None:
        super().__init__()
        if focal_gamma < 0.0:
            raise ValueError(f"focal_gamma must not be negative, got {focal_gamma}.")
        if normalize not in {"teacher_mass", "mean"}:
            raise ValueError(f"normalize must be 'teacher_mass' or 'mean', got {normalize!r}.")
        self.focal_gamma = float(focal_gamma)
        self.normalize = normalize

    def forward(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
        """Compute the soft binary cross entropy.

        Args:
            student_logits: Logits of the student.
            teacher_logits: Logits of the teacher, same shape.

        Returns:
            Scalar loss.
        """
        student_logits = student_logits.float()
        teacher_probs = torch.sigmoid(teacher_logits.float())
        loss = F.binary_cross_entropy_with_logits(student_logits, teacher_probs, reduction="none")
        if self.focal_gamma > 0.0:
            gap = (torch.sigmoid(student_logits) - teacher_probs).abs()
            loss = loss * gap.pow(self.focal_gamma)
        if self.normalize == "mean":
            return loss.mean()
        return loss.sum() / teacher_probs.sum().clamp(min=1.0)
