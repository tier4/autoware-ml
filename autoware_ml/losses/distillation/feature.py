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

"""Losses between intermediate features of a student and a teacher."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureMSELoss(nn.Module):
    """Mean squared error between two feature tensors of the same shape.

    Args:
        normalize: Scale both tensors to unit norm along ``channel_dim`` first, so the term
            matches directions rather than magnitudes.
        channel_dim: Dimension of the channels, read when normalizing.
    """

    def __init__(self, normalize: bool = False, channel_dim: int = 1) -> None:
        super().__init__()
        self.normalize = bool(normalize)
        self.channel_dim = int(channel_dim)

    def forward(self, student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
        """Compute the error.

        Args:
            student: Student features.
            teacher: Teacher features, same shape.

        Returns:
            Scalar loss.
        """
        student = student.float()
        teacher = teacher.float()
        if self.normalize:
            student = F.normalize(student, dim=self.channel_dim)
            teacher = F.normalize(teacher, dim=self.channel_dim)
        return F.mse_loss(student, teacher)
