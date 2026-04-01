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

"""Boundary-aware losses for segmentation models.

This module provides reusable boundary-focused segmentation losses for dense
semantic prediction heads.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BoundaryLoss(nn.Module):
    """Compute a boundary-aware F1 loss for dense segmentation logits.

    The loss estimates boundaries from prediction and target masks and then
    penalizes disagreement between their boundary precision and recall.
    """

    def __init__(
        self, theta0: int = 3, ignore_index: int | None = None, loss_weight: float = 1.0
    ) -> None:
        """Initialize the boundary loss.

        Args:
            theta0: Pooling kernel size used to estimate boundaries.
            ignore_index: Label value excluded from the loss.
            loss_weight: Multiplicative loss weight.
        """
        super().__init__()
        self.theta0 = theta0
        self.ignore_index = ignore_index
        self.loss_weight = loss_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute boundary loss for range-view segmentation.

        Args:
            pred: Raw segmentation logits.
            target: Ground-truth class indices.

        Returns:
            Scalar boundary loss value.
        """
        pred = F.softmax(pred, dim=1)
        valid_mask = torch.ones_like(target, dtype=torch.bool)
        if self.ignore_index is not None:
            valid_mask = target != self.ignore_index

        safe_target = target.clone()
        if self.ignore_index is not None:
            safe_target = safe_target.masked_fill(~valid_mask, 0)

        one_hot_target = (
            F.one_hot(safe_target, num_classes=pred.shape[1]).permute(0, 3, 1, 2).float()
        )
        valid_mask = valid_mask.unsqueeze(1)
        one_hot_target = one_hot_target * valid_mask
        pred = pred * valid_mask

        target_boundary = F.max_pool2d(
            1.0 - one_hot_target,
            kernel_size=self.theta0,
            stride=1,
            padding=(self.theta0 - 1) // 2,
        ) - (1.0 - one_hot_target)
        pred_boundary = F.max_pool2d(
            1.0 - pred,
            kernel_size=self.theta0,
            stride=1,
            padding=(self.theta0 - 1) // 2,
        ) - (1.0 - pred)

        target_boundary = target_boundary.flatten(start_dim=2)
        pred_boundary = pred_boundary.flatten(start_dim=2)
        precision = (pred_boundary * target_boundary).sum(dim=2) / (pred_boundary.sum(dim=2) + 1e-7)
        recall = (pred_boundary * target_boundary).sum(dim=2) / (target_boundary.sum(dim=2) + 1e-7)
        boundary_f1 = 2.0 * precision * recall / (precision + recall + 1e-7)
        return self.loss_weight * torch.mean(1.0 - boundary_f1)
