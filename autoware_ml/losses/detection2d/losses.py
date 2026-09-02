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

"""2D detection losses for auxiliary image-plane supervision.

These losses back the auxiliary 2D head of camera 3D detectors
(Focal-PETR-style): quality focal classification, GIoU box regression, and
Gaussian-heatmap centerness supervision.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from autoware_ml.models.detection3d.task_modules.boxes2d import bbox_overlaps


class QualityFocalLoss(nn.Module):
    """Quality focal loss over sigmoid logits with IoU-quality targets.

    Positive queries are supervised toward the IoU between their predicted
    and assigned boxes instead of a hard 1, following Generalized Focal Loss.
    """

    def __init__(self, beta: float = 2.0, loss_weight: float = 1.0) -> None:
        """Initialize the quality focal loss.

        Args:
            beta: Modulating exponent on the quality gap.
            loss_weight: Multiplier applied to the final loss.
        """
        super().__init__()
        self.beta = beta
        self.loss_weight = loss_weight

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        quality_scores: torch.Tensor,
        class_weights: torch.Tensor | None = None,
        avg_factor: float | None = None,
    ) -> torch.Tensor:
        """Compute quality focal loss.

        Args:
            logits: Classification logits with shape ``(N, C)``.
            labels: Class indices with shape ``(N,)``; ``C`` marks background.
            quality_scores: IoU quality targets with shape ``(N,)``, consumed
                only at positive rows.
            class_weights: Optional per-element weights with shape ``(N, C)``
                (used by partial-ignore to mask class columns).
            avg_factor: Optional normalization factor.

        Returns:
            Scalar loss value.
        """
        num_classes = logits.size(1)
        prob = logits.sigmoid()
        # All positions start as background: target 0 modulated by p^beta.
        loss = F.binary_cross_entropy_with_logits(
            logits, torch.zeros_like(logits), reduction="none"
        ) * prob.pow(self.beta)

        pos_inds = torch.nonzero((labels >= 0) & (labels < num_classes), as_tuple=False).squeeze(-1)
        if pos_inds.numel() > 0:
            pos_labels = labels[pos_inds].long()
            quality = quality_scores[pos_inds].to(logits.dtype)
            gap = (quality - prob[pos_inds, pos_labels]).abs().pow(self.beta)
            loss[pos_inds, pos_labels] = (
                F.binary_cross_entropy_with_logits(
                    logits[pos_inds, pos_labels], quality, reduction="none"
                )
                * gap
            )
        if class_weights is not None:
            loss = loss * class_weights
        loss = loss.sum()
        if avg_factor is not None:
            loss = loss / max(avg_factor, 1.0)
        return self.loss_weight * loss


class GIoULoss(nn.Module):
    """GIoU loss between aligned unnormalized ``(x1, y1, x2, y2)`` boxes."""

    def __init__(self, loss_weight: float = 1.0) -> None:
        """Initialize the GIoU loss.

        Args:
            loss_weight: Multiplier applied to the final loss.
        """
        super().__init__()
        self.loss_weight = loss_weight

    def forward(
        self,
        pred_boxes: torch.Tensor,
        target_boxes: torch.Tensor,
        weights: torch.Tensor | None = None,
        avg_factor: float | None = None,
    ) -> torch.Tensor:
        """Compute the GIoU loss on aligned box pairs.

        Args:
            pred_boxes: Predicted boxes with shape ``(N, 4)``.
            target_boxes: Target boxes with shape ``(N, 4)``.
            weights: Optional per-box weights with shape ``(N,)`` or ``(N, 4)``.
            avg_factor: Optional normalization factor.

        Returns:
            Scalar loss value.
        """
        giou = bbox_overlaps(pred_boxes, target_boxes, mode="giou", is_aligned=True)
        loss = 1.0 - giou
        if weights is not None:
            if weights.dim() > 1:
                weights = weights.mean(dim=-1)
            loss = loss * weights
        loss = loss.sum()
        if avg_factor is not None:
            loss = loss / max(avg_factor, 1.0)
        return self.loss_weight * loss


class WeightedL1Loss(nn.Module):
    """Elementwise-weighted L1 loss normalized by an average factor."""

    def __init__(self, loss_weight: float = 1.0) -> None:
        """Initialize the weighted L1 loss.

        Args:
            loss_weight: Multiplier applied to the final loss.
        """
        super().__init__()
        self.loss_weight = loss_weight

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        weights: torch.Tensor | None = None,
        avg_factor: float | None = None,
    ) -> torch.Tensor:
        """Compute the weighted L1 loss.

        Args:
            prediction: Predicted values.
            target: Target values with the same shape.
            weights: Optional elementwise weights with the same shape.
            avg_factor: Optional normalization factor.

        Returns:
            Scalar loss value.
        """
        loss = (prediction - target).abs()
        if weights is not None:
            loss = loss * weights
        loss = loss.sum()
        if avg_factor is not None:
            loss = loss / max(avg_factor, 1.0)
        return self.loss_weight * loss


class HeatmapGaussianFocalLoss(nn.Module):
    """Gaussian focal loss on already-sigmoided probability heatmaps.

    The normalization factor is supplied by the caller (the auxiliary 2D head
    shares one positive count across its five losses).
    """

    def __init__(self, alpha: float = 2.0, gamma: float = 4.0, loss_weight: float = 1.0) -> None:
        """Initialize the heatmap Gaussian focal loss.

        Args:
            alpha: Focusing parameter shared by positive and negative terms.
            gamma: Modulating exponent on the negative Gaussian weights.
            loss_weight: Multiplier applied to the final loss.
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.loss_weight = loss_weight

    def forward(
        self,
        probabilities: torch.Tensor,
        target: torch.Tensor,
        avg_factor: float | None = None,
    ) -> torch.Tensor:
        """Compute the Gaussian focal loss on probability heatmaps.

        Args:
            probabilities: Sigmoid probabilities clipped away from 0 and 1.
            target: Gaussian heatmap targets in ``[0, 1]``.
            avg_factor: Optional normalization factor.

        Returns:
            Scalar loss value.
        """
        pos_weights = target.eq(1).to(probabilities.dtype)
        # neg_weights is exactly zero wherever target == 1.
        neg_weights = (1.0 - target).pow(self.gamma)
        pos_loss = -probabilities.log() * (1.0 - probabilities).pow(self.alpha) * pos_weights
        neg_loss = -(1.0 - probabilities).log() * probabilities.pow(self.alpha) * neg_weights
        loss = pos_loss.sum() + neg_loss.sum()
        if avg_factor is not None:
            loss = loss / max(avg_factor, 1.0)
        return self.loss_weight * loss
