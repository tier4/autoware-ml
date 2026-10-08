"""Center-based focal loss implementations for 3D detection.

This module provides dense heatmap focal losses used by CenterPoint-style
detectors.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class GaussianFocalLoss(nn.Module):
    """Compute Gaussian focal loss for dense CenterPoint heatmaps.

    The loss treats heatmap peaks as positives and modulates surrounding
    negatives according to the Gaussian target value.
    """

    def __init__(self, alpha: float = 2.0, beta: float = 4.0) -> None:
        """Initialize the Gaussian focal loss.

        Args:
            alpha: Focusing parameter for positive samples.
            beta: Modulating exponent for negative samples.
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute Gaussian focal loss on dense heatmaps.

        Args:
            prediction: Raw heatmap logits.
            target: Gaussian heatmap targets.
            weights: Optional weights broadcastable to the heatmap, scaling the loss of every
                cell, for example one weight per sample and class.

        Returns:
            Scalar heatmap loss value.
        """
        prediction = prediction.sigmoid().clamp(min=1e-4, max=1 - 1e-4)
        pos_mask = target.eq(1).float()
        neg_mask = target.lt(1).float()
        neg_weights = (1 - target).pow(self.beta)

        pos_loss = -torch.log(prediction) * (1 - prediction).pow(self.alpha) * pos_mask
        neg_loss = -torch.log(1 - prediction) * prediction.pow(self.alpha) * neg_weights * neg_mask
        if weights is not None:
            pos_loss = pos_loss * weights
            neg_loss = neg_loss * weights
        return (pos_loss.sum() + neg_loss.sum()) / pos_mask.sum().clamp_min(1)
