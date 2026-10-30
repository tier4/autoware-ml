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
        avg_factor: float | None = None,
    ) -> torch.Tensor:
        """Compute Gaussian focal loss on dense heatmaps.

        Args:
            prediction: Raw heatmap logits of any float dtype; the loss is computed in float32.
            target: Gaussian heatmap targets.
            weights: Optional weights broadcastable to ``prediction``, for example one weight
                per sample and class. A zero drops the cell from the loss and, if it is a peak,
                from the positive count normalizing the loss.
            avg_factor: Normalizer of the summed loss given by the caller and used as given,
                for example a positive count shared with other losses, which averaged over the
                DDP ranks can be below 1. None divides by the weighted number of peaks, at
                least 1.

        Returns:
            Scalar heatmap loss value.
        """
        # In float32: bfloat16 rounds the clamp bound 1 - 1e-4 to 1, so under bf16 autocast a
        # confident logit would make log(1 - prediction) infinite and the gradients NaN.
        prediction = prediction.float().sigmoid().clamp(min=1e-4, max=1 - 1e-4)
        pos_mask = target.eq(1).float()
        neg_mask = target.lt(1).float()
        if weights is not None:
            pos_mask = pos_mask * weights
            neg_mask = neg_mask * weights
        neg_weights = (1 - target).pow(self.beta)

        # The masks carry the weights, so every cell is weighted once and a zero weighted peak
        # leaves the positive count normalizing the loss.
        pos_loss = -torch.log(prediction) * (1 - prediction).pow(self.alpha) * pos_mask
        neg_loss = -torch.log(1 - prediction) * prediction.pow(self.alpha) * neg_weights * neg_mask
        normalizer = pos_mask.sum().clamp_min(1) if avg_factor is None else avg_factor
        return (pos_loss.sum() + neg_loss.sum()) / normalizer
