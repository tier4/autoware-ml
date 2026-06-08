"""Focal classification losses used by detection3d models.

This module provides reusable focal-loss variants for sparse and query-based
3D detection heads.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SigmoidFocalLoss(nn.Module):
    """Compute sigmoid focal loss for sparse query classification.

    This loss is used by query-based detection heads where class targets are
    represented as one-hot vectors over a sparse proposal set.
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 0.25) -> None:
        """Initialize the sigmoid focal loss.

        Args:
            gamma: Focusing parameter.
            alpha: Positive class balancing factor.
        """
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        weights: torch.Tensor | None = None,
        avg_factor: float | None = None,
    ) -> torch.Tensor:
        """Compute focal loss on one-hot classification targets.

        Args:
            logits: Raw classification logits with shape ``(N, C)``.
            targets: One-hot classification targets with shape ``(N, C)``.
            weights: Optional per-query weights with shape ``(N,)``.
            avg_factor: Optional normalization factor.

        Returns:
            Scalar focal loss value.
        """
        prob = logits.sigmoid()
        ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p_t = prob * targets + (1.0 - prob) * (1.0 - targets)
        alpha_factor = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
        loss = ce * alpha_factor * (1.0 - p_t).pow(self.gamma)
        if weights is not None:
            loss = loss * weights.unsqueeze(-1)
        loss = loss.sum()
        if avg_factor is not None:
            loss = loss / max(avg_factor, 1.0)
        return loss
