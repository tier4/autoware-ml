"""Focal classification losses used by detection3d models.

This module provides reusable focal-loss variants for sparse and query-based
3D detection heads.
"""

from __future__ import annotations

from jaxtyping import Float32
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
        logits: Float32[torch.Tensor, "num_queries num_classes"],
        targets: Float32[torch.Tensor, "num_queries num_classes"],
        weights: Float32[torch.Tensor, "num_queries num_classes"] | None = None,
        avg_factor: float | None = None,
    ) -> Float32[torch.Tensor, ""]:
        """Compute focal loss on one-hot classification targets.

        Args:
            logits: Raw classification logits with shape ``(N, C)``, float32: the focusing
                term ``(1 - p_t)^gamma`` of a confident query rounds to 0 in half precision,
                so a head training under autocast casts its logits first.
            targets: One-hot classification targets with shape ``(N, C)``.
            weights: Optional per-query, per-class weights with shape ``(N, C)``.
            avg_factor: Optional normalizer of the summed loss, used as given. The caller
                floors it: a positive count averaged over the DDP ranks can be below 1, and
                flooring it again on every rank would change the gradient of the step.

        Returns:
            Scalar focal loss value.

        Raises:
            ValueError: If ``logits`` is not float32.
        """
        if logits.dtype != torch.float32:
            raise ValueError(
                f"SigmoidFocalLoss takes float32 logits, got {logits.dtype}; cast them in the "
                "head, under autocast its logits are half precision."
            )
        prob = logits.sigmoid()
        ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p_t = prob * targets + (1.0 - prob) * (1.0 - targets)
        alpha_factor = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
        loss = ce * alpha_factor * (1.0 - p_t).pow(self.gamma)
        if weights is not None:
            loss = loss * weights
        loss = loss.sum()
        if avg_factor is not None:
            loss = loss / avg_factor
        return loss
