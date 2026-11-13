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

"""Quality focal classification loss of the auxiliary 2D head of camera 3D detectors."""

from __future__ import annotations

from jaxtyping import Float32, Int
import torch
import torch.nn as nn
import torch.nn.functional as F


class QualityFocalLoss(nn.Module):
    """Quality focal loss over sigmoid logits with IoU-quality targets.

    A positive is supervised toward the IoU between its predicted and its assigned box
    instead of a hard 1, following Generalized Focal Loss: its class term is
    ``BCE(logit, q) * |q - sigmoid(logit)|^beta``, every other term ``BCE(logit, 0) *
    sigmoid(logit)^beta``.

    The loss takes float32 logits and qualities: in bfloat16 the gap ``|q - sigmoid(logit)|``
    of a positive scored within 1/256 of its quality rounds to 0 and the positive gets no
    gradient, so a head training under autocast casts its logits first.
    """

    def __init__(self, beta: float = 2.0) -> None:
        """Initialize the quality focal loss.

        Args:
            beta: Modulating exponent on the quality gap.
        """
        super().__init__()
        self.beta = beta

    def forward(
        self,
        logits: Float32[torch.Tensor, "num_queries num_classes"],
        labels: Int[torch.Tensor, " num_queries"],
        quality_scores: Float32[torch.Tensor, " num_queries"],
        avg_factor: float,
        weights: Float32[torch.Tensor, "num_queries num_classes"] | None = None,
    ) -> Float32[torch.Tensor, ""]:
        """Compute the quality focal loss.

        Args:
            logits: Classification logits of every query.
            labels: Class index of every query; ``num_classes`` or a negative value marks
                background.
            quality_scores: IoU quality target of every query, read at the positives only.
                The caller detaches it: the IoU of the predicted box is a target, not a path
                for the gradient of the box.
            avg_factor: Normalizer of the summed loss, used as given. The caller floors it: a
                positive count averaged over the DDP ranks can be below 1, and flooring it
                again on every rank would change the gradient of the step.
            weights: Optional per-query, per-class weights, for example 0 on the classes a
                sample is not annotated with.

        Returns:
            Scalar loss value.

        Raises:
            ValueError: If ``logits`` or ``quality_scores`` is not float32.
        """
        for name, tensor in (("logits", logits), ("quality_scores", quality_scores)):
            if tensor.dtype != torch.float32:
                raise ValueError(
                    f"QualityFocalLoss takes float32 {name}, got {tensor.dtype}; cast them in "
                    "the head, under autocast its logits are half precision."
                )
        num_classes = logits.size(1)
        prob = logits.sigmoid()
        # Every term starts as background: target 0 modulated by sigmoid(logit)^beta.
        loss = F.binary_cross_entropy_with_logits(
            logits, torch.zeros_like(logits), reduction="none"
        ) * prob.pow(self.beta)

        positive = (labels >= 0) & (labels < num_classes)
        pos_inds = torch.nonzero(positive, as_tuple=False).squeeze(-1)
        if pos_inds.numel() > 0:
            pos_labels = labels[pos_inds].long()
            quality = quality_scores[pos_inds]
            gap = (quality - prob[pos_inds, pos_labels]).abs().pow(self.beta)
            loss[pos_inds, pos_labels] = (
                F.binary_cross_entropy_with_logits(
                    logits[pos_inds, pos_labels], quality, reduction="none"
                )
                * gap
            )
        if weights is not None:
            loss = loss * weights
        return loss.sum() / avg_factor
