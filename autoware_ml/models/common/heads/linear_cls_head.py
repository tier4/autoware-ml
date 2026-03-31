# Copyright 2025 TIER IV, Inc.
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

"""Classification heads for image-based models.

This module provides reusable classification heads built on top of image backbone features.
"""

from collections.abc import Sequence
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.functional import accuracy


class ClsHead(nn.Module):
    """Provide common loss and prediction utilities for classification heads.

    Subclasses define the concrete feature-to-logit mapping while inheriting
    shared loss, accuracy, and prediction behavior.
    """

    def __init__(
        self,
        loss: nn.Module,
        topk: Sequence[int],
        num_classes: int,
        cal_acc: bool = False,
    ):
        """Initialize a generic classification head.

        Args:
            loss: Loss module to use.
            topk: Top-k values for accuracy computation.
            num_classes: Number of classes.
            cal_acc: Whether to compute accuracy metrics.
        """
        super().__init__()
        self.topk = tuple(topk)
        self.cal_acc = cal_acc
        self.num_classes = num_classes
        self.loss_module = loss

    def pre_logits(self, feats: torch.Tensor) -> torch.Tensor:
        """Normalize backbone outputs before the final classification layer.

        Args:
            feats: Feature maps from backbone/neck. Can be a single tensor or tuple/list.
                  If tuple/list, takes the last element (deepest features).
                  Expected to be already flattened to (B, C) by neck if needed.

        Returns:
            Features ready for the final classification layer.
        """
        if isinstance(feats, (tuple, list)):
            assert len(feats) > 0, "Tuple/list input must not be empty"
            feats = feats[-1]
        return feats

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        """Return pre-logit features consumed by downstream classifier layers.

        Args:
            feats: Feature maps from the backbone or neck.

        Returns:
            Normalized feature tensor.
        """
        return self.pre_logits(feats)

    def loss(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        """Calculate losses from model logits.

        Args:
            logits: Model outputs (already passed through the head).
            target: Ground truth labels (B, ) or scores.
            **kwargs: Extra arguments for the loss module.

        Returns:
            Dictionary containing the classification loss and optional accuracy metrics.
        """
        losses: dict[str, torch.Tensor] = {}
        losses["loss"] = self.loss_module(logits, target, **kwargs)

        if self.cal_acc:
            for k in self.topk:
                acc_k = accuracy(
                    logits,
                    target,
                    task="multiclass",
                    num_classes=self.num_classes,
                    top_k=k,
                )
                losses[f"accuracy_top-{k}"] = acc_k

        return losses

    def predict(self, logits: torch.Tensor) -> torch.Tensor:
        """Convert logits to class probabilities.

        Args:
            logits: Model outputs (already passed through the head).

        Returns:
            Softmax probabilities.
        """
        return F.softmax(logits, dim=1)


class LinearClsHead(ClsHead):
    """Implement a linear classifier head on top of flattened features.

    The head applies one fully connected layer after the shared
    :class:`ClsHead` preprocessing step.
    """

    def __init__(
        self,
        num_classes: int,
        in_channels: int,
        loss: nn.Module,
        topk: Sequence[int],
        cal_acc: bool = False,
    ):
        """Initialize the linear classification head.

        Args:
            num_classes: Number of target classes.
            in_channels: Input feature dimension.
            loss: Loss module used for supervision.
            topk: Top-k values used for accuracy computation.
            cal_acc: Whether to compute accuracy metrics during training.
        """
        super().__init__(loss=loss, topk=topk, num_classes=num_classes, cal_acc=cal_acc)

        self.in_channels = in_channels
        self.num_classes = num_classes

        if self.num_classes <= 0:
            raise ValueError(f"num_classes={num_classes} must be a positive integer")

        self.fc = nn.Linear(self.in_channels, self.num_classes)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        """Map backbone features to classification logits.

        Args:
            feats: Feature maps from the backbone or neck.

        Returns:
            Classification logits.
        """
        pre_logits = self.pre_logits(feats)
        # The final classification head.
        cls_score = self.fc(pre_logits)
        return cls_score
