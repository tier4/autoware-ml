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

from typing import Any, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.functional import accuracy


class ClsHead(nn.Module):
    """Base class for classification heads."""

    def __init__(
        self,
        loss: nn.Module,
        topk: List[int],
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
        self.topk = topk
        self.cal_acc = cal_acc
        self.num_classes = num_classes
        self.loss_module = loss

    def pre_logits(self, feats: torch.Tensor) -> torch.Tensor:
        """
        The process before the final classification head.
        Handles both single Tensor inputs and Tuple/List inputs (from backbones/necks).

        Args:
            feats: Feature maps from backbone/neck. Can be a single tensor or tuple/list.
                  If tuple/list, takes the last element (deepest features).
                  Expected to be already flattened to (B, C) by neck if needed.

        Returns:
            torch.Tensor: Features ready for classification head.
        """
        if isinstance(feats, (tuple, list)):
            assert len(feats) > 0, "Tuple/list input must not be empty"
            feats = feats[-1]
        return feats

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        """The forward process."""
        return self.pre_logits(feats)

    def loss(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        **kwargs: Any,
    ) -> Dict[str, torch.Tensor]:
        """
        Calculate losses from model logits.

        Args:
            logits: Model outputs (already passed through the head).
            target: Ground truth labels (B, ) or scores.
            **kwargs: Extra arguments for the loss module.

        Returns:
            Dict containing the loss and optionally accuracy.
        """
        losses: Dict[str, torch.Tensor] = {}
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
        """
        Inference method.

        Args:
            logits: Model outputs (already passed through the head).

        Returns:
            torch.Tensor: Softmax probabilities.
        """
        return F.softmax(logits, dim=1)


class LinearClsHead(ClsHead):
    """
    Linear classifier head.

    Args:
        num_classes (int): Number of categories.
        in_channels (int): Number of channels in the input feature map.
        loss (nn.Module): Loss module.
        topk (list): Top-k accuracy.
        cal_acc (bool): Calculate accuracy during training.
    """

    def __init__(
        self,
        num_classes: int,
        in_channels: int,
        loss: nn.Module,
        topk: List[int],
        cal_acc: bool = False,
    ):
        super().__init__(loss=loss, topk=topk, num_classes=num_classes, cal_acc=cal_acc)

        self.in_channels = in_channels
        self.num_classes = num_classes

        if self.num_classes <= 0:
            raise ValueError(f"num_classes={num_classes} must be a positive integer")

        self.fc = nn.Linear(self.in_channels, self.num_classes)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        """The forward process."""
        pre_logits = self.pre_logits(feats)
        # The final classification head.
        cls_score = self.fc(pre_logits)
        return cls_score
