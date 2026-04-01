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

"""Lovasz losses for semantic segmentation.

This module contains Lovasz-based losses used by segmentation models.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss


def _lovasz_grad(sorted_ground_truth: torch.Tensor) -> torch.Tensor:
    """Compute the Lovasz extension gradient.

    Args:
        sorted_ground_truth: Ground-truth foreground mask sorted by prediction error.

    Returns:
        Per-sample gradient weights for the Lovasz extension.
    """
    point_count = len(sorted_ground_truth)
    foreground_total = sorted_ground_truth.sum()
    intersection = foreground_total - sorted_ground_truth.float().cumsum(0)
    union = foreground_total + (1 - sorted_ground_truth).float().cumsum(0)
    jaccard = 1.0 - intersection / union
    if point_count > 1:
        jaccard[1:point_count] = jaccard[1:point_count] - jaccard[0:-1]
    return jaccard


def _flatten_probabilities(
    probabilities: torch.Tensor, labels: torch.Tensor, ignore_index: int | None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Flatten dense segmentation probabilities and labels.

    Args:
        probabilities: Dense probabilities with the class dimension at index 1.
        labels: Dense integer labels.
        ignore_index: Label value to exclude from the loss.

    Returns:
        Tuple of flattened probabilities and labels.
    """
    class_count = probabilities.size(1)
    probabilities = torch.movedim(probabilities, 1, -1).contiguous().view(-1, class_count)
    labels = labels.view(-1)
    if ignore_index is None:
        return probabilities, labels
    valid = labels != ignore_index
    return probabilities[valid], labels[valid]


class LovaszLoss(_Loss):
    """Compute multi-class Lovasz-Softmax loss for segmentation.

    This wrapper aggregates the class-wise Lovasz loss used to optimize IoU-like
    objectives for semantic segmentation.
    """

    def __init__(
        self,
        ignore_index: int | None = None,
        per_image: bool = False,
        loss_weight: float = 1.0,
    ) -> None:
        """Initialize the Lovasz-Softmax loss.

        Args:
            ignore_index: Label value excluded from the loss.
            per_image: Whether to compute the loss independently per image.
            loss_weight: Scalar weight applied to the loss.
        """
        super().__init__()
        self.ignore_index = ignore_index
        self.per_image = per_image
        self.loss_weight = loss_weight

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute Lovasz-Softmax loss from logits.

        Args:
            logits: Dense segmentation logits.
            target: Dense integer labels.

        Returns:
            Scalar Lovasz-Softmax loss.
        """
        probabilities = logits.softmax(dim=1)
        if self.per_image:
            losses = [
                self._lovasz_softmax_flat(
                    *_flatten_probabilities(
                        prob.unsqueeze(0), label.unsqueeze(0), self.ignore_index
                    )
                )
                for prob, label in zip(probabilities, target)
            ]
            return self.loss_weight * torch.stack(losses).mean()
        return self.loss_weight * self._lovasz_softmax_flat(
            *_flatten_probabilities(probabilities, target, self.ignore_index)
        )

    def _lovasz_softmax_flat(
        self, probabilities: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """Compute Lovasz-Softmax loss on flattened tensors.

        Args:
            probabilities: Flattened class probabilities.
            labels: Flattened integer labels.

        Returns:
            Scalar Lovasz-Softmax loss.
        """
        if probabilities.numel() == 0:
            # Maintain the computational graph through probabilities so that
            # autograd does not lose the connection to upstream parameters.
            return (probabilities * 0.0).sum()

        losses = []
        for class_index in labels.unique():
            foreground = (labels == class_index).type_as(probabilities)
            if foreground.sum() == 0:
                continue
            class_errors = (foreground - probabilities[:, class_index]).abs()
            class_errors, permutation = torch.sort(class_errors, descending=True)
            foreground = foreground[permutation]
            losses.append(torch.dot(class_errors, _lovasz_grad(foreground)))
        return torch.stack(losses).mean()


class LovaszSoftmaxLoss(nn.Module):
    """Compute per-sample Lovasz-Softmax loss for dense segmentation logits.

    The module applies Lovasz-Softmax to one segmentation prediction tensor and
    its corresponding dense ground-truth labels.
    """

    def __init__(self, ignore_index: int, loss_weight: float = 1.0) -> None:
        """Initialize the dense Lovasz-Softmax loss wrapper.

        Args:
            ignore_index: Label value excluded from the loss.
            loss_weight: Scalar weight applied to the loss.
        """
        super().__init__()
        self.ignore_index = ignore_index
        self.loss_weight = loss_weight

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute Lovasz-Softmax loss on dense logits.

        Args:
            logits: Dense segmentation logits.
            target: Dense integer labels.

        Returns:
            Scalar Lovasz-Softmax loss.
        """
        probabilities = F.softmax(logits, dim=1)
        losses = []
        for prob_sample, target_sample in zip(probabilities, target):
            loss = self._lovasz_softmax_flat(
                prob_sample.permute(1, 2, 0).reshape(-1, prob_sample.shape[0]),
                target_sample.reshape(-1),
            )
            losses.append(loss)
        return self.loss_weight * torch.stack(losses).mean()

    def _lovasz_softmax_flat(
        self, probabilities: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """Compute Lovasz-Softmax loss on flattened tensors.

        Args:
            probabilities: Flattened class probabilities.
            labels: Flattened integer labels.

        Returns:
            Scalar Lovasz-Softmax loss.
        """
        valid_mask = labels != self.ignore_index
        probabilities = probabilities[valid_mask]
        labels = labels[valid_mask]
        if labels.numel() == 0:
            # Maintain the computational graph through probabilities so that
            # autograd does not lose the connection to upstream parameters.
            return (probabilities * 0.0).sum()

        class_losses = []
        for class_index in range(probabilities.shape[1]):
            foreground = (labels == class_index).float()
            if foreground.sum() == 0:
                continue
            class_errors = (foreground - probabilities[:, class_index]).abs()
            class_errors, permutation = torch.sort(class_errors, descending=True)
            foreground = foreground[permutation]
            class_losses.append(torch.dot(class_errors, _lovasz_grad(foreground)))

        if not class_losses:
            return (probabilities * 0.0).sum()
        return torch.stack(class_losses).mean()
