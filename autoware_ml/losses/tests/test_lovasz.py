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

"""Equivalence tests for the Lovasz losses after the host side class loops."""

from __future__ import annotations

import torch

from autoware_ml.losses.segmentation3d.lovasz import (
    LovaszLoss,
    LovaszSoftmaxLoss,
    _flatten_probabilities,
    _lovasz_grad,
)


def _reference_flat(probabilities: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """The per class loop over ``labels.unique()`` as it was before, kept verbatim."""
    if probabilities.numel() == 0:
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


def _reference_dense_flat(
    probabilities: torch.Tensor, labels: torch.Tensor, ignore_index: int
) -> torch.Tensor:
    """The ``range(num_classes)`` loop of the dense loss as it was before."""
    valid_mask = labels != ignore_index
    probabilities = probabilities[valid_mask]
    labels = labels[valid_mask]
    if labels.numel() == 0:
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
    return torch.stack(class_losses).mean()


def test_point_lovasz_matches_the_reference_loop() -> None:
    generator = torch.Generator().manual_seed(7)
    num_classes, ignore_index = 6, -1
    loss = LovaszLoss(ignore_index=ignore_index, loss_weight=1.0)
    for _ in range(20):
        logits = torch.randn(2, num_classes, 300, generator=generator)
        # Only a subset of the classes is present, and some points are ignored.
        labels = torch.randint(0, 3, (2, 300), generator=generator)
        labels[torch.rand(2, 300, generator=generator) < 0.2] = ignore_index
        # The default loss flattens the whole batch into one class loop.
        expected = _reference_flat(
            *_flatten_probabilities(logits.softmax(dim=1), labels, ignore_index)
        )
        torch.testing.assert_close(loss(logits, labels), expected)


def test_point_lovasz_keeps_the_graph_when_everything_is_ignored() -> None:
    loss = LovaszLoss(ignore_index=-1, loss_weight=1.0)
    logits = torch.randn(1, 4, 10, requires_grad=True)
    labels = torch.full((1, 10), -1)
    value = loss(logits, labels)
    value.backward()
    assert value.item() == 0.0
    assert logits.grad is not None


def test_dense_lovasz_matches_the_reference_loop() -> None:
    generator = torch.Generator().manual_seed(11)
    num_classes, ignore_index = 5, 255
    loss = LovaszSoftmaxLoss(ignore_index=ignore_index, loss_weight=2.0)
    for _ in range(10):
        logits = torch.randn(2, num_classes, 8, 9, generator=generator)
        labels = torch.randint(0, 3, (2, 8, 9), generator=generator)
        labels[torch.rand(2, 8, 9, generator=generator) < 0.3] = ignore_index
        expected = (
            torch.stack(
                [
                    _reference_dense_flat(
                        prob.permute(1, 2, 0).reshape(-1, num_classes),
                        label.reshape(-1),
                        ignore_index,
                    )
                    for prob, label in zip(logits.softmax(dim=1), labels)
                ]
            ).mean()
            * 2.0
        )
        torch.testing.assert_close(loss(logits, labels), expected)
