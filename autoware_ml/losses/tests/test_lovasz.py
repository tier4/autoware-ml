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

"""Tests for the point-wise Lovasz-Softmax loss."""

import pytest
import torch

from autoware_ml.losses.segmentation3d.lovasz import LovaszLoss, _lovasz_grad

IGNORE = -1


def _reference_lovasz_softmax_flat(
    probabilities: torch.Tensor, labels: torch.Tensor
) -> torch.Tensor:
    """The original class loop, kept as the reference for the loss."""
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


def _random_batch(seed: int, num_points: int = 600, num_classes: int = 15):
    generator = torch.Generator().manual_seed(seed)
    logits = torch.randn(num_points, num_classes, generator=generator)
    labels = torch.randint(0, num_classes, (num_points,), generator=generator)
    labels[torch.rand(num_points, generator=generator) < 0.2] = IGNORE
    labels[labels == 3] = 4  # a class that is absent from the batch
    return logits, labels


@pytest.mark.parametrize("seed", range(5))
def test_lovasz_loss_matches_the_reference_class_loop(seed: int) -> None:
    """Loss value and gradient must equal the per-class reference, with ignored
    points excluded and absent classes contributing nothing."""
    logits, labels = _random_batch(seed)
    logits = logits.requires_grad_(True)
    loss = LovaszLoss(ignore_index=IGNORE)(logits, labels)
    (grad,) = torch.autograd.grad(loss, logits)

    ref_logits = logits.detach().requires_grad_(True)
    valid = labels != IGNORE
    ref_loss = _reference_lovasz_softmax_flat(ref_logits.softmax(dim=1)[valid], labels[valid])
    (ref_grad,) = torch.autograd.grad(ref_loss, ref_logits)

    assert torch.allclose(loss, ref_loss, rtol=1e-6, atol=0)
    assert torch.allclose(grad, ref_grad, rtol=1e-5, atol=1e-8)
    assert torch.all(grad[~valid] == 0)


def test_lovasz_loss_averages_over_present_classes_only() -> None:
    """With a single present class the loss is that class's Lovasz term alone."""
    logits, _ = _random_batch(0, num_points=64, num_classes=4)
    labels = torch.full((64,), 2)
    loss = LovaszLoss(ignore_index=IGNORE)(logits, labels)
    probabilities = logits.softmax(dim=1)
    errors, permutation = torch.sort((1.0 - probabilities[:, 2]).abs(), descending=True)
    expected = torch.dot(errors, _lovasz_grad(torch.ones(64)[permutation]))
    assert torch.allclose(loss, expected)


def test_lovasz_loss_keeps_the_graph_when_everything_is_ignored() -> None:
    """An all-ignored batch yields a zero loss that still differentiates to zeros."""
    logits, _ = _random_batch(1, num_points=32)
    logits = logits.requires_grad_(True)
    loss = LovaszLoss(ignore_index=IGNORE)(logits, torch.full((32,), IGNORE))
    (grad,) = torch.autograd.grad(loss, logits)
    assert loss.item() == 0.0
    assert torch.all(grad == 0)


def test_lovasz_loss_applies_the_weight_and_per_image_mean() -> None:
    """``per_image`` averages the per-sample losses; ``loss_weight`` scales the result."""
    logits, labels = _random_batch(2, num_points=200)
    joint = LovaszLoss(ignore_index=IGNORE)
    per_sample = torch.stack([joint(logits[i : i + 100], labels[i : i + 100]) for i in (0, 100)])
    per_image = LovaszLoss(ignore_index=IGNORE, per_image=True, loss_weight=0.5)(
        logits.view(2, 100, -1).movedim(-1, 1), labels.view(2, 100)
    )
    assert torch.allclose(per_image, 0.5 * per_sample.mean())
