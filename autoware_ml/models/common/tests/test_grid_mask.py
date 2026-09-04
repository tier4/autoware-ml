"""Unit tests for the grid-mask image augmentation."""

from __future__ import annotations

import torch

from autoware_ml.models.common.grid_mask import GridMask


def test_eval_mode_passes_images_through_unchanged() -> None:
    images = torch.ones(2, 3, 32, 32)
    masked = GridMask(prob=1.0).eval()(images)

    assert torch.equal(masked, images)


def test_probability_zero_passes_images_through_unchanged() -> None:
    images = torch.ones(2, 3, 32, 32)
    masked = GridMask(prob=0.0).train()(images)

    assert torch.equal(masked, images)


def test_training_masks_part_of_every_image_and_keeps_shape() -> None:
    torch.manual_seed(0)
    images = torch.ones(2, 3, 32, 32)
    masked = GridMask(prob=1.0, rotate=1).train()(images)

    assert masked.shape == images.shape
    # Some pixels are zeroed and some survive: a grid, not a blanket.
    assert (masked == 0).any()
    assert (masked == 1).any()
