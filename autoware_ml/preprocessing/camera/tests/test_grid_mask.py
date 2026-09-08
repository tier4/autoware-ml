"""Unit tests for the batched grid-mask preprocessing layer."""

from __future__ import annotations

import torch

from autoware_ml.preprocessing.camera.grid_mask import BatchGridMask


def _batch(batch_size: int = 2, num_cams: int = 3) -> list[torch.Tensor]:
    return [torch.ones(num_cams, 3, 32, 32) for _ in range(batch_size)]


def test_eval_mode_is_a_no_op() -> None:
    batch = {"img": _batch()}
    assert BatchGridMask(prob=1.0)(batch, is_training=False) == {}


def test_probability_zero_is_a_no_op() -> None:
    batch = {"img": _batch()}
    assert BatchGridMask(prob=0.0)(batch, is_training=True) == {}


def test_training_masks_part_of_every_image_and_keeps_shape() -> None:
    torch.manual_seed(0)
    images = torch.ones(6, 3, 32, 32)
    masked = BatchGridMask(prob=1.0, rotate=1).mask_images(images)

    assert masked.shape == images.shape
    # Some pixels are zeroed and some survive: a grid, not a blanket.
    assert (masked == 0).any()
    assert (masked == 1).any()


def test_rotate_zero_disables_rotation_without_error() -> None:
    torch.manual_seed(0)
    images = torch.ones(2, 3, 32, 32)
    masked = BatchGridMask(prob=1.0, rotate=0).mask_images(images)

    assert masked.shape == images.shape
    assert (masked == 0).any()
    assert (masked == 1).any()


def test_list_input_returns_list_with_shared_mask() -> None:
    batch = {"img": _batch(batch_size=2, num_cams=3)}
    out = BatchGridMask(prob=1.0)(batch, is_training=True)

    assert isinstance(out["img"], list)
    assert len(out["img"]) == 2
    assert out["img"][0].shape == (3, 3, 32, 32)
    # One grid is shared by every image in the batch.
    assert torch.equal(out["img"][0][0], out["img"][1][2])


def test_tensor_input_returns_tensor() -> None:
    batch = {"img": torch.ones(2, 3, 3, 32, 32)}
    out = BatchGridMask(prob=1.0)(batch, is_training=True)

    assert isinstance(out["img"], torch.Tensor)
    assert out["img"].shape == (2, 3, 3, 32, 32)
