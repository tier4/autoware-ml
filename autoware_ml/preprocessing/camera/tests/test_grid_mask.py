"""Unit tests for the batched grid-mask preprocessing layer."""

from __future__ import annotations

import torch

from autoware_ml.dataclasses.geometry.images import ImageGTBatch
from autoware_ml.dataclasses.models.model_batch_inputs import ModelBatchInputs
from autoware_ml.models.tests.batch_inputs_fixtures import build_batch_inputs
from autoware_ml.preprocessing.camera.grid_mask import BatchGridMask


def _inputs(batch_size: int = 2, num_cams: int = 3) -> ModelBatchInputs:
    images = ImageGTBatch(
        images=torch.ones(batch_size, num_cams, 3, 32, 32),
        depth_maps=None,
        camera_intrinsics=torch.eye(3).expand(batch_size, num_cams, 3, 3),
        image_augmentation_matrices=torch.eye(4).expand(batch_size, num_cams, 4, 4),
        lidar2images=torch.eye(4).expand(batch_size, num_cams, 4, 4),
        lidar2cams=torch.eye(4).expand(batch_size, num_cams, 4, 4),
        calibration_statuses=None,
    )
    return build_batch_inputs(images=images)


def test_eval_mode_is_a_no_op() -> None:
    inputs = _inputs()
    assert BatchGridMask(prob=1.0)(inputs, is_training=False) is inputs


def test_probability_zero_is_a_no_op() -> None:
    inputs = _inputs()
    assert BatchGridMask(prob=0.0)(inputs, is_training=True) is inputs


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


def test_training_replaces_the_images_with_one_shared_mask() -> None:
    inputs = _inputs(batch_size=2, num_cams=3)

    outputs = BatchGridMask(prob=1.0)(inputs, is_training=True)

    assert outputs.image_data is not None
    masked = outputs.image_data.images
    assert masked.shape == (2, 3, 3, 32, 32)
    # One grid is shared by every image in the batch.
    assert torch.equal(masked[0, 0], masked[1, 2])
    assert outputs.multi_task_gt_batch is inputs.multi_task_gt_batch
