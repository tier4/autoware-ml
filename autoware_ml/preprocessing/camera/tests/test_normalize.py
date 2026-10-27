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

"""Unit tests for the batched image normalization layer."""

from __future__ import annotations

import pytest
import torch

from autoware_ml.dataclasses.geometry.images import ImageGTBatch
from autoware_ml.dataclasses.models.model_batch_inputs import ModelBatchInputs
from autoware_ml.models.tests.batch_inputs_fixtures import build_batch_inputs
from autoware_ml.preprocessing.camera.normalize import BatchNormalizeImages
from autoware_ml.preprocessing.data_preprocessor import DataPreprocessor

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def _inputs(images: torch.Tensor) -> ModelBatchInputs:
    batch_size, num_cams = images.shape[:2]
    return build_batch_inputs(
        images=ImageGTBatch(
            images=images,
            depth_maps=None,
            camera_intrinsics=torch.eye(3).expand(batch_size, num_cams, 3, 3),
            image_augmentation_matrices=torch.eye(4).expand(batch_size, num_cams, 4, 4),
            lidar2images=torch.eye(4).expand(batch_size, num_cams, 4, 4),
            lidar2cams=torch.eye(4).expand(batch_size, num_cams, 4, 4),
            calibration_statuses=None,
        )
    )


@pytest.mark.parametrize("is_training", [True, False])
def test_matches_the_runtime_normalization_of_8_bit_pixels(is_training: bool) -> None:
    """
    Input: random 8-bit pixels divided by 255, in training and in evaluation.
    Expected: the unit-range statistics standardize the images exactly as the runtime does,
    which feeds the graph (pixel - mean) / std with 0-255 pixels and statistics.
    Check: the normalized images against the runtime formula on the 8-bit pixels.
    """
    pixels = torch.randint(0, 256, (2, 3, 3, 4, 5)).float()
    mean_255 = torch.tensor([123.675, 116.28, 103.53]).view(1, 1, 3, 1, 1)
    std_255 = torch.tensor([58.395, 57.12, 57.375]).view(1, 1, 3, 1, 1)

    normalized = BatchNormalizeImages(IMAGENET_MEAN, IMAGENET_STD)(
        _inputs(pixels / 255.0), is_training=is_training
    )

    torch.testing.assert_close(normalized.image_data.images, (pixels - mean_255) / std_255)


@pytest.mark.parametrize("is_training", [True, False])
def test_refuses_images_in_0_255(is_training: bool) -> None:
    """
    Input: images of 0-255 pixels, as LoadImagesFromFile(normalize_to_unit=False) serves them,
    in training and in evaluation.
    Expected: the unit-range statistics would standardize them without any error into values far
    from the ones the models read, so the layer refuses them.
    Check: a ValueError naming LoadImagesFromFile(normalize_to_unit=True) is raised.
    """
    pixels = torch.randint(0, 256, (2, 3, 3, 4, 5), generator=torch.Generator().manual_seed(0))

    with pytest.raises(ValueError, match=r"LoadImagesFromFile\(normalize_to_unit=True\)"):
        BatchNormalizeImages(IMAGENET_MEAN, IMAGENET_STD)(
            _inputs(pixels.float()), is_training=is_training
        )


def test_accepts_unit_range_images_with_interpolation_overshoot() -> None:
    """
    Input: unit-range images with one pixel at 1.2, an overshoot an image interpolation can
    produce.
    Expected: the range check leaves room for the overshoot of an interpolation, so the images
    are normalized as usual.
    Check: the normalized images against (images - mean) / std.
    """
    images = torch.rand(2, 3, 3, 4, 5, generator=torch.Generator().manual_seed(0))
    images[0, 0, 0, 0, 0] = 1.2

    normalized = BatchNormalizeImages(IMAGENET_MEAN, IMAGENET_STD)(
        _inputs(images), is_training=True
    )

    mean = torch.tensor(IMAGENET_MEAN).view(1, 1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(1, 1, 3, 1, 1)
    torch.testing.assert_close(normalized.image_data.images, (images - mean) / std)


def test_checks_only_the_first_batch() -> None:
    """
    Input: a unit-range batch, then a batch of 255 pixels, through the same layer.
    Expected: the layer reads the pixel range of the first batch only, one reduction and one
    host synchronization per run, so it does not check a later batch.
    Check: the second batch is normalized without an error.
    """
    normalize = BatchNormalizeImages(IMAGENET_MEAN, IMAGENET_STD)
    normalize(_inputs(torch.rand(1, 2, 3, 4, 5)), is_training=True)

    normalize(_inputs(torch.full((1, 2, 3, 4, 5), 255.0)), is_training=True)


def test_keeps_the_calibration() -> None:
    """
    Input: a unit-range batch with its calibration.
    Expected: the normalization replaces the images only, so the calibration reaches the model
    as it is.
    Check: the lidar-to-image transforms are the same tensor.
    """
    inputs = _inputs(torch.rand(1, 2, 3, 4, 5))

    normalized = BatchNormalizeImages(IMAGENET_MEAN, IMAGENET_STD)(inputs, is_training=True)

    assert normalized.image_data.lidar2images is inputs.image_data.lidar2images


def test_rejects_images_of_another_channel_count() -> None:
    """
    Input: images of four channels with statistics of three.
    Expected: the statistics cannot standardize channels they do not describe.
    Check: a ValueError naming the channels is raised.
    """
    with pytest.raises(ValueError, match="channels"):
        BatchNormalizeImages(IMAGENET_MEAN, IMAGENET_STD)(
            _inputs(torch.rand(1, 2, 4, 4, 5)), is_training=True
        )


def test_rejects_inconsistent_statistics() -> None:
    """
    Input: a mean of two channels with a standard deviation of three, and a standard deviation
    of zero.
    Expected: statistics that disagree on the channel count, or that would divide by a
    non-positive deviation, cannot standardize the images, so the layer refuses them when it is
    built.
    Check: a ValueError is raised for each.
    """
    with pytest.raises(ValueError):
        BatchNormalizeImages([0.5, 0.5], IMAGENET_STD)
    with pytest.raises(ValueError):
        BatchNormalizeImages(IMAGENET_MEAN, [0.2, 0.0, 0.2])


def test_runs_inside_the_data_preprocessor() -> None:
    """
    Input: a DataPreprocessor built with the normalization as its only module.
    Expected: DataPreprocessor accepts only DataPreprocessorModule instances, so the
    normalization has to be one to sit first in ``data_preprocessor.preprocessor_modules``.
    Check: the preprocessor holds the module.
    """
    preprocessor = DataPreprocessor(
        preprocessor_modules=[BatchNormalizeImages(mean=[0.5] * 3, std=[0.5] * 3)]
    )
    assert len(preprocessor.preprocessor_modules) == 1
