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

import unittest

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

    def identity(size: int) -> torch.Tensor:
        return torch.eye(size, device=images.device).expand(batch_size, num_cams, size, size)

    return build_batch_inputs(
        images=ImageGTBatch(
            images=images,
            depth_maps=None,
            camera_intrinsics=identity(3),
            image_augmentation_matrices=identity(4),
            lidar2images=identity(4),
            lidar2cams=identity(4),
            calibration_statuses=None,
        )
    )


class TestBatchNormalizeImages(unittest.TestCase):
    """Tests for BatchNormalizeImages."""

    def test_matches_the_runtime_normalization_of_8_bit_pixels(self) -> None:
        """Test that unit-range pixels and statistics give the runtime's (pixel - mean) / std."""
        pixels = torch.randint(0, 256, (2, 3, 3, 4, 5)).float()
        mean_255 = torch.tensor([123.675, 116.28, 103.53]).view(1, 1, 3, 1, 1)
        std_255 = torch.tensor([58.395, 57.12, 57.375]).view(1, 1, 3, 1, 1)

        for is_training in (True, False):
            with self.subTest(is_training=is_training):
                normalized = BatchNormalizeImages(IMAGENET_MEAN, IMAGENET_STD)(
                    _inputs(pixels / 255.0), is_training=is_training
                )
                torch.testing.assert_close(
                    normalized.image_data.images, (pixels - mean_255) / std_255
                )

    @unittest.skipUnless(torch.cuda.is_available(), "needs a CUDA device")
    def test_moves_the_statistics_to_the_device_once(self) -> None:
        """Test that after the first batch on a device no copy waits on that device again."""
        normalize = BatchNormalizeImages(IMAGENET_MEAN, IMAGENET_STD)
        inputs = _inputs(torch.rand(1, 2, 3, 4, 5, device="cuda"))
        normalize(inputs, is_training=True)
        torch.cuda.synchronize()

        torch.cuda.set_sync_debug_mode("error")
        try:
            normalized = normalize(inputs, is_training=True)
        finally:
            torch.cuda.set_sync_debug_mode("default")

        mean = torch.tensor(IMAGENET_MEAN, device="cuda").view(1, 1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD, device="cuda").view(1, 1, 3, 1, 1)
        torch.testing.assert_close(
            normalized.image_data.images, (inputs.image_data.images - mean) / std
        )

    def test_keeps_the_calibration(self) -> None:
        """Test that only the images are replaced; the calibration reaches the model as it is."""
        inputs = _inputs(torch.rand(1, 2, 3, 4, 5))

        normalized = BatchNormalizeImages(IMAGENET_MEAN, IMAGENET_STD)(inputs, is_training=True)

        self.assertIs(normalized.image_data.lidar2images, inputs.image_data.lidar2images)

    def test_rejects_images_of_another_channel_count(self) -> None:
        """Test that statistics of three channels cannot standardize images of four."""
        with self.assertRaisesRegex(ValueError, "channels"):
            BatchNormalizeImages(IMAGENET_MEAN, IMAGENET_STD)(
                _inputs(torch.rand(1, 2, 4, 4, 5)), is_training=True
            )

    def test_rejects_statistics_that_cannot_standardize_unit_range_images(self) -> None:
        """Test that unusable statistics are refused when the layer is built."""
        cases = {
            "channel-count": ([0.5, 0.5], IMAGENET_STD),
            # The 0-255 ImageNet statistics of the mmdetection configs would turn unit-range
            # images into a near-constant -2.
            "0-255": ([123.675, 116.28, 103.53], [58.395, 57.12, 57.375]),
            "zero-std": (IMAGENET_MEAN, [0.2, 0.0, 0.2]),
            "nan-std": (IMAGENET_MEAN, [0.2, float("nan"), 0.2]),
            "inf-mean": ([0.5, float("inf"), 0.5], IMAGENET_STD),
        }
        for name, (mean, std) in cases.items():
            with self.subTest(name), self.assertRaises(ValueError):
                BatchNormalizeImages(mean, std)

    def test_runs_inside_the_data_preprocessor(self) -> None:
        """Test that DataPreprocessor accepts the layer as a DataPreprocessorModule."""
        preprocessor = DataPreprocessor(
            preprocessor_modules=[BatchNormalizeImages(mean=[0.5] * 3, std=[0.5] * 3)]
        )
        self.assertEqual(len(preprocessor.preprocessor_modules), 1)


if __name__ == "__main__":
    unittest.main()
