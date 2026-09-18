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

"""Unit tests for ``ModelBatchInputs``."""

from __future__ import annotations

import unittest

from pydantic import ValidationError
import torch

from autoware_ml.dataclasses.batch.sample_batch import ModelGTBatch
from autoware_ml.dataclasses.geometry.images import ImageGTBatch
from autoware_ml.dataclasses.models.model_batch_inputs import ModelBatchInputs
from autoware_ml.ops.voxelization.voxelization import VoxelsData


class TestModelBatchInputs(unittest.TestCase):
    """Unit tests for the post-preprocessing model input payload."""

    def setUp(self) -> None:
        """Set up an empty GT batch and small voxel and image payloads."""
        self.gt_batch = ModelGTBatch(
            point_cloud_gt_batch=None, detection3d_gt_batch=None, image_gt_batch=None
        )
        num_voxels, max_points, channels = 4, 5, 4
        self.voxels_data = VoxelsData(
            voxels=torch.zeros(num_voxels, max_points, channels, dtype=torch.float32),
            coords=torch.zeros(num_voxels, 3, dtype=torch.int32),
            num_points=torch.ones(num_voxels, dtype=torch.int32),
            batch_indices=torch.zeros(num_voxels, dtype=torch.int32),
        )
        batch_size, num_cameras = 2, 3
        self.image_data = ImageGTBatch(
            images=torch.zeros(batch_size, num_cameras, 3, 8, 8),
            depth_maps=None,
            camera_intrinsics=torch.eye(3).expand(batch_size, num_cameras, 3, 3),
            image_augmentation_matrices=torch.eye(4).expand(batch_size, num_cameras, 4, 4),
            lidar2images=torch.eye(4).expand(batch_size, num_cameras, 4, 4),
            lidar2cams=torch.eye(4).expand(batch_size, num_cameras, 4, 4),
        )

    def test_lidar_only_inputs_leave_images_absent(self) -> None:
        """Test that a lidar-only payload keeps the GT batch and voxels and no image data."""
        batch_inputs = ModelBatchInputs(
            multi_task_gt_batch=self.gt_batch, voxels_data=self.voxels_data, image_data=None
        )

        # Pydantic rebuilds NamedTuple fields on validation, so compare contents, not identity.
        self.assertEqual(batch_inputs.multi_task_gt_batch, self.gt_batch)
        assert batch_inputs.voxels_data is not None
        self.assertTrue(torch.equal(batch_inputs.voxels_data.voxels, self.voxels_data.voxels))
        self.assertTrue(torch.equal(batch_inputs.voxels_data.coords, self.voxels_data.coords))
        self.assertIsNone(batch_inputs.image_data)

    def test_camera_lidar_inputs_carry_both_modalities(self) -> None:
        """Test that voxels and images travel together untouched."""
        batch_inputs = ModelBatchInputs(
            multi_task_gt_batch=self.gt_batch,
            voxels_data=self.voxels_data,
            image_data=self.image_data,
        )

        assert batch_inputs.voxels_data is not None
        assert batch_inputs.image_data is not None
        self.assertTrue(torch.equal(batch_inputs.voxels_data.voxels, self.voxels_data.voxels))
        self.assertTrue(torch.equal(batch_inputs.image_data.images, self.image_data.images))
        self.assertIsNone(batch_inputs.image_data.depth_maps)
        self.assertEqual(tuple(batch_inputs.image_data.images.shape), (2, 3, 3, 8, 8))

    def test_gt_batch_only_inputs_are_allowed(self) -> None:
        """Test that a payload without any preprocessed modality is still valid."""
        batch_inputs = ModelBatchInputs(
            multi_task_gt_batch=self.gt_batch, voxels_data=None, image_data=None
        )

        self.assertIsNone(batch_inputs.voxels_data)
        self.assertIsNone(batch_inputs.image_data)

    def test_rejects_gt_batch_that_is_not_a_model_gt_batch(self) -> None:
        """Test that the GT batch must be a ``ModelGTBatch`` instance, not a look-alike."""
        with self.assertRaises(ValidationError):
            ModelBatchInputs(
                multi_task_gt_batch={"point_cloud_gt_batch": None},  # type: ignore[arg-type]
                voxels_data=None,
                image_data=None,
            )

    def test_rejects_voxels_that_are_not_voxels_data(self) -> None:
        """Test that a raw tensor cannot stand in for ``VoxelsData``."""
        with self.assertRaises(ValidationError):
            ModelBatchInputs(
                multi_task_gt_batch=self.gt_batch,
                voxels_data=torch.zeros(4, 5, 4),  # type: ignore[arg-type]
                image_data=None,
            )

    def test_is_frozen(self) -> None:
        """Test that the payload cannot be mutated after construction."""
        batch_inputs = ModelBatchInputs(
            multi_task_gt_batch=self.gt_batch, voxels_data=None, image_data=None
        )

        with self.assertRaises(ValidationError):
            batch_inputs.voxels_data = self.voxels_data  # type: ignore[misc]


if __name__ == "__main__":
    unittest.main()
