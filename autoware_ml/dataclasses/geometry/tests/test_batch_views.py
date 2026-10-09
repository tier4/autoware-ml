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

"""Unit tests for the per sample views of the collated batch dataclasses."""

from __future__ import annotations

import unittest

import torch

from autoware_ml.dataclasses.batch.detection3d import Detection3DGTBatch
from autoware_ml.dataclasses.geometry.images import ImageGTBatch
from autoware_ml.dataclasses.geometry.point_clouds import PointCloudGTBatch
from autoware_ml.dataclasses.geometry.voxels import VoxelsData


class TestPointCloudGTBatchViews(unittest.TestCase):
    """The point batch splits back into its samples, empty trailing samples included."""

    def setUp(self) -> None:
        """Collate two points of the first sample, none of the second, one of the third."""
        self.batch = PointCloudGTBatch(
            points=torch.arange(12, dtype=torch.float32).reshape(3, 4),
            batch_indices=torch.tensor([0, 0, 2], dtype=torch.int32),
            batch_size=4,
            timestamp_difference_dim=-1,
        )

    def test_offsets_end_every_sample(self) -> None:
        self.assertEqual(self.batch.offsets().tolist(), [2, 2, 3, 3])

    def test_split_points_keeps_empty_samples(self) -> None:
        samples = self.batch.split_points()

        self.assertEqual([sample.shape[0] for sample in samples], [2, 0, 1, 0])
        self.assertTrue(torch.equal(samples[2], self.batch.points[2:]))


class TestDetection3DGTBatchViews(unittest.TestCase):
    """The padded boxes give back only the valid boxes of every sample."""

    def test_valid_views_drop_the_padding(self) -> None:
        batch = Detection3DGTBatch(
            gt_bboxes_3d=torch.arange(40, dtype=torch.float32).reshape(2, 2, 10),
            gt_labels_3d=torch.tensor([[3, -1], [1, 2]], dtype=torch.int32),
            gt_valid_bboxes=torch.tensor([1, 2], dtype=torch.int32),
            gt_bboxes_num_points=torch.tensor([[5, 0], [6, 7]], dtype=torch.int32),
        )

        boxes = batch.valid_bboxes_3d()
        labels = batch.valid_labels_3d()

        self.assertEqual([box.shape[0] for box in boxes], [1, 2])
        self.assertEqual([label.tolist() for label in labels], [[3], [1, 2]])
        self.assertEqual(labels[0].dtype, torch.int64)
        self.assertEqual([n.tolist() for n in batch.valid_bboxes_num_points()], [[5], [6, 7]])


class TestImageGTBatchFusedImages(unittest.TestCase):
    """The fused image stacks the depth maps onto the images, one per camera."""

    def _batch(self, depth_maps: torch.Tensor | None) -> ImageGTBatch:
        return ImageGTBatch(
            images=torch.zeros(2, 3, 3, 4, 4),
            depth_maps=depth_maps,
            camera_intrinsics=torch.eye(3).expand(2, 3, 3, 3),
            image_augmentation_matrices=torch.eye(4).expand(2, 3, 4, 4),
            lidar2images=torch.eye(4).expand(2, 3, 4, 4),
            lidar2cams=torch.eye(4).expand(2, 3, 4, 4),
            calibration_statuses=None,
        )

    def test_fuses_the_depth_channels_per_camera(self) -> None:
        fused = self._batch(torch.ones(2, 3, 2, 4, 4)).fused_images()

        self.assertEqual(tuple(fused.shape), (6, 5, 4, 4))
        self.assertTrue(bool((fused[:, 3:] == 1).all()))

    def test_requires_depth_maps(self) -> None:
        with self.assertRaises(ValueError):
            self._batch(None).fused_images()


class TestVoxelsDataCoords(unittest.TestCase):
    """The voxel coordinates come out with the sample first and the axes in either order."""

    def test_batch_coords(self) -> None:
        voxels = VoxelsData(
            voxels=torch.zeros(2, 1, 4),
            coords=torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.int32),
            num_points=torch.ones(2, dtype=torch.int32),
            batch_indices=torch.tensor([0, 1], dtype=torch.int32),
            point_voxel_indices=torch.tensor([0, 1]),
            num_dropped_voxels=torch.zeros((), dtype=torch.int64),
        )

        zyx = voxels.batch_zyx_coords()
        xyz = voxels.batch_xyz_coords()

        self.assertEqual(zyx.tolist(), [[0, 3, 2, 1], [1, 6, 5, 4]])
        self.assertEqual(xyz.tolist(), [[0, 1, 2, 3], [1, 4, 5, 6]])
        self.assertEqual(zyx.dtype, torch.int32)


if __name__ == "__main__":
    unittest.main()
