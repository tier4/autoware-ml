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

"""Unit tests for naming the tensors of a typed batch as model inputs."""

from __future__ import annotations

import unittest

import torch

from autoware_ml.dataclasses.batch.detection3d import Detection3DGTBatch
from autoware_ml.dataclasses.batch.sample_batch import ModelGTBatch
from autoware_ml.dataclasses.batch.segmentation3d import Segmentation3DGTBatch
from autoware_ml.dataclasses.geometry.images import ImageGTBatch
from autoware_ml.dataclasses.geometry.point_clouds import PointCloudGTBatch
from autoware_ml.preprocessing.batch_adapter import ModelGTBatchAdapter
from autoware_ml.types.geometry import Box3DFieldIndex


def build_batch(with_segmentation: bool = False, with_detection: bool = False) -> ModelGTBatch:
    """Build a two sample batch holding three and two points."""
    points = torch.arange(5 * 4, dtype=torch.float32).reshape(5, 4)
    point_cloud_gt_batch = PointCloudGTBatch(
        points=points,
        batch_indices=torch.tensor([0, 0, 0, 1, 1], dtype=torch.int32),
        batch_size=2,
    )
    segmentation3d_gt_batch = (
        Segmentation3DGTBatch(
            gt_semantic_masks=torch.tensor([1, 2, 3, 4, 5], dtype=torch.int64),
            batch_indices=torch.tensor([0, 0, 0, 1, 1], dtype=torch.int32),
        )
        if with_segmentation
        else None
    )
    detection3d_gt_batch = (
        Detection3DGTBatch(
            gt_bboxes_3d=torch.zeros((2, 3, len(Box3DFieldIndex)), dtype=torch.float32),
            gt_labels_3d=torch.tensor([[0, 1, -1], [2, -1, -1]], dtype=torch.int32),
            gt_valid_bboxes=torch.tensor([2, 1], dtype=torch.int32),
            gt_bboxes_num_points=torch.tensor([[7, 8, 0], [9, 0, 0]], dtype=torch.int32),
        )
        if with_detection
        else None
    )
    return ModelGTBatch(
        point_cloud_gt_batch=point_cloud_gt_batch,
        detection3d_gt_batch=detection3d_gt_batch,
        segmentation3d_gt_batch=segmentation3d_gt_batch,
        image_gt_batch=None,
    )


class TestModelGTBatchAdapter(unittest.TestCase):
    """Model inputs derived from the typed batch."""

    def setUp(self) -> None:
        """Build the adapter shared by every test."""
        self.adapter = ModelGTBatchAdapter()

    def test_splits_the_points_per_sample(self) -> None:
        batch_inputs_dict = self.adapter(build_batch())

        self.assertEqual([len(points) for points in batch_inputs_dict["points"]], [3, 2])
        self.assertEqual(batch_inputs_dict["offset"].tolist(), [3, 5])
        self.assertEqual(batch_inputs_dict["sample_count"], 2)

    def test_coord_is_the_leading_three_features(self) -> None:
        batch_inputs_dict = self.adapter(build_batch())

        self.assertEqual(batch_inputs_dict["coord"].shape, (5, 3))
        self.assertEqual(batch_inputs_dict["feat"].shape, (5, 4))

    def test_carries_the_semantic_labels_when_present(self) -> None:
        batch_inputs_dict = self.adapter(build_batch(with_segmentation=True))

        self.assertEqual(batch_inputs_dict["segment"].tolist(), [1, 2, 3, 4, 5])

    def test_leaves_out_the_tasks_the_batch_does_not_carry(self) -> None:
        batch_inputs_dict = self.adapter(build_batch())

        self.assertNotIn("segment", batch_inputs_dict)
        self.assertNotIn("gt_boxes", batch_inputs_dict)

    def test_unpads_the_boxes_of_every_sample(self) -> None:
        batch_inputs_dict = self.adapter(build_batch(with_detection=True))

        self.assertEqual([len(boxes) for boxes in batch_inputs_dict["gt_boxes"]], [2, 1])
        self.assertEqual(batch_inputs_dict["gt_labels"][0].tolist(), [0, 1])
        self.assertEqual(batch_inputs_dict["gt_labels"][1].tolist(), [2])
        self.assertEqual(batch_inputs_dict["gt_num_points"][0].tolist(), [7, 8])


class TestModelGTBatchAdapterImages(unittest.TestCase):
    """Camera model inputs derived from the typed batch."""

    def setUp(self) -> None:
        """Build the adapter shared by every test."""
        self.adapter = ModelGTBatchAdapter()

    def _build_image_batch(
        self, with_depth: bool = False, with_status: bool = False
    ) -> ModelGTBatch:
        """Build a two sample batch of three cameras each."""
        batch_size, num_cameras = 2, 3
        cameras = (batch_size, num_cameras)
        image_gt_batch = ImageGTBatch(
            images=torch.zeros((*cameras, 3, 4, 4), dtype=torch.float32),
            depth_maps=(
                torch.ones((*cameras, 2, 4, 4), dtype=torch.float32) if with_depth else None
            ),
            camera_intrinsics=torch.eye(3, dtype=torch.float32).expand(*cameras, 3, 3).clone(),
            image_augmentation_matrices=torch.eye(4, dtype=torch.float32)
            .expand(*cameras, 4, 4)
            .clone(),
            lidar2images=torch.eye(4, dtype=torch.float32).expand(*cameras, 4, 4).clone(),
            lidar2cams=torch.eye(4, dtype=torch.float32).expand(*cameras, 4, 4).clone(),
            calibration_statuses=(torch.zeros(cameras, dtype=torch.int64) if with_status else None),
        )
        return ModelGTBatch(
            point_cloud_gt_batch=None,
            detection3d_gt_batch=None,
            segmentation3d_gt_batch=None,
            image_gt_batch=image_gt_batch,
        )

    def test_groups_the_cameras_per_sample(self) -> None:
        batch_inputs_dict = self.adapter(self._build_image_batch())

        self.assertEqual([images.shape[0] for images in batch_inputs_dict["img"]], [3, 3])
        self.assertEqual(len(batch_inputs_dict["lidar2img"]), 2)

    def test_fuses_the_images_with_the_projected_channels(self) -> None:
        batch_inputs_dict = self.adapter(self._build_image_batch(with_depth=True))

        self.assertEqual(batch_inputs_dict["fused_img"].shape, (6, 5, 4, 4))

    def test_leaves_out_the_fusion_without_the_projection(self) -> None:
        batch_inputs_dict = self.adapter(self._build_image_batch())

        self.assertNotIn("fused_img", batch_inputs_dict)
        self.assertNotIn("gt_calibration_status", batch_inputs_dict)

    def test_carries_the_calibration_status_when_present(self) -> None:
        batch_inputs_dict = self.adapter(self._build_image_batch(with_status=True))

        self.assertEqual(batch_inputs_dict["gt_calibration_status"].shape, (6,))
