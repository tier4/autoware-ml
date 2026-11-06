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

"""Tests for the lidar 3D boxes."""

from __future__ import annotations

import math
import unittest

import numpy as np
import torch

from autoware_ml.geometry.bbox_3d.lidar_bbox3d import LidarBBoxes3D
from autoware_ml.geometry.utils import points_in_rotated_box
from autoware_ml.types.geometry import Box3DCenterCoordinateType, Box3DFieldIndex


def boxes(params: list[list[float]]) -> LidarBBoxes3D:
    """
    Lidar boxes from (x, y, z, length, width, height, yaw) rows, every other field zero.

    Args:
      params: One row per box.

    Returns:
      LidarBBoxes3D: The boxes, labelled as cars.
    """
    count = len(params)
    bbox_params = torch.zeros((count, len(Box3DFieldIndex)), dtype=torch.float32)
    bbox_params[:, : Box3DFieldIndex.YAW + 1] = torch.tensor(params, dtype=torch.float32).reshape(
        -1, Box3DFieldIndex.YAW + 1
    )
    return LidarBBoxes3D(
        bbox_params=bbox_params,
        bbox_labels=torch.zeros(count, dtype=torch.int32),
        bbox_label_names=["car"] * count,
        bbox_num_lidar_points=torch.zeros(count, dtype=torch.int32),
        bbox_center_coordinate_type=Box3DCenterCoordinateType.GRAVITY_CENTER,
    )


class TestComputePointsInBBoxes(unittest.TestCase):
    """The point membership test of the lidar boxes."""

    def test_follows_the_box_frame(self) -> None:
        # A box of length 4 along its heading, rotated by 90 degrees, so its length lies along y
        box = boxes([[0.0, 0.0, 0.0, 4.0, 1.0, 2.0, math.pi / 2]])
        points = torch.tensor(
            [
                [0.0, 1.9, 0.0],  # along the rotated length, inside
                [1.9, 0.0, 0.0],  # along the rotated width, outside
                [0.0, 0.0, 0.9],  # below the roof, inside
                [0.0, 0.0, 1.1],  # above the roof, outside
                [0.4, 1.9, -0.9],  # near a corner, inside
                [0.6, 0.0, 0.0],  # past the half width, outside
            ]
        )

        mask = box.compute_points_in_bboxes(points)

        self.assertEqual(mask.dtype, torch.bool)
        self.assertEqual(mask.tolist(), [[True, False, True, False, True, False]])

    def test_counts_the_box_surface_as_inside(self) -> None:
        box = boxes([[1.0, -2.0, 0.5, 2.0, 1.0, 1.0, 0.0]])
        points = torch.tensor(
            [[2.0, -2.0, 0.5], [1.0, -1.5, 0.5], [1.0, -2.0, 1.0], [2.0, -1.5, 1.0]]
        )

        self.assertTrue(box.compute_points_in_bboxes(points).all())

    def test_matches_the_per_box_reference_on_a_random_cloud(self) -> None:
        generator = torch.Generator().manual_seed(0)
        num_boxes, num_points = 40, 20_000
        params = torch.zeros((num_boxes, 7))
        params[:, :2] = torch.randn((num_boxes, 2), generator=generator) * 20
        params[:, 2] = torch.randn(num_boxes, generator=generator)
        params[:, 3:6] = torch.rand((num_boxes, 3), generator=generator) * 6 + 0.5
        params[:, 6] = (torch.rand(num_boxes, generator=generator) - 0.5) * 2 * math.pi
        points = torch.randn((num_points, 3), generator=generator) * torch.tensor([20.0, 20.0, 1.5])

        mask = boxes(params.tolist()).compute_points_in_bboxes(points)

        expected = np.stack([points_in_rotated_box(points.numpy(), row) for row in params.numpy()])
        self.assertEqual(mask.shape, (num_boxes, num_points))
        self.assertGreater(int(mask.sum()), 0)
        np.testing.assert_array_equal(mask.numpy(), expected)

    def test_is_empty_without_boxes_or_points(self) -> None:
        points = torch.zeros((3, 3))

        without_boxes = boxes([]).compute_points_in_bboxes(points)
        without_points = boxes([[0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0]]).compute_points_in_bboxes(
            points[:0]
        )

        self.assertEqual((without_boxes.shape, without_boxes.dtype), ((0, 3), torch.bool))
        self.assertEqual((without_points.shape, without_points.dtype), ((1, 0), torch.bool))


if __name__ == "__main__":
    unittest.main()
