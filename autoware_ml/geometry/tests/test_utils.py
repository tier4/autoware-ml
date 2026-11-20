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

"""Unit tests for the geometry helpers."""

import math
import unittest

import torch

from autoware_ml.geometry.utils import transform_points


def _pose(yaw: float, translation: tuple[float, float, float]) -> torch.Tensor:
    pose = torch.eye(4, dtype=torch.float64)
    pose[:2, :2] = torch.tensor(
        [[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]], dtype=torch.float64
    )
    pose[:3, 3] = torch.tensor(translation, dtype=torch.float64)
    return pose


class TestGeometryUtils(unittest.TestCase):
    def test_transform_points_matches_the_homogeneous_product(self) -> None:
        """
        Input: points of two samples and one transform per sample, broadcast over the points.
        Expected: the homogeneous matrix product of every point.
        """
        points = torch.tensor(
            [[[1.0, 2.0, 3.0], [-4.0, 0.5, 1.0]], [[100.0, -50.0, 2.0], [0.0, 0.0, 0.0]]],
            dtype=torch.float64,
        )
        poses = torch.stack([_pose(0.3, (1.0, -2.0, 0.5)), _pose(-1.2, (65000.0, 12.0, 3.0))])
        homogeneous = torch.cat([points, torch.ones_like(points[..., :1])], dim=-1)
        expected = torch.einsum("bij,bkj->bki", poses, homogeneous)[..., :3]

        torch.testing.assert_close(transform_points(points, poses[:, None]), expected)
