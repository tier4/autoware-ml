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

"""Unit tests for the point-cloud perturbation transforms."""

import unittest

import torch

from autoware_ml.dataclasses.batch.sample_batch import ModelGTSample
from autoware_ml.geometry.points.lidar_points import LiDARPoints
from autoware_ml.transforms.point_cloud.geometry import PointsRandomShuffle
from autoware_ml.types.geometry import PointFeatureName


class TestPointsRandomShuffle(unittest.TestCase):
    """Unit tests for the PointsRandomShuffle transform."""

    def build_point_cloud_data(self, num_points: int = 20) -> LiDARPoints:
        """Build a point cloud whose rows are all distinct so a permutation is detectable.

        Args:
            num_points: Number of points to generate.

        Returns:
            LiDARPoints holding ``num_points`` distinct points.
        """
        # Each row is [i, 2i, 3i, i / 10], hence every row is unique.
        index = torch.arange(num_points, dtype=torch.float32).unsqueeze(1)
        points = torch.cat([index, 2.0 * index, 3.0 * index, index / 10.0], dim=1)
        return LiDARPoints(
            points=points,
            point_feature_names=[
                PointFeatureName.X,
                PointFeatureName.Y,
                PointFeatureName.Z,
                PointFeatureName.INTENSITY,
            ],
            timestamp=0.0,
        )

    def build_multi_task_gt_sample(self, point_cloud_data: LiDARPoints | None) -> ModelGTSample:
        """Build a minimal sample holding only the given point cloud.

        Args:
            point_cloud_data: The point cloud to put in the sample, or ``None``.

        Returns:
            ModelGTSample with the point cloud and every other field unset.
        """
        return ModelGTSample(
            lidar_point_cloud_samples=None,
            image_samples=None,
            point_cloud_data=point_cloud_data,
            camera_image_data=None,
            detection3d_gt_bboxes_3d=None,
            segmentation3d_gt_sample=None,
        )

    def assert_same_point_set(self, points: torch.Tensor, expected_points: torch.Tensor) -> None:
        """Assert that two point tensors hold the same rows, in any order.

        Args:
            points: The points under test.
            expected_points: The points they must be a permutation of.
        """
        self.assertEqual(points.shape, expected_points.shape)
        # Rows are unique (see build_point_cloud_data), so sorting by the X feature is a
        # canonical order for both tensors.
        sorted_points = points[torch.argsort(points[:, 0])]
        sorted_expected_points = expected_points[torch.argsort(expected_points[:, 0])]
        self.assertTrue(torch.equal(sorted_points, sorted_expected_points))

    def test_shuffles_point_order(self) -> None:
        """Test that the points are permuted and no point is lost, added, or altered."""
        torch.manual_seed(0)
        point_cloud_data = self.build_point_cloud_data()
        original_points = point_cloud_data.points.clone()
        sample = self.build_multi_task_gt_sample(point_cloud_data)

        output = PointsRandomShuffle()(sample)

        assert output.point_cloud_data is not None
        shuffled_points = output.point_cloud_data.points
        # With 20 distinct rows the odds of drawing the identity permutation are negligible.
        self.assertFalse(torch.equal(shuffled_points, original_points))
        self.assert_same_point_set(shuffled_points, original_points)

    def test_rows_stay_intact(self) -> None:
        """Test that features move together with their point, i.e. whole rows are permuted."""
        torch.manual_seed(1)
        point_cloud_data = self.build_point_cloud_data()
        sample = self.build_multi_task_gt_sample(point_cloud_data)

        output = PointsRandomShuffle()(sample)

        assert output.point_cloud_data is not None
        points = output.point_cloud_data.points
        # Every row keeps the relation [i, 2i, 3i, i / 10] of the builder.
        self.assertTrue(torch.allclose(points[:, 1], 2.0 * points[:, 0]))
        self.assertTrue(torch.allclose(points[:, 2], 3.0 * points[:, 0]))
        self.assertTrue(torch.allclose(points[:, 3], points[:, 0] / 10.0))

    def test_matches_torch_permutation(self) -> None:
        """Test that the shuffle follows torch's random state, so it is reproducible."""
        point_cloud_data = self.build_point_cloud_data()
        original_points = point_cloud_data.points.clone()
        sample = self.build_multi_task_gt_sample(point_cloud_data)

        torch.manual_seed(42)
        expected_index = torch.randperm(len(point_cloud_data))
        torch.manual_seed(42)
        output = PointsRandomShuffle()(sample)

        assert output.point_cloud_data is not None
        self.assertTrue(
            torch.equal(output.point_cloud_data.points, original_points[expected_index])
        )

    def test_shuffles_in_place_and_returns_same_sample(self) -> None:
        """Test that the point cloud is shuffled in place and the sample is returned as is."""
        point_cloud_data = self.build_point_cloud_data()
        sample = self.build_multi_task_gt_sample(point_cloud_data)

        output = PointsRandomShuffle()(sample)

        self.assertIs(output, sample)
        self.assertIs(output.point_cloud_data, point_cloud_data)

    def test_empty_point_cloud_is_returned_unchanged(self) -> None:
        """Test that an empty point cloud is passed through without error."""
        point_cloud_data = self.build_point_cloud_data(num_points=0)
        sample = self.build_multi_task_gt_sample(point_cloud_data)

        output = PointsRandomShuffle()(sample)

        assert output.point_cloud_data is not None
        self.assertIs(output, sample)
        self.assertEqual(len(output.point_cloud_data), 0)
        self.assertEqual(output.point_cloud_data.points.shape, (0, 4))

    def test_missing_point_cloud_data_key(self) -> None:
        """Test that missing 'point_cloud_data' raises KeyError."""
        sample = self.build_multi_task_gt_sample(point_cloud_data=None)

        with self.assertRaises(KeyError):
            PointsRandomShuffle()(sample)


if __name__ == "__main__":
    unittest.main()
