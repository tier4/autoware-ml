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

"""Unit tests for the point cloud transforms running on ``ModelGTSample``."""

from __future__ import annotations

from collections.abc import Sequence
import unittest

import torch

from autoware_ml.dataclasses.batch.sample_batch import ModelGTSample
from autoware_ml.dataclasses.batch.segmentation3d import Segmentation3DGTSample
from autoware_ml.geometry.points.lidar_points import LiDARPoints
from autoware_ml.transforms.point_cloud.geometry import (
    GlobalRotScaleTrans,
    PointsRandomShuffle,
    PointsRangeFilter,
    RandomRotateTargetAngle,
)
from autoware_ml.transforms.point_cloud.perturbation import RandomJitter, RandomStrengthJitter
from autoware_ml.transforms.point_cloud.sampling import RandomDropout
from autoware_ml.types.geometry import PointFeatureName


def build_points(coords: Sequence[Sequence[float]]) -> LiDARPoints:
    """Build a point cloud holding only the coordinates of the given points."""
    return LiDARPoints(
        points=torch.tensor(coords, dtype=torch.float32),
        point_feature_names=[PointFeatureName.X, PointFeatureName.Y, PointFeatureName.Z],
        timestamp=0.0,
    )


def build_sample(
    points: LiDARPoints, labels: Sequence[int] | None = None, ignore_index: int = -1
) -> ModelGTSample:
    """Build a sample holding a point cloud and optionally its semantic labels."""
    segmentation3d_gt_sample = (
        None
        if labels is None
        else Segmentation3DGTSample(
            gt_semantic_mask=torch.tensor(labels, dtype=torch.int64),
            ignore_index=ignore_index,
        )
    )
    return ModelGTSample(
        lidar_point_cloud_samples=None,
        image_samples=None,
        point_cloud_data=points,
        camera_image_data=None,
        detection3d_gt_bboxes_3d=None,
        segmentation3d_gt_sample=segmentation3d_gt_sample,
    )


class TestPointsRangeFilter(unittest.TestCase):
    """Range filtering of the points and of their semantic labels."""

    def setUp(self) -> None:
        """Build four points, the second and the fourth out of range."""
        self.points = build_points(
            [[0.0, 0.0, 0.0], [9.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 9.0, 0.0]]
        )
        self.transform = PointsRangeFilter(points_range=(-2.0, -2.0, -2.0, 2.0, 2.0, 2.0))

    def test_keeps_only_the_points_in_range(self) -> None:
        sample = self.transform(build_sample(self.points))

        self.assertEqual(
            sample.point_cloud_data.coords.tolist(), [[0.0, 0.0, 0.0], [1.0, 1.0, 0.0]]
        )

    def test_drops_the_labels_of_the_removed_points(self) -> None:
        sample = self.transform(build_sample(self.points, labels=[5, 6, 7, 8]))

        self.assertEqual(sample.segmentation3d_gt_sample.gt_semantic_mask.tolist(), [5, 7])

    def test_keeps_the_sample_without_labels(self) -> None:
        sample = self.transform(build_sample(self.points))

        self.assertIsNone(sample.segmentation3d_gt_sample)

    def test_the_range_is_half_open_like_the_grid_over_it(self) -> None:
        # A voxel grid over the range ends at the maximum, so a point on the maximum would fall
        # one cell past it and is dropped, while a point on the minimum stays
        points = build_points(
            [[-2.0, -2.0, -2.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]]
        )

        sample = self.transform(build_sample(points, labels=[1, 2, 3, 4]))

        self.assertEqual(sample.point_cloud_data.coords.tolist(), [[-2.0, -2.0, -2.0]])
        self.assertEqual(sample.segmentation3d_gt_sample.gt_semantic_mask.tolist(), [1])


class TestPointsRandomShuffle(unittest.TestCase):
    """Shuffling of the points and of their semantic labels."""

    def test_labels_follow_the_points(self) -> None:
        points = build_points([[float(index), 0.0, 0.0] for index in range(32)])
        transform = PointsRandomShuffle()

        sample = transform(build_sample(points, labels=list(range(32))))

        shuffled_x = sample.point_cloud_data.coords[:, 0].to(torch.int64)
        self.assertEqual(
            sample.segmentation3d_gt_sample.gt_semantic_mask.tolist(), shuffled_x.tolist()
        )

    def test_keeps_the_sample_without_labels(self) -> None:
        transform = PointsRandomShuffle()

        sample = transform(build_sample(build_points([[1.0, 0.0, 0.0]])))

        self.assertIsNone(sample.segmentation3d_gt_sample)


class TestGlobalRotScaleTrans(unittest.TestCase):
    """The augmentation keeps every field of the sample it does not touch."""

    def test_keeps_the_labels_and_records_the_transformation(self) -> None:
        transform = GlobalRotScaleTrans(
            yaw_rot_range=[0.0, 0.0], scale_ratio_range=[2.0, 2.0], translation_std=None
        )

        sample = transform(build_sample(build_points([[1.0, 0.0, 0.0]]), labels=[3]))

        self.assertEqual(sample.point_cloud_data.coords.tolist(), [[2.0, 0.0, 0.0]])
        self.assertEqual(sample.segmentation3d_gt_sample.gt_semantic_mask.tolist(), [3])
        self.assertIsNotNone(sample.lidar_transformation_sample)


class TestRandomRotateTargetAngle(unittest.TestCase):
    """Rotation by one of a few target yaw angles."""

    def test_rotates_by_the_only_target_angle(self) -> None:
        transform = RandomRotateTargetAngle(probability=1.0, yaw_angle_ratios=[1.0])

        sample = transform(build_sample(build_points([[1.0, 0.0, 0.0]])))

        torch.testing.assert_close(
            sample.point_cloud_data.coords,
            torch.tensor([[-1.0, 0.0, 0.0]]),
            atol=1e-6,
            rtol=0.0,
        )

    def test_skips_the_rotation_at_zero_probability(self) -> None:
        transform = RandomRotateTargetAngle(probability=0.0, yaw_angle_ratios=[1.0])

        sample = transform(build_sample(build_points([[1.0, 0.0, 0.0]])))

        self.assertEqual(sample.point_cloud_data.coords.tolist(), [[1.0, 0.0, 0.0]])


class TestRandomJitter(unittest.TestCase):
    """Clipped Gaussian noise on the coordinates."""

    def test_jitter_stays_within_the_clip(self) -> None:
        points = build_points([[0.0, 0.0, 0.0]] * 256)
        transform = RandomJitter(sigma=1.0, clip=0.05)

        sample = transform(build_sample(points))

        self.assertLessEqual(float(sample.point_cloud_data.coords.abs().max()), 0.05 + 1e-6)

    def test_leaves_the_point_count_untouched(self) -> None:
        points = build_points([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        transform = RandomJitter(sigma=0.01, clip=0.05)

        sample = transform(build_sample(points, labels=[1, 2]))

        self.assertEqual(len(sample.point_cloud_data), 2)
        self.assertEqual(sample.segmentation3d_gt_sample.gt_semantic_mask.tolist(), [1, 2])


class TestRandomStrengthJitter(unittest.TestCase):
    """Gamma, scale, and shift applied to the intensity."""

    def _build_sample_with_intensity(self, intensities: Sequence[float]) -> ModelGTSample:
        """Build a sample whose points carry coordinates and an intensity."""
        points = LiDARPoints(
            points=torch.tensor(
                [[0.0, 0.0, 0.0, intensity] for intensity in intensities], dtype=torch.float32
            ),
            point_feature_names=[
                PointFeatureName.X,
                PointFeatureName.Y,
                PointFeatureName.Z,
                PointFeatureName.INTENSITY,
            ],
            timestamp=0.0,
        )
        return build_sample(points)

    def test_identity_parameters_keep_the_intensity(self) -> None:
        transform = RandomStrengthJitter(
            gamma_range=[1.0, 1.0], scale_range=[1.0, 1.0], shift_range=[0.0, 0.0]
        )

        sample = transform(self._build_sample_with_intensity([0.0, 0.25, 1.0]))

        torch.testing.assert_close(
            sample.point_cloud_data.feature(PointFeatureName.INTENSITY),
            torch.tensor([0.0, 0.25, 1.0]),
        )

    def test_the_result_stays_normalized(self) -> None:
        transform = RandomStrengthJitter(
            gamma_range=[0.5, 0.5], scale_range=[4.0, 4.0], shift_range=[0.5, 0.5]
        )

        sample = transform(self._build_sample_with_intensity([0.0, 0.25, 1.0]))

        intensity = sample.point_cloud_data.feature(PointFeatureName.INTENSITY)
        self.assertEqual(intensity.tolist(), [0.5, 1.0, 1.0])

    def test_rejects_a_descending_range(self) -> None:
        with self.assertRaises(ValueError):
            RandomStrengthJitter(
                gamma_range=[1.0, 0.5], scale_range=[1.0, 1.0], shift_range=[0.0, 0.0]
            )

    def test_rejects_points_without_intensity(self) -> None:
        transform = RandomStrengthJitter(
            gamma_range=[1.0, 1.0], scale_range=[1.0, 1.0], shift_range=[0.0, 0.0]
        )

        with self.assertRaises(ValueError):
            transform(build_sample(build_points([[0.0, 0.0, 0.0]])))


class TestRandomDropout(unittest.TestCase):
    """Random removal of a fraction of the points."""

    def test_keeps_the_expected_number_of_points(self) -> None:
        points = build_points([[float(index), 0.0, 0.0] for index in range(10)])
        transform = RandomDropout(dropout_ratio=0.2, probability=1.0)

        sample = transform(build_sample(points))

        self.assertEqual(len(sample.point_cloud_data), 8)

    def test_labels_follow_the_kept_points(self) -> None:
        points = build_points([[float(index), 0.0, 0.0] for index in range(10)])
        transform = RandomDropout(dropout_ratio=0.5, probability=1.0)

        sample = transform(build_sample(points, labels=list(range(10))))

        kept_x = sample.point_cloud_data.coords[:, 0].to(torch.int64)
        self.assertEqual(sample.segmentation3d_gt_sample.gt_semantic_mask.tolist(), kept_x.tolist())

    def test_skips_the_dropout_at_zero_probability(self) -> None:
        points = build_points([[float(index), 0.0, 0.0] for index in range(10)])
        transform = RandomDropout(dropout_ratio=0.9, probability=0.0)

        sample = transform(build_sample(points))

        self.assertEqual(len(sample.point_cloud_data), 10)
