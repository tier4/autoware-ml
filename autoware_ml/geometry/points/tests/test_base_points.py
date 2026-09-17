"""Unit tests for the BasePoints container."""

from __future__ import annotations

import unittest

import torch

from autoware_ml.geometry.points.lidar_points import LiDARPoints
from autoware_ml.types.geometry import PointFeatureName

XYZI = [PointFeatureName.X, PointFeatureName.Y, PointFeatureName.Z, PointFeatureName.INTENSITY]
XYZIT = XYZI + [PointFeatureName.TIMESTAMP_DIFFERENCE]


class TestTimestampDifferenceDim(unittest.TestCase):
    """The timestamp difference dimension through construction, mutation and concatenation."""

    def test_defaults_to_absent(self) -> None:
        points = LiDARPoints(torch.zeros((2, 4)), XYZI, timestamp=1.0)

        self.assertEqual(points.timestamp_difference_dim, -1)

    def test_accepts_matching_dimension(self) -> None:
        points = LiDARPoints(torch.zeros((2, 5)), XYZIT, timestamp=1.0, timestamp_difference_dim=4)

        self.assertEqual(points.timestamp_difference_dim, 4)

    def test_rejects_out_of_range_or_mismatched_dimension(self) -> None:
        with self.assertRaisesRegex(ValueError, "within"):
            LiDARPoints(torch.zeros((2, 5)), XYZIT, timestamp=1.0, timestamp_difference_dim=5)
        with self.assertRaisesRegex(ValueError, "points at feature"):
            LiDARPoints(torch.zeros((2, 5)), XYZIT, timestamp=1.0, timestamp_difference_dim=3)

    def test_add_timestamp_difference_sets_dimension(self) -> None:
        points = LiDARPoints(torch.zeros((2, 4)), XYZI, timestamp=1.0)

        points.add_timestamp_difference(0.5)

        self.assertEqual(points.timestamp_difference_dim, 4)
        torch.testing.assert_close(points.points[:, 4], torch.full((2,), 0.5))

    def test_concat_keeps_dimension(self) -> None:
        first = LiDARPoints(torch.zeros((2, 4)), XYZI, timestamp=1.0)
        second = LiDARPoints(torch.ones((3, 4)), XYZI, timestamp=0.9)
        first.add_timestamp_difference(0.0)
        second.add_timestamp_difference(0.1)

        merged = LiDARPoints.concat([first, second])

        self.assertEqual(merged.shape, (5, 5))
        self.assertEqual(merged.timestamp_difference_dim, 4)
        self.assertEqual(merged.timestamp, 1.0)
        torch.testing.assert_close(merged.points[:, 4], torch.tensor([0.0, 0.0, 0.1, 0.1, 0.1]))

    def test_concat_without_dimension_stays_absent(self) -> None:
        merged = LiDARPoints.concat(
            [
                LiDARPoints(torch.zeros((2, 4)), XYZI, 1.0),
                LiDARPoints(torch.ones((1, 4)), XYZI, 1.0),
            ]
        )

        self.assertEqual(merged.timestamp_difference_dim, -1)

    def test_concat_rejects_mismatched_dimension(self) -> None:
        first = LiDARPoints(torch.zeros((2, 5)), XYZIT, timestamp=1.0, timestamp_difference_dim=4)
        second = LiDARPoints(torch.zeros((2, 5)), XYZIT, timestamp=1.0)

        with self.assertRaisesRegex(ValueError, "timestamp_difference_dim"):
            LiDARPoints.concat([first, second])

    def test_copies_keep_dimension(self) -> None:
        points = LiDARPoints(torch.zeros((2, 5)), XYZIT, timestamp=1.0, timestamp_difference_dim=4)

        self.assertEqual(points.deep_copy().timestamp_difference_dim, 4)
        self.assertEqual(points.shallow_copy().timestamp_difference_dim, 4)
        self.assertEqual(points.detach().timestamp_difference_dim, 4)


if __name__ == "__main__":
    unittest.main()
