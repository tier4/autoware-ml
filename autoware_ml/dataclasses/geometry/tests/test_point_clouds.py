"""Unit tests for the PointCloudGTBatch and LiDARPointCloudSample named tuples."""

from __future__ import annotations

import unittest

import torch

from autoware_ml.dataclasses.geometry.point_clouds import (
    LiDARPointCloudSample,
    PointCloudGTBatch,
)
from autoware_ml.geometry.points.lidar_points import LiDARPoints
from autoware_ml.types.geometry import PointFeatureName


class PointCloudGTBatchTestCase(unittest.TestCase):
    """Shared fixtures for the PointCloudGTBatch tests."""

    FEATURE_NAMES = [
        PointFeatureName.X,
        PointFeatureName.Y,
        PointFeatureName.Z,
        PointFeatureName.INTENSITY,
    ]

    def setUp(self) -> None:
        """Build two samples with 3 and 2 points whose features carry distinct values."""
        self.device = torch.device("cpu")
        self.samples = [self.make_points(3, fill=1.0), self.make_points(2, fill=2.0)]

    def make_points(self, num_points: int, fill: float) -> LiDARPoints:
        """Build LiDAR points whose every feature carries ``fill``."""
        return LiDARPoints(
            torch.full((num_points, len(self.FEATURE_NAMES)), fill, dtype=torch.float32),
            self.FEATURE_NAMES,
            timestamp=1.0,
        )


class TestPointCloudGTBatchFields(PointCloudGTBatchTestCase):
    """The named tuple contract of PointCloudGTBatch."""

    def test_field_order(self) -> None:
        self.assertEqual(PointCloudGTBatch._fields, ("points", "batch_indices"))

    def test_is_immutable(self) -> None:
        batch = PointCloudGTBatch.collate_gt_samples(self.samples)

        with self.assertRaises(AttributeError):
            batch.points = torch.zeros(1)  # type: ignore


class TestPointCloudGTBatchCollate(PointCloudGTBatchTestCase):
    """PointCloudGTBatch.collate_gt_samples."""

    def test_empty_sequence_returns_none(self) -> None:
        self.assertIsNone(PointCloudGTBatch.collate_gt_samples([]))

    def test_concatenates_points_and_assigns_batch_indices(self) -> None:
        batch = PointCloudGTBatch.collate_gt_samples(self.samples)

        self.assertIsNotNone(batch)
        assert batch is not None
        self.assertEqual(batch.points.shape, (5, len(self.FEATURE_NAMES)))
        self.assertEqual(batch.batch_indices.shape, (5,))
        self.assertEqual(batch.batch_indices.tolist(), [0, 0, 0, 1, 1])
        self.assertTrue(torch.all(batch.points[:3] == 1.0))
        self.assertTrue(torch.all(batch.points[3:] == 2.0))

    def test_output_dtypes(self) -> None:
        batch = PointCloudGTBatch.collate_gt_samples(self.samples)

        assert batch is not None
        self.assertEqual(batch.points.dtype, torch.float32)
        self.assertEqual(batch.batch_indices.dtype, torch.int32)

    def test_single_sample_gets_zero_indices(self) -> None:
        sample = self.make_points(4, fill=3.0)

        batch = PointCloudGTBatch.collate_gt_samples([sample])

        assert batch is not None
        self.assertTrue(torch.equal(batch.points, sample.points))
        self.assertEqual(batch.batch_indices.tolist(), [0, 0, 0, 0])

    def test_sample_without_points_keeps_later_indices(self) -> None:
        samples = [
            self.make_points(2, fill=1.0),
            self.make_points(0, fill=0.0),
            self.make_points(1, fill=3.0),
        ]

        batch = PointCloudGTBatch.collate_gt_samples(samples)

        assert batch is not None
        # The empty sample contributes no rows, but the following sample still gets index 2.
        self.assertEqual(batch.points.shape, (3, len(self.FEATURE_NAMES)))
        self.assertEqual(batch.batch_indices.tolist(), [0, 0, 2])

    def test_all_samples_without_points_gives_empty_batch(self) -> None:
        batch = PointCloudGTBatch.collate_gt_samples([self.make_points(0, fill=0.0)])

        assert batch is not None
        self.assertEqual(batch.points.shape, (0, len(self.FEATURE_NAMES)))
        self.assertEqual(batch.batch_indices.shape, (0,))


class TestPointCloudGTBatchToDevice(PointCloudGTBatchTestCase):
    """PointCloudGTBatch.to_device."""

    def setUp(self) -> None:
        """Collate the fixture samples once."""
        super().setUp()
        self.batch = PointCloudGTBatch.collate_gt_samples(self.samples)

    def test_returns_new_batch_with_equal_tensors(self) -> None:
        assert self.batch is not None
        moved = self.batch.to_device(self.device)

        self.assertIsInstance(moved, PointCloudGTBatch)
        self.assertIsNot(moved, self.batch)
        self.assertTrue(torch.equal(moved.points, self.batch.points))
        self.assertTrue(torch.equal(moved.batch_indices, self.batch.batch_indices))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required to move tensors to GPU")
    def test_moves_every_tensor_to_cuda(self) -> None:
        assert self.batch is not None
        moved = self.batch.to_device(torch.device("cuda"))

        self.assertEqual(moved.points.device.type, "cuda")
        self.assertEqual(moved.batch_indices.device.type, "cuda")
        self.assertEqual(self.batch.points.device.type, "cpu")
        self.assertEqual(self.batch.batch_indices.device.type, "cpu")


class LiDARPointCloudSampleTestCase(unittest.TestCase):
    """Shared fixtures for the LiDARPointCloudSample tests."""

    def setUp(self) -> None:
        """Build one LiDAR sweep record with distinguishable transformation matrices."""
        self.sample = LiDARPointCloudSample(
            point_cloud_path="/data/lidar/000001.pcd.bin",
            timestamp=1700000000.5,
            sensor_to_ego_pose_matrix=torch.eye(4, dtype=torch.float32),
            lidar_to_ego_pose_to_global_matrix=torch.eye(4, dtype=torch.float32) * 2.0,
            lidar_sensor_to_lidar_sweep_matrix=torch.eye(4, dtype=torch.float32) * 3.0,
        )


class TestLiDARPointCloudSample(LiDARPointCloudSampleTestCase):
    """The named tuple contract of LiDARPointCloudSample."""

    def test_field_order(self) -> None:
        self.assertEqual(
            LiDARPointCloudSample._fields,
            (
                "point_cloud_path",
                "timestamp",
                "sensor_to_ego_pose_matrix",
                "lidar_to_ego_pose_to_global_matrix",
                "lidar_sensor_to_lidar_sweep_matrix",
            ),
        )

    def test_holds_given_fields(self) -> None:
        self.assertEqual(self.sample.point_cloud_path, "/data/lidar/000001.pcd.bin")
        self.assertEqual(self.sample.timestamp, 1700000000.5)
        self.assertTrue(torch.equal(self.sample.sensor_to_ego_pose_matrix, torch.eye(4)))
        self.assertTrue(
            torch.equal(self.sample.lidar_to_ego_pose_to_global_matrix, torch.eye(4) * 2.0)
        )
        self.assertTrue(
            torch.equal(self.sample.lidar_sensor_to_lidar_sweep_matrix, torch.eye(4) * 3.0)
        )

    def test_is_immutable(self) -> None:
        with self.assertRaises(AttributeError):
            self.sample.timestamp = 0.0  # type: ignore

    def test_replace_returns_new_sample(self) -> None:
        replaced = self.sample._replace(timestamp=5.0)

        self.assertEqual(self.sample.timestamp, 1700000000.5)
        self.assertEqual(replaced.timestamp, 5.0)
        self.assertIs(replaced.sensor_to_ego_pose_matrix, self.sample.sensor_to_ego_pose_matrix)


if __name__ == "__main__":
    unittest.main()
