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
        """Build two samples with 3 and 2 points whose features carry 1.0 and 2.0."""
        self.device = torch.device("cpu")
        self.samples = [self.make_points(3, fill=1.0), self.make_points(2, fill=2.0)]

    def make_points(self, num_points: int, fill: float) -> LiDARPoints:
        """Build LiDAR points with x, y, z and intensity features that all carry ``fill``."""
        return LiDARPoints(
            torch.full((num_points, len(self.FEATURE_NAMES)), fill, dtype=torch.float32),
            self.FEATURE_NAMES,
            timestamp=1.0,
        )

    def collate(self, samples: list[LiDARPoints]) -> PointCloudGTBatch:
        """
        Collate the given samples into a batch.

        The samples are never empty here, so the None branch of ``collate_gt_samples`` is
        asserted away to narrow the return type for the callers.
        """
        batch = PointCloudGTBatch.collate_gt_samples(samples)
        assert batch is not None
        return batch


class TestPointCloudGTBatchFields(PointCloudGTBatchTestCase):
    """The named tuple contract of PointCloudGTBatch."""

    def test_field_order(self) -> None:
        """
        Input: the class itself.
        Expected: the tuple exposes ``points`` then ``batch_indices``, since downstream code
        unpacks the batch positionally.
        Check: compare ``_fields`` against the expected names.
        """
        self.assertEqual(PointCloudGTBatch._fields, ("points", "batch_indices"))

    def test_is_immutable(self) -> None:
        """
        Input: a batch collated from the fixture samples.
        Expected: named tuple fields cannot be reassigned after construction.
        Check: assigning to ``points`` raises AttributeError.
        """
        batch = self.collate(self.samples)

        with self.assertRaises(AttributeError):
            batch.points = torch.zeros(1)


class TestPointCloudGTBatchCollate(PointCloudGTBatchTestCase):
    """PointCloudGTBatch.collate_gt_samples."""

    def test_empty_sequence_returns_none(self) -> None:
        """
        Input: an empty sample sequence.
        Expected: there is nothing to collate, so the result is None instead of an empty batch.
        Check: the return value is None.
        """
        self.assertIsNone(PointCloudGTBatch.collate_gt_samples([]))

    def test_concatenates_points_and_assigns_batch_indices(self) -> None:
        """
        Input: the fixture samples with 3 points of value 1.0 and 2 points of value 2.0.
        Expected: the points are concatenated into 5 rows in sample order and every row is
        tagged with the index of the sample it came from, giving ``[0, 0, 0, 1, 1]``.
        Check: compare the shapes, the index list, and assert the first 3 rows are all 1.0 and
        the last 2 rows are all 2.0.
        """
        batch = self.collate(self.samples)

        self.assertEqual(batch.points.shape, (5, len(self.FEATURE_NAMES)))
        self.assertEqual(batch.batch_indices.shape, (5,))
        self.assertEqual(batch.batch_indices.tolist(), [0, 0, 0, 1, 1])
        self.assertTrue(torch.all(batch.points[:3] == 1.0))
        self.assertTrue(torch.all(batch.points[3:] == 2.0))

    def test_output_dtypes(self) -> None:
        """
        Input: the fixture samples.
        Expected: the points keep float32 and the batch indices are int32, as the voxelization
        ops consume them.
        Check: read the dtype of both fields.
        """
        batch = self.collate(self.samples)

        self.assertEqual(batch.points.dtype, torch.float32)
        self.assertEqual(batch.batch_indices.dtype, torch.int32)

    def test_single_sample_gets_zero_indices(self) -> None:
        """
        Input: a single sample with 4 points.
        Expected: the points pass through unchanged and every row gets batch index 0.
        Check: compare the points with ``torch.equal`` and the indices against four zeros.
        """
        sample = self.make_points(4, fill=3.0)

        batch = self.collate([sample])

        self.assertTrue(torch.equal(batch.points, sample.points))
        self.assertEqual(batch.batch_indices.tolist(), [0, 0, 0, 0])

    def test_sample_without_points_keeps_later_indices(self) -> None:
        """
        Input: samples with 2, 0 and 1 points.
        Expected: the empty sample contributes no rows but still consumes batch index 1, so the
        last sample's row is tagged 2 and the model can map rows back to samples.
        Check: the batch has 3 rows and the indices are ``[0, 0, 2]``.
        """
        samples = [
            self.make_points(2, fill=1.0),
            self.make_points(0, fill=0.0),
            self.make_points(1, fill=3.0),
        ]

        batch = self.collate(samples)

        # The empty sample contributes no rows, but the following sample still gets index 2.
        self.assertEqual(batch.points.shape, (3, len(self.FEATURE_NAMES)))
        self.assertEqual(batch.batch_indices.tolist(), [0, 0, 2])

    def test_all_samples_without_points_gives_empty_batch(self) -> None:
        """
        Input: a single sample with 0 points.
        Expected: a non-empty sample sequence still yields a batch, with zero rows but the full
        feature width.
        Check: the points have shape ``(0, 4)`` and the indices have shape ``(0,)``.
        """
        batch = self.collate([self.make_points(0, fill=0.0)])

        self.assertEqual(batch.points.shape, (0, len(self.FEATURE_NAMES)))
        self.assertEqual(batch.batch_indices.shape, (0,))


class TestPointCloudGTBatchToDevice(PointCloudGTBatchTestCase):
    """PointCloudGTBatch.to_device."""

    def setUp(self) -> None:
        """Collate the fixture samples once."""
        super().setUp()
        self.batch = self.collate(self.samples)

    def test_returns_new_batch_with_equal_tensors(self) -> None:
        """
        Input: the fixture batch, moved to the CPU it already lives on.
        Expected: a new PointCloudGTBatch is returned rather than the same object, and both
        fields hold the same values.
        Check: assert the identity differs and compare both fields with ``torch.equal``.
        """
        moved = self.batch.to_device(self.device)

        self.assertIsInstance(moved, PointCloudGTBatch)
        self.assertIsNot(moved, self.batch)
        self.assertTrue(torch.equal(moved.points, self.batch.points))
        self.assertTrue(torch.equal(moved.batch_indices, self.batch.batch_indices))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required to move tensors to GPU")
    def test_moves_every_tensor_to_cuda(self) -> None:
        """
        Input: the fixture batch, moved to CUDA.
        Expected: both fields of the result live on CUDA while the original stays on CPU,
        since named tuples are immutable and the move must not alias.
        Check: read ``device.type`` of both fields on both batches.
        """
        moved = self.batch.to_device(torch.device("cuda"))

        self.assertEqual(moved.points.device.type, "cuda")
        self.assertEqual(moved.batch_indices.device.type, "cuda")
        self.assertEqual(self.batch.points.device.type, "cpu")
        self.assertEqual(self.batch.batch_indices.device.type, "cpu")


class LiDARPointCloudSampleTestCase(unittest.TestCase):
    """Shared fixtures for the LiDARPointCloudSample tests."""

    def setUp(self) -> None:
        """Build one LiDAR sweep record whose three matrices are identity times 1, 2 and 3."""
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
        """
        Input: the class itself.
        Expected: the tuple exposes its five fields in the documented order, matching the
        dataset record layout.
        Check: compare ``_fields`` against the expected names.
        """
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
        """
        Input: the fixture sample.
        Expected: every field is stored as given, so the three matrices can be told apart by
        their scale.
        Check: read the path and timestamp back and compare each matrix with ``torch.equal``.
        """
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
        """
        Input: the fixture sample.
        Expected: named tuple fields cannot be reassigned after construction.
        Check: assigning to ``timestamp`` raises AttributeError.
        """
        with self.assertRaises(AttributeError):
            self.sample.timestamp = 0.0

    def test_replace_returns_new_sample(self) -> None:
        """
        Input: the fixture sample and a new timestamp of 5.0.
        Expected: ``_replace`` returns a new sample with the new timestamp, leaves the original
        untouched, and shares the unchanged matrix fields.
        Check: compare both timestamps and assert identity of the shared matrix.
        """
        replaced = self.sample._replace(timestamp=5.0)

        self.assertEqual(self.sample.timestamp, 1700000000.5)
        self.assertEqual(replaced.timestamp, 5.0)
        self.assertIs(replaced.sensor_to_ego_pose_matrix, self.sample.sensor_to_ego_pose_matrix)


if __name__ == "__main__":
    unittest.main()
