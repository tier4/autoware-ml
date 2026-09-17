"""Unit tests for the Detection3DGTBatch named tuple."""

from __future__ import annotations

import logging
import unittest

import torch

from autoware_ml.dataclasses.batch.detection3d import Detection3DGTBatch
from autoware_ml.geometry.bbox_3d.lidar_bbox3d import LidarBBoxes3D
from autoware_ml.types.geometry import Box3DCenterCoordinateType, Box3DFieldIndex


class Detection3DGTBatchTestCase(unittest.TestCase):
    """Shared fixtures for the Detection3DGTBatch tests."""

    NUM_BOX_PARAMS = len(Box3DFieldIndex)
    LABEL_NAMES = ["car", "truck", "bus", "bicycle", "pedestrian"]

    def setUp(self) -> None:
        """Build a two-sample batch input with 2 and 3 boxes and room for one more box."""
        self.device = torch.device("cpu")
        self.max_num_3d_gt_bboxes = 4
        self.samples = [self.make_bboxes(2, offset=100.0), self.make_bboxes(3, offset=200.0)]

    def make_bboxes(self, num_bboxes: int, offset: float = 0.0) -> LidarBBoxes3D:
        """
        Build LiDAR bounding boxes whose parameters, labels and point counts are all distinct so
        a test can tell which box landed in which batch slot.
        """
        bbox_params = (
            torch.arange(num_bboxes * self.NUM_BOX_PARAMS, dtype=torch.float32).reshape(
                num_bboxes, self.NUM_BOX_PARAMS
            )
            + offset
        )
        bbox_labels = torch.arange(num_bboxes, dtype=torch.int32)
        bbox_num_lidar_points = torch.arange(num_bboxes, dtype=torch.int32) * 10 + 1
        return LidarBBoxes3D(
            bbox_params=bbox_params,
            bbox_labels=bbox_labels,
            bbox_label_names=self.LABEL_NAMES,
            bbox_num_lidar_points=bbox_num_lidar_points,
            bbox_center_coordinate_type=Box3DCenterCoordinateType.GRAVITY_CENTER,
        )

    def collate(self, samples=None, status=None, max_num=None) -> Detection3DGTBatch | None:
        """Collate the fixture samples, or the given ones, with the fixture box budget."""
        return Detection3DGTBatch.collate_gt_samples(
            self.samples if samples is None else samples,
            max_num_3d_gt_bboxes=self.max_num_3d_gt_bboxes if max_num is None else max_num,
            detection3d_traffic_cone_barrier_bbox_status=status,
        )


class TestDetection3DGTBatchFields(Detection3DGTBatchTestCase):
    """The named tuple contract of Detection3DGTBatch."""

    def test_field_order(self) -> None:
        self.assertEqual(
            Detection3DGTBatch._fields,
            (
                "gt_bboxes_3d",
                "gt_labels_3d",
                "gt_valid_bboxes",
                "gt_bboxes_num_points",
                "gt_traffic_cone_barrier_bbox_status",
            ),
        )

    def test_traffic_cone_barrier_bbox_status_defaults_to_none(self) -> None:
        batch = Detection3DGTBatch(
            gt_bboxes_3d=torch.zeros((1, 2, self.NUM_BOX_PARAMS)),
            gt_labels_3d=torch.zeros((1, 2), dtype=torch.int32),
            gt_valid_bboxes=torch.zeros((1,), dtype=torch.int32),
            gt_bboxes_num_points=torch.zeros((1, 2), dtype=torch.int32),
        )

        self.assertIsNone(batch.gt_traffic_cone_barrier_bbox_status)
        self.assertEqual(
            Detection3DGTBatch._field_defaults, {"gt_traffic_cone_barrier_bbox_status": None}
        )

    def test_is_immutable(self) -> None:
        batch = self.collate()

        with self.assertRaises(AttributeError):
            batch.gt_valid_bboxes = torch.zeros(1, dtype=torch.int32)  # type: ignore


class TestCollateGtSamples(Detection3DGTBatchTestCase):
    """Detection3DGTBatch.collate_gt_samples through padding, trimming and empty inputs."""

    def test_empty_sequence_returns_none(self) -> None:
        self.assertIsNone(self.collate(samples=[]))

    def test_pads_samples_to_max_num_bboxes(self) -> None:
        batch = self.collate()

        self.assertIsNotNone(batch)
        max_num = self.max_num_3d_gt_bboxes
        assert batch is not None
        self.assertEqual(batch.gt_bboxes_3d.shape, (2, max_num, self.NUM_BOX_PARAMS))
        self.assertEqual(batch.gt_labels_3d.shape, (2, max_num))
        self.assertEqual(batch.gt_valid_bboxes.shape, (2,))
        self.assertEqual(batch.gt_bboxes_num_points.shape, (2, max_num))
        self.assertIsNone(batch.gt_traffic_cone_barrier_bbox_status)

        self.assertEqual(batch.gt_valid_bboxes.tolist(), [2, 3])

        # Valid slots carry the sample data, in order.
        self.assertTrue(torch.equal(batch.gt_bboxes_3d[0, :2], self.samples[0].bbox_params))
        self.assertTrue(torch.equal(batch.gt_bboxes_3d[1, :3], self.samples[1].bbox_params))
        self.assertEqual(batch.gt_labels_3d[0, :2].tolist(), [0, 1])
        self.assertEqual(batch.gt_labels_3d[1, :3].tolist(), [0, 1, 2])
        self.assertEqual(batch.gt_bboxes_num_points[0, :2].tolist(), [1, 11])
        self.assertEqual(batch.gt_bboxes_num_points[1, :3].tolist(), [1, 11, 21])

        # Padding slots are zero for params and point counts, and -1 for labels.
        self.assertTrue(torch.all(batch.gt_bboxes_3d[0, 2:] == 0))
        self.assertTrue(torch.all(batch.gt_bboxes_3d[1, 3:] == 0))
        self.assertEqual(batch.gt_labels_3d[0, 2:].tolist(), [-1, -1])
        self.assertEqual(batch.gt_labels_3d[1, 3:].tolist(), [-1])
        self.assertTrue(torch.all(batch.gt_bboxes_num_points[0, 2:] == 0))
        self.assertTrue(torch.all(batch.gt_bboxes_num_points[1, 3:] == 0))

    def test_output_dtypes(self) -> None:
        batch = self.collate()

        assert batch is not None
        self.assertEqual(batch.gt_bboxes_3d.dtype, torch.float32)
        self.assertEqual(batch.gt_labels_3d.dtype, torch.int32)
        self.assertEqual(batch.gt_valid_bboxes.dtype, torch.int32)
        self.assertEqual(batch.gt_bboxes_num_points.dtype, torch.int32)

    def test_trims_samples_exceeding_max_num_bboxes(self) -> None:
        sample = self.make_bboxes(5)

        with self.assertLogs(
            "autoware_ml.dataclasses.batch.detection3d", level=logging.INFO
        ) as logs:
            batch = self.collate(samples=[sample], max_num=3)

        assert batch is not None
        self.assertEqual(batch.gt_valid_bboxes.tolist(), [3])
        self.assertTrue(torch.equal(batch.gt_bboxes_3d[0], sample.bbox_params[:3]))
        self.assertEqual(batch.gt_labels_3d[0].tolist(), [0, 1, 2])
        self.assertEqual(batch.gt_bboxes_num_points[0].tolist(), [1, 11, 21])
        self.assertTrue(any("trimmed" in message for message in logs.output))

    def test_sample_without_bboxes_is_fully_padded(self) -> None:
        batch = self.collate(samples=[self.make_bboxes(0), self.make_bboxes(1)], max_num=2)

        assert batch is not None
        self.assertEqual(batch.gt_valid_bboxes.tolist(), [0, 1])
        self.assertEqual(batch.gt_labels_3d[0].tolist(), [-1, -1])
        self.assertTrue(torch.all(batch.gt_bboxes_3d[0] == 0))
        self.assertTrue(torch.all(batch.gt_bboxes_num_points[0] == 0))

    def test_exact_max_num_bboxes_has_no_padding(self) -> None:
        sample = self.make_bboxes(3)

        batch = self.collate(samples=[sample], max_num=3)

        assert batch is not None
        self.assertEqual(batch.gt_valid_bboxes.tolist(), [3])
        self.assertTrue(torch.equal(batch.gt_bboxes_3d[0], sample.bbox_params))
        self.assertFalse(torch.any(batch.gt_labels_3d == -1))


class TestCollateTrafficConeBarrierBBoxStatus(Detection3DGTBatchTestCase):
    """The optional per-sample traffic cone/barrier annotation flag."""

    def test_all_none_returns_none(self) -> None:
        self.assertIsNone(
            Detection3DGTBatch.collate_traffic_cone_barrier_bbox_status(
                [None, None], device=self.device
            )
        )

    def test_all_set_returns_bool_tensor(self) -> None:
        status = Detection3DGTBatch.collate_traffic_cone_barrier_bbox_status(
            [True, False, True], device=self.device
        )

        assert status is not None
        self.assertEqual(status.dtype, torch.bool)
        self.assertEqual(status.tolist(), [True, False, True])

    def test_mixed_flags_raise(self) -> None:
        with self.assertRaisesRegex(ValueError, "All samples must have"):
            Detection3DGTBatch.collate_traffic_cone_barrier_bbox_status(
                [True, None], device=self.device
            )

    def test_collate_gt_samples_without_flags_leaves_status_none(self) -> None:
        batch = self.collate()

        assert batch is not None
        self.assertIsNone(batch.gt_traffic_cone_barrier_bbox_status)

    def test_collate_gt_samples_with_all_none_flags_leaves_status_none(self) -> None:
        batch = self.collate(status=[None, None])

        assert batch is not None
        self.assertIsNone(batch.gt_traffic_cone_barrier_bbox_status)

    def test_collate_gt_samples_with_flags_sets_status(self) -> None:
        batch = self.collate(status=[False, True])

        assert batch is not None
        self.assertEqual(batch.gt_traffic_cone_barrier_bbox_status.dtype, torch.bool)
        self.assertEqual(batch.gt_traffic_cone_barrier_bbox_status.tolist(), [False, True])

    def test_collate_gt_samples_with_mixed_flags_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "All samples must have"):
            self.collate(status=[True, None])

    def test_collate_gt_samples_with_wrong_flag_count_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "one entry per sample"):
            self.collate(status=[True])


class TestToDevice(Detection3DGTBatchTestCase):
    """Detection3DGTBatch.to_device."""

    def setUp(self) -> None:
        """Collate the fixture samples once, with and without the traffic cone flag."""
        super().setUp()
        self.batch_with_status = self.collate(status=[True, False])
        self.batch_without_status = self.collate()

    def test_returns_new_batch_with_equal_tensors(self) -> None:
        moved = self.batch_with_status.to_device(self.device)  # type: ignore

        self.assertIsInstance(moved, Detection3DGTBatch)
        self.assertIsNot(moved, self.batch_with_status)
        assert self.batch_with_status is not None
        for original, result in zip(self.batch_with_status, moved):
            self.assertTrue(torch.equal(original, result))  # type: ignore

    def test_preserves_none_status(self) -> None:
        moved = self.batch_without_status.to_device(self.device)  # type: ignore

        self.assertIsNone(moved.gt_traffic_cone_barrier_bbox_status)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required to move tensors to GPU")
    def test_moves_every_tensor_to_cuda(self) -> None:
        moved = self.batch_with_status.to_device(torch.device("cuda"))  # type: ignore

        for tensor in moved:
            self.assertEqual(tensor.device.type, "cuda")  # type: ignore
        # The original batch stays on CPU.
        assert self.batch_with_status is not None
        for tensor in self.batch_with_status:
            self.assertEqual(tensor.device.type, "cpu")  # type: ignore


if __name__ == "__main__":
    unittest.main()
