"""Unit tests for the per-camera 2D detection ground truth."""

from __future__ import annotations

import unittest

import torch

from autoware_ml.dataclasses.batch.detection2d import Detection2DGTBatch, Detection2DGTSample
from autoware_ml.types.geometry import Box2DFieldIndex


def build_sample(counts: list[int], offset: float = 0.0) -> Detection2DGTSample:
    """2D ground truth with the given number of boxes per camera, values tagged by ``offset``.

    The box edges of camera ``c`` hold ``offset + c`` and its centers ``offset - c``.
    """
    boxes = []
    for camera, count in enumerate(counts):
        camera_boxes = torch.full((count, len(Box2DFieldIndex)), offset + camera)
        camera_boxes[:, Box2DFieldIndex.CENTER_X :] = offset - camera
        boxes.append(camera_boxes)
    return Detection2DGTSample(
        gt_bboxes_2d=boxes,
        gt_labels_2d=[
            torch.arange(count, dtype=torch.int32) + camera for camera, count in enumerate(counts)
        ],
    )


class TestDetection2DGTBatch(unittest.TestCase):
    """Collation of the per-camera 2D boxes of a batch."""

    MAX_NUM_2D_GT_BBOXES = 4

    def setUp(self) -> None:
        """Two samples of two cameras holding 2, 0 and 1, 3 boxes."""
        self.samples = [build_sample([2, 0], offset=10.0), build_sample([1, 3], offset=20.0)]

    def collate(self, samples: list[Detection2DGTSample]) -> Detection2DGTBatch:
        """Collate the samples with the padding size of the tests."""
        return Detection2DGTBatch.collate_gt_samples(
            samples, max_num_2d_gt_bboxes=self.MAX_NUM_2D_GT_BBOXES
        )

    def test_pads_every_camera_to_max_num_2d_gt_bboxes(self) -> None:
        """
        Input: two samples of two cameras holding 2, 0 and 1, 3 boxes, collated with
        max_num_2d_gt_bboxes 4.
        Expected: every camera is padded to the fixed 4 boxes rather than to the largest camera
        of the batch (3), so the batch shape does not depend on the samples.
        Check: the box and label shapes, the valid counts, and zero boxes with label -1 past
        the valid count.
        """
        batch = self.collate(self.samples)

        self.assertEqual(tuple(batch.gt_bboxes_2d.shape), (2, 2, 4, len(Box2DFieldIndex)))
        self.assertEqual(tuple(batch.gt_labels_2d.shape), (2, 2, 4))
        self.assertEqual(batch.gt_valid_bboxes_2d.tolist(), [[2, 0], [1, 3]])
        self.assertEqual(batch.gt_labels_2d[0, 0].tolist(), [0, 1, -1, -1])
        self.assertEqual(batch.gt_labels_2d[0, 1].tolist(), [-1, -1, -1, -1])
        self.assertTrue((batch.gt_bboxes_2d[0, 0, 2:] == 0).all())
        self.assertTrue((batch.gt_bboxes_2d[1, 1, 3] == 0).all())

    def test_keeps_the_first_boxes_of_a_camera_holding_more_than_the_maximum(self) -> None:
        """
        Input: a camera holding 3 boxes, collated with max_num_2d_gt_bboxes 2.
        Expected: the camera keeps its first 2 boxes and labels in the order the sample lists
        them, the third is left out of the batch, and the cut is logged.
        Check: the valid count, the kept boxes and labels, and the log naming the camera.
        """
        sample = build_sample([1, 3], offset=20.0)
        sample = sample._replace(
            gt_bboxes_2d=[
                sample.gt_bboxes_2d[0],
                sample.gt_bboxes_2d[1] + torch.arange(3.0).view(3, 1),
            ]
        )

        with self.assertLogs("autoware_ml.dataclasses.batch.detection2d", "INFO") as logs:
            batch = Detection2DGTBatch.collate_gt_samples([sample], max_num_2d_gt_bboxes=2)

        self.assertEqual(batch.gt_valid_bboxes_2d.tolist(), [[1, 2]])
        torch.testing.assert_close(batch.gt_bboxes_2d[0, 1], sample.gt_bboxes_2d[1][:2])
        self.assertEqual(batch.gt_labels_2d[0, 1].tolist(), sample.gt_labels_2d[1][:2].tolist())
        self.assertIn("Sample 0, camera 1 holds 3 2D boxes", logs.output[0])

    def test_unpads_back_to_every_camera_of_every_sample(self) -> None:
        batch = self.collate(self.samples)

        for name, collated in (
            ("gt_bboxes_2d", batch.valid_bboxes_2d()),
            ("gt_labels_2d", batch.valid_labels_2d()),
        ):
            for sample, collated_sample in zip(self.samples, collated, strict=True):
                for expected, unpadded in zip(getattr(sample, name), collated_sample, strict=True):
                    torch.testing.assert_close(unpadded, expected, check_dtype=False)
        # The consumers index with the labels, which come back widened to int64.
        self.assertEqual(batch.valid_labels_2d()[1][1].dtype, torch.int64)

    def test_a_batch_without_boxes_keeps_the_fixed_size(self) -> None:
        """
        Input: one sample of two cameras holding no box.
        Expected: the batch keeps its fixed size, every row is padding, and unpadding gives
        every camera an empty box set.
        Check: the box shape, the valid counts and the unpadded box shape.
        """
        batch = self.collate([build_sample([0, 0])])

        self.assertEqual(tuple(batch.gt_bboxes_2d.shape), (1, 2, 4, len(Box2DFieldIndex)))
        self.assertEqual(batch.gt_valid_bboxes_2d.tolist(), [[0, 0]])
        self.assertEqual(batch.valid_bboxes_2d()[0][1].shape, (0, len(Box2DFieldIndex)))

    def test_rejects_samples_with_different_cameras(self) -> None:
        with self.assertRaisesRegex(ValueError, "cameras"):
            self.collate([build_sample([1, 1]), build_sample([1])])

    def test_rejects_a_camera_whose_labels_differ_from_its_boxes(self) -> None:
        # Camera 1 of sample 1 holds 2 boxes. A single label used to broadcast over both of
        # them, and 3 labels failed on the copy without naming the sample or camera.
        for num_labels in (1, 3):
            with self.subTest(labels=num_labels):
                sample = build_sample([1, 2])
                sample = sample._replace(
                    gt_labels_2d=[
                        sample.gt_labels_2d[0],
                        torch.zeros(num_labels, dtype=torch.int32),
                    ],
                )

                with self.assertRaisesRegex(
                    ValueError, f"sample 1, camera 1 differ in count: 2 and {num_labels}"
                ):
                    self.collate([build_sample([1, 1]), sample])

    def test_rejects_boxes_not_laid_out_by_box2d_field_index(self) -> None:
        """
        Input: a camera whose boxes carry 4 columns (the edges alone) or 1 column.
        Expected: every box row holds the 6 Box2DFieldIndex columns. A single column would
        broadcast over the row and 4 columns would fail on the copy without naming the camera,
        so both are refused before any copy.
        Check: a ValueError naming the sample, the camera and the shape is raised.
        """
        for num_columns in (4, 1):
            with self.subTest(columns=num_columns):
                sample = build_sample([1, 2])
                sample = sample._replace(
                    gt_bboxes_2d=[sample.gt_bboxes_2d[0], torch.zeros(2, num_columns)]
                )

                with self.assertRaisesRegex(
                    ValueError, rf"sample 1, camera 1 have shape \(2, {num_columns}\)"
                ):
                    self.collate([build_sample([1, 1]), sample])

    def test_rejects_an_empty_batch(self) -> None:
        with self.assertRaises(ValueError):
            self.collate([])


if __name__ == "__main__":
    unittest.main()
