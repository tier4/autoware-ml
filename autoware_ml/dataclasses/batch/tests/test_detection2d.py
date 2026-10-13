"""Unit tests for the per-camera 2D detection ground truth."""

from __future__ import annotations

import unittest

import torch

from autoware_ml.dataclasses.batch.detection2d import Detection2DGTBatch, Detection2DGTSample


def build_sample(counts: list[int], offset: float = 0.0) -> Detection2DGTSample:
    """2D ground truth with the given number of boxes per camera, values tagged by ``offset``."""
    return Detection2DGTSample(
        gt_bboxes_2d=[
            torch.full((count, 4), offset + camera) for camera, count in enumerate(counts)
        ],
        gt_centers_2d=[
            torch.full((count, 2), offset - camera) for camera, count in enumerate(counts)
        ],
        gt_labels_2d=[
            torch.arange(count, dtype=torch.int32) + camera for camera, count in enumerate(counts)
        ],
    )


class TestDetection2DGTBatch(unittest.TestCase):
    """Collation of the per-camera 2D boxes of a batch."""

    def setUp(self) -> None:
        """Two samples of two cameras holding 2, 0 and 1, 3 boxes."""
        self.samples = [build_sample([2, 0], offset=10.0), build_sample([1, 3], offset=20.0)]

    def test_pads_every_camera_to_the_largest_one(self) -> None:
        batch = Detection2DGTBatch.collate_gt_samples(self.samples)

        self.assertEqual(tuple(batch.gt_bboxes_2d.shape), (2, 2, 3, 4))
        self.assertEqual(tuple(batch.gt_centers_2d.shape), (2, 2, 3, 2))
        self.assertEqual(batch.gt_valid_bboxes_2d.tolist(), [[2, 0], [1, 3]])
        self.assertEqual(batch.gt_labels_2d[0, 0].tolist(), [0, 1, -1])
        self.assertEqual(batch.gt_labels_2d[0, 1].tolist(), [-1, -1, -1])
        self.assertTrue((batch.gt_bboxes_2d[0, 0, 2] == 0).all())

    def test_unpads_back_to_every_camera_of_every_sample(self) -> None:
        batch = Detection2DGTBatch.collate_gt_samples(self.samples)

        for name, collated in (
            ("gt_bboxes_2d", batch.valid_bboxes_2d()),
            ("gt_centers_2d", batch.valid_centers_2d()),
            ("gt_labels_2d", batch.valid_labels_2d()),
        ):
            for sample, collated_sample in zip(self.samples, collated, strict=True):
                for expected, unpadded in zip(getattr(sample, name), collated_sample, strict=True):
                    torch.testing.assert_close(unpadded, expected, check_dtype=False)
        # The consumers index with the labels, which come back widened to int64.
        self.assertEqual(batch.valid_labels_2d()[1][1].dtype, torch.int64)

    def test_a_batch_without_boxes_keeps_an_empty_box_axis(self) -> None:
        batch = Detection2DGTBatch.collate_gt_samples([build_sample([0, 0])])

        self.assertEqual(tuple(batch.gt_bboxes_2d.shape), (1, 2, 0, 4))
        self.assertEqual(batch.valid_bboxes_2d()[0][1].shape, (0, 4))

    def test_rejects_samples_with_different_cameras(self) -> None:
        with self.assertRaisesRegex(ValueError, "cameras"):
            Detection2DGTBatch.collate_gt_samples([build_sample([1, 1]), build_sample([1])])

    def test_rejects_an_empty_batch(self) -> None:
        with self.assertRaises(ValueError):
            Detection2DGTBatch.collate_gt_samples([])


if __name__ == "__main__":
    unittest.main()
