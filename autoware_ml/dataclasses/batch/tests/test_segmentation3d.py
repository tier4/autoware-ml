"""Unit tests for the Segmentation3DGTSample named tuple."""

from __future__ import annotations

import unittest

import torch

from autoware_ml.dataclasses.batch.segmentation3d import Segmentation3DGTSample


class Segmentation3DGTSampleTestCase(unittest.TestCase):
    """Shared fixtures for the Segmentation3DGTSample tests."""

    def setUp(self) -> None:
        """Build a three-point semantic mask and the sample wrapping it."""
        self.mask = torch.tensor([0, 1, 2], dtype=torch.int64)
        self.sample = Segmentation3DGTSample(gt_semantic_mask=self.mask, ignore_index=-1)

    def make_mask(self, num_points: int, fill: int = 0) -> torch.Tensor:
        """Build a ``(num_points,)`` int64 semantic mask filled with ``fill``."""
        return torch.full((num_points,), fill, dtype=torch.int64)


class TestSegmentation3DGTSample(Segmentation3DGTSampleTestCase):
    """The named tuple contract of Segmentation3DGTSample."""

    def test_field_order(self) -> None:
        """
        Input: the class itself.
        Expected: the tuple holds the semantic mask and the ignore index, in that order.
        Check: compare ``_fields`` against the expected names.
        """
        self.assertEqual(Segmentation3DGTSample._fields, ("gt_semantic_mask", "ignore_index"))

    def test_holds_semantic_mask(self) -> None:
        """
        Input: the fixture sample built from a ``(3,)`` int64 mask.
        Expected: the sample stores the very same tensor without copying or casting, with one
        label per point.
        Check: assert identity with the source tensor and read its shape and dtype.
        """
        self.assertIs(self.sample.gt_semantic_mask, self.mask)
        self.assertEqual(self.sample.gt_semantic_mask.shape, (3,))
        self.assertEqual(self.sample.gt_semantic_mask.dtype, torch.int64)

    def test_holds_ignore_index(self) -> None:
        """
        Input: the fixture sample built with an ignore index of ``-1``.
        Expected: the ignore index travels with the mask, so a transform that appends points
        can label them ignored without reading the taxonomy.
        Check: read the field back.
        """
        self.assertEqual(self.sample.ignore_index, -1)

    def test_positional_construction_and_unpacking(self) -> None:
        """
        Input: a four-point mask and an ignore index passed positionally.
        Expected: the tuple can be built without keywords and unpacked back into its fields.
        Check: unpack both elements, assert identity of the mask, and assert the length is 2.
        """
        mask = self.make_mask(4)

        sample = Segmentation3DGTSample(mask, -1)
        unpacked_mask, unpacked_ignore_index = sample

        self.assertIs(unpacked_mask, mask)
        self.assertEqual(unpacked_ignore_index, -1)
        self.assertEqual(len(sample), 2)

    def test_is_immutable(self) -> None:
        """
        Input: the fixture sample.
        Expected: the mask field cannot be reassigned after construction.
        Check: assigning a new mask raises AttributeError.
        """
        with self.assertRaises(AttributeError):
            self.sample.gt_semantic_mask = self.make_mask(3, fill=1)

    def test_replace_returns_new_sample(self) -> None:
        """
        Input: the fixture sample and a new mask filled with ones.
        Expected: ``_replace`` returns a new sample carrying the new mask while the original
        keeps the old one.
        Check: assert identity of each sample's mask against the tensor it should hold.
        """
        new_mask = self.make_mask(3, fill=1)

        replaced = self.sample._replace(gt_semantic_mask=new_mask)

        self.assertIs(self.sample.gt_semantic_mask, self.mask)
        self.assertIs(replaced.gt_semantic_mask, new_mask)


class TestSegmentation3DGTSampleAlignment(Segmentation3DGTSampleTestCase):
    """The helpers that keep the mask aligned with a changing point cloud."""

    def test_remove_labels_keeps_the_masked_points(self) -> None:
        """
        Input: the fixture sample and a mask keeping the first and the last point.
        Expected: the labels of the dropped points go with them, so the mask still holds one
        label per surviving point.
        Check: read the remaining labels and the untouched ignore index.
        """
        kept = self.sample.remove_labels(torch.tensor([True, False, True]))

        self.assertEqual(kept.gt_semantic_mask.tolist(), [0, 2])
        self.assertEqual(kept.ignore_index, -1)

    def test_reorder_labels_follows_the_permutation(self) -> None:
        """
        Input: the fixture sample and the reversing permutation.
        Expected: the labels follow the points, so a shuffled cloud keeps its labels.
        Check: read the reordered labels.
        """
        reordered = self.sample.reorder_labels(torch.tensor([2, 1, 0], dtype=torch.int32))

        self.assertEqual(reordered.gt_semantic_mask.tolist(), [2, 1, 0])

    def test_append_ignored_labels_covers_the_appended_points(self) -> None:
        """
        Input: the fixture sample and two appended points.
        Expected: the appended points take the ignore index, so points a sweep brings in shape
        the geometry without reaching the loss or the metrics.
        Check: read the extended labels and their dtype.
        """
        extended = self.sample.append_ignored_labels(2)

        self.assertEqual(extended.gt_semantic_mask.tolist(), [0, 1, 2, -1, -1])
        self.assertEqual(extended.gt_semantic_mask.dtype, torch.int64)

    def test_append_ignored_labels_without_points_keeps_the_mask(self) -> None:
        """
        Input: the fixture sample and no appended points.
        Expected: a sweep that brought nothing leaves the mask as it was.
        Check: read the labels back unchanged.
        """
        kept = self.sample.append_ignored_labels(0)

        self.assertEqual(kept.gt_semantic_mask.tolist(), [0, 1, 2])


if __name__ == "__main__":
    unittest.main()
