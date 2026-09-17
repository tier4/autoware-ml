"""Unit tests for the Segmentation3DGTSample named tuple."""

from __future__ import annotations

import unittest

import numpy as np

from autoware_ml.dataclasses.batch.segmentation3d import Segmentation3DGTSample


class Segmentation3DGTSampleTestCase(unittest.TestCase):
    """Shared fixtures for the Segmentation3DGTSample tests."""

    def setUp(self) -> None:
        """Build a three-point semantic mask and the sample wrapping it."""
        self.mask = np.array([[0], [1], [2]], dtype=np.int32)
        self.sample = Segmentation3DGTSample(gt_semantic_mask=self.mask)

    def make_mask(self, num_points: int, fill: int = 0) -> np.ndarray:
        """Build an ``(num_points, 1)`` int32 semantic mask filled with ``fill``."""
        return np.full((num_points, 1), fill, dtype=np.int32)


class TestSegmentation3DGTSample(Segmentation3DGTSampleTestCase):
    """The named tuple contract of Segmentation3DGTSample."""

    def test_field_order(self) -> None:
        """
        Input: the class itself.
        Expected: the tuple has exactly one field, the semantic mask.
        Check: compare ``_fields`` against the single expected name.
        """
        self.assertEqual(Segmentation3DGTSample._fields, ("gt_semantic_mask",))

    def test_holds_semantic_mask(self) -> None:
        """
        Input: the fixture sample built from a ``(3, 1)`` int32 mask.
        Expected: the sample stores the very same array without copying or casting.
        Check: assert identity with the source array and read its shape and dtype.
        """
        self.assertIs(self.sample.gt_semantic_mask, self.mask)
        self.assertEqual(self.sample.gt_semantic_mask.shape, (3, 1))
        self.assertEqual(self.sample.gt_semantic_mask.dtype, np.int32)

    def test_positional_construction_and_unpacking(self) -> None:
        """
        Input: a four-point mask passed positionally.
        Expected: the tuple can be built without keywords and unpacked back into that array.
        Check: unpack the single element, assert identity, and assert the length is 1.
        """
        mask = self.make_mask(4)

        sample = Segmentation3DGTSample(mask)
        (unpacked,) = sample

        self.assertIs(unpacked, mask)
        self.assertEqual(len(sample), 1)

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
        Check: assert identity of each sample's mask against the array it should hold.
        """
        new_mask = self.make_mask(3, fill=1)

        replaced = self.sample._replace(gt_semantic_mask=new_mask)

        self.assertIs(self.sample.gt_semantic_mask, self.mask)
        self.assertIs(replaced.gt_semantic_mask, new_mask)


if __name__ == "__main__":
    unittest.main()
