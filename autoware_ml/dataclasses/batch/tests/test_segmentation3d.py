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
        self.assertEqual(Segmentation3DGTSample._fields, ("gt_semantic_mask",))

    def test_holds_semantic_mask(self) -> None:
        self.assertIs(self.sample.gt_semantic_mask, self.mask)
        self.assertEqual(self.sample.gt_semantic_mask.shape, (3, 1))
        self.assertEqual(self.sample.gt_semantic_mask.dtype, np.int32)

    def test_positional_construction_and_unpacking(self) -> None:
        mask = self.make_mask(4)

        sample = Segmentation3DGTSample(mask)
        (unpacked,) = sample

        self.assertIs(unpacked, mask)
        self.assertEqual(len(sample), 1)

    def test_is_immutable(self) -> None:
        with self.assertRaises(AttributeError):
            self.sample.gt_semantic_mask = self.make_mask(3, fill=1)  # type: ignore

    def test_replace_returns_new_sample(self) -> None:
        new_mask = self.make_mask(3, fill=1)

        replaced = self.sample._replace(gt_semantic_mask=new_mask)

        self.assertIs(self.sample.gt_semantic_mask, self.mask)
        self.assertIs(replaced.gt_semantic_mask, new_mask)


if __name__ == "__main__":
    unittest.main()
