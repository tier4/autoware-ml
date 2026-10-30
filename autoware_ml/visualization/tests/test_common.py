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

"""Tests for shared visualization conversion helpers."""

from __future__ import annotations

import unittest

import numpy as np

from autoware_ml.visualization.colors import build_label_palette
from autoware_ml.visualization.common import (
    build_class_annotation_context,
    build_sample_metadata_events,
    build_scene_extent_event,
    ensure_image_uint8,
    ensure_xyz,
    format_class_label,
    resolve_palette_size,
)


class TestResolvePaletteSize(unittest.TestCase):
    """The palette covers every declared class and every observed label."""

    def test_covers_observed_labels(self) -> None:
        """The largest observed label sets the palette size."""
        self.assertEqual(resolve_palette_size([np.array([0, 3], dtype=np.int64)]), 4)

    def test_covers_all_declared_classes(self) -> None:
        """A sample using two of ten classes must still publish a ten-entry legend."""
        palette_size = resolve_palette_size(
            [np.array([0, 1], dtype=np.int64)],
            class_names=[f"class-{index}" for index in range(10)],
        )
        self.assertEqual(palette_size, 10)

    def test_ignores_negative_and_empty_labels(self) -> None:
        """Negative labels, empty arrays and absent arrays add no palette entry."""
        self.assertEqual(resolve_palette_size([np.array([-1, -1], dtype=np.int64)]), 0)
        self.assertEqual(resolve_palette_size([np.array([], dtype=np.int64), None]), 0)

    def test_takes_the_maximum_across_arrays(self) -> None:
        """The palette covers the largest label of every array, past the declared classes."""
        palette_size = resolve_palette_size(
            [np.array([1], dtype=np.int64), np.array([7], dtype=np.int64)],
            class_names=["a", "b"],
        )
        self.assertEqual(palette_size, 8)


class TestClassLabels(unittest.TestCase):
    """Class legends and labels name every class, falling back to the index."""

    def test_annotation_context_labels_every_palette_entry(self) -> None:
        """The legend has one RGBA entry per palette color, named or indexed."""
        context = build_class_annotation_context("root", build_label_palette(3), ["road", "car"])

        self.assertIsNotNone(context)
        self.assertEqual(context.path, "root")
        self.assertEqual([annotation.id for annotation in context.annotations], [0, 1, 2])
        self.assertEqual(
            [annotation.label for annotation in context.annotations], ["road", "car", "2"]
        )
        for annotation in context.annotations:
            with self.subTest(id=annotation.id):
                self.assertEqual(len(annotation.color), 4)

    def test_annotation_context_is_none_for_an_empty_palette(self) -> None:
        """An empty palette produces no legend."""
        self.assertIsNone(build_class_annotation_context("root", build_label_palette(0)))

    def test_format_class_label_appends_score_when_present(self) -> None:
        """The label is the class name, then the score when given, else the index."""
        self.assertEqual(format_class_label(1, ["road", "car"]), "car")
        self.assertEqual(format_class_label(1, ["road", "car"], 0.5), "car (0.50)")
        self.assertEqual(format_class_label(9, ["road", "car"]), "9")


class TestSampleMetadata(unittest.TestCase):
    """The sample name is logged under the scene metadata."""

    def test_is_empty_without_a_name(self) -> None:
        """Without a name nothing is logged; with one, it is logged under the metadata."""
        self.assertEqual(len(build_sample_metadata_events("root", None)), 0)
        self.assertEqual(
            build_sample_metadata_events("root", "sample-1")[0].path, "root/meta/sample"
        )


class TestArrayConversion(unittest.TestCase):
    """Points and images are normalized to the layouts the viewer expects."""

    def test_ensure_xyz_rejects_non_point_shapes(self) -> None:
        """Points need at least the x, y and z columns."""
        with self.assertRaisesRegex(ValueError, r"shape \(N, >=3\)"):
            ensure_xyz(np.zeros((4, 2), dtype=np.float32))

    def test_ensure_image_uint8_transposes_channel_first_images(self) -> None:
        """A CHW image becomes HWC."""
        image = ensure_image_uint8(np.zeros((3, 32, 64), dtype=np.uint8))
        self.assertEqual(image.shape, (32, 64, 3))

    def test_ensure_image_uint8_scales_float_images(self) -> None:
        """A floating-point image in [0, 1] is scaled to the full uint8 range."""
        image = ensure_image_uint8(np.ones((8, 8, 3), dtype=np.float32))
        self.assertEqual(image.dtype, np.uint8)
        self.assertEqual(int(image.max()), 255)


class TestSceneExtent(unittest.TestCase):
    """The scene extent spans the points and every box of a frame."""

    def test_spans_points_and_every_box_yaw(self) -> None:
        """The two invisible corners bound the points and the boxes at any yaw."""
        points = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]], dtype=np.float32)
        boxes = np.array([[10.0, 0.0, 0.0, 2.0, 2.0, 2.0, 0.3]], dtype=np.float32)

        event = build_scene_extent_event(points, (boxes,), root_path="scene")

        self.assertIsNotNone(event)
        self.assertEqual(event.path, "scene/meta/extent")
        half_diagonal = np.sqrt(12.0) / 2.0
        np.testing.assert_allclose(
            event.positions[0], [0.0, -half_diagonal, -half_diagonal], atol=1e-5
        )
        np.testing.assert_allclose(event.positions[1], [10.0 + half_diagonal, 2.0, 3.0], atol=1e-5)
        self.assertIsNotNone(event.colors)
        self.assertEqual(event.colors.shape, (2, 4))
        self.assertFalse(event.colors.any(), "the extent must stay invisible")
        self.assertAlmostEqual(event.radii, 1e-3)

    def test_is_empty_without_geometry(self) -> None:
        """A frame without points or boxes has no extent."""
        cases = {
            "nothing": (None, ()),
            "empty arrays": (
                np.zeros((0, 3), dtype=np.float32),
                (np.zeros((0, 7), dtype=np.float32),),
            ),
        }
        for case, (points, boxes) in cases.items():
            with self.subTest(case=case):
                self.assertIsNone(build_scene_extent_event(points, boxes, root_path="scene"))


if __name__ == "__main__":
    unittest.main()
