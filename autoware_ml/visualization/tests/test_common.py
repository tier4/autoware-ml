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

import numpy as np
import pytest

from autoware_ml.visualization.colors import build_label_palette
from autoware_ml.visualization.common import (
    build_scene_extent_event,
    build_class_annotation_context,
    build_sample_metadata_events,
    ensure_image_uint8,
    ensure_xyz,
    format_class_label,
    point_cloud_colors,
    resolve_palette_size,
)


def test_resolve_palette_size_covers_observed_labels() -> None:
    assert resolve_palette_size([np.array([0, 3], dtype=np.int64)]) == 4


def test_resolve_palette_size_covers_all_declared_classes() -> None:
    """A sample using two of ten classes must still publish a ten-entry legend."""
    palette_size = resolve_palette_size(
        [np.array([0, 1], dtype=np.int64)],
        class_names=[f"class-{index}" for index in range(10)],
    )
    assert palette_size == 10


def test_resolve_palette_size_ignores_negative_and_empty_labels() -> None:
    assert resolve_palette_size([np.array([-1, -1], dtype=np.int64)]) == 0
    assert resolve_palette_size([np.array([], dtype=np.int64), None]) == 0


def test_resolve_palette_size_takes_the_maximum_across_arrays() -> None:
    palette_size = resolve_palette_size(
        [np.array([1], dtype=np.int64), np.array([7], dtype=np.int64)],
        class_names=["a", "b"],
    )
    assert palette_size == 8


def test_build_class_annotation_context_labels_every_palette_entry() -> None:
    context = build_class_annotation_context("root", build_label_palette(3), ["road", "car"])

    assert context is not None
    assert context.path == "root"
    assert [annotation.id for annotation in context.annotations] == [0, 1, 2]
    assert [annotation.label for annotation in context.annotations] == [
        "road",
        "car",
        "2",
    ]
    assert all(len(annotation.color) == 4 for annotation in context.annotations)


def test_build_class_annotation_context_returns_none_for_empty_palette() -> None:
    assert build_class_annotation_context("root", build_label_palette(0)) is None


def test_format_class_label_appends_score_when_present() -> None:
    assert format_class_label(1, ["road", "car"]) == "car"
    assert format_class_label(1, ["road", "car"], 0.5) == "car (0.50)"
    assert format_class_label(9, ["road", "car"]) == "9"


def test_build_sample_metadata_events_is_empty_without_a_name() -> None:
    assert build_sample_metadata_events("root", None) == []
    assert build_sample_metadata_events("root", "sample-1")[0].path == "root/meta/sample"


def test_ensure_xyz_rejects_non_point_shapes() -> None:
    with pytest.raises(ValueError, match=r"shape \(N, >=3\)"):
        ensure_xyz(np.zeros((4, 2), dtype=np.float32))


def test_ensure_image_uint8_transposes_channel_first_images() -> None:
    image = ensure_image_uint8(np.zeros((3, 32, 64), dtype=np.uint8))
    assert image.shape == (32, 64, 3)


def test_ensure_image_uint8_scales_float_images() -> None:
    image = ensure_image_uint8(np.ones((8, 8, 3), dtype=np.float32))
    assert image.dtype == np.uint8
    assert int(image.max()) == 255


def test_point_cloud_colors_can_render_intensity() -> None:
    colors = point_cloud_colors(
        np.array([[0.0, 0.0, 0.0, 2.0], [1.0, 0.0, 0.0, 8.0]], dtype=np.float32),
        "intensity",
    )
    assert colors is not None
    assert colors.shape == (2, 4)
    assert tuple(colors[0]) != tuple(colors[1])


def test_point_cloud_colors_requires_intensity_column() -> None:
    with pytest.raises(ValueError, match="intensity column"):
        point_cloud_colors(np.zeros((1, 3), dtype=np.float32), "intensity")


def test_build_scene_extent_event_spans_points_and_every_box_yaw() -> None:
    points = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]], dtype=np.float32)
    boxes = np.array([[10.0, 0.0, 0.0, 2.0, 2.0, 2.0, 0.3]], dtype=np.float32)

    event = build_scene_extent_event(points, (boxes, None))

    assert event is not None
    assert event.path == "scene/meta/extent"
    half_diagonal = np.sqrt(12.0) / 2.0
    np.testing.assert_allclose(event.positions[0], [0.0, -half_diagonal, -half_diagonal], atol=1e-5)
    np.testing.assert_allclose(event.positions[1], [10.0 + half_diagonal, 2.0, 3.0], atol=1e-5)
    assert event.colors is not None
    assert event.colors.shape == (2, 4)
    assert not event.colors.any(), "the extent must stay invisible"
    assert event.radii == pytest.approx(1e-3)


def test_build_scene_extent_event_is_empty_without_geometry() -> None:
    assert build_scene_extent_event(None) is None
    assert (
        build_scene_extent_event(
            np.zeros((0, 3), dtype=np.float32), (np.zeros((0, 7), dtype=np.float32), None)
        )
        is None
    )
