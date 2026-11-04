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

"""Tests for the LiDAR point-cloud layers shared by the task adapters."""

from __future__ import annotations

import unittest

import numpy as np

from autoware_ml.visualization.sensors.lidar import (
    build_lidar_reference_events,
    point_cloud_colors,
)


class TestPointCloudColors(unittest.TestCase):
    """Point colors follow the requested color mode."""

    def test_can_render_intensity(self) -> None:
        """Different intensities give different RGBA colors."""
        colors = point_cloud_colors(
            np.array([[0.0, 0.0, 0.0, 2.0], [1.0, 0.0, 0.0, 8.0]], dtype=np.float32),
            "intensity",
        )

        self.assertIsNotNone(colors)
        self.assertEqual(colors.shape, (2, 4))
        self.assertNotEqual(tuple(colors[0]), tuple(colors[1]))

    def test_requires_intensity_column(self) -> None:
        """The intensity mode needs a fourth column."""
        with self.assertRaisesRegex(ValueError, "intensity column"):
            point_cloud_colors(np.zeros((1, 3), dtype=np.float32), "intensity")

    def test_rejects_unknown_mode(self) -> None:
        """Only the declared color modes are accepted."""
        with self.assertRaisesRegex(ValueError, "point color mode must be"):
            point_cloud_colors(np.zeros((1, 4), dtype=np.float32), "height")


class TestLidarReferenceEvents(unittest.TestCase):
    """The reference layers are the solid cloud and, with intensities, the intensity cloud."""

    def test_logs_solid_and_intensity_layers(self) -> None:
        """Points with an intensity column give both layers under the scene root."""
        events = build_lidar_reference_events(
            np.zeros((2, 4), dtype=np.float32), root_path="scene", point_radius=0.05
        )

        self.assertEqual(
            [event.path for event in events], ["scene/lidar/solid", "scene/lidar/intensity"]
        )
        for event in events:
            with self.subTest(path=event.path):
                self.assertEqual(event.positions.shape, (2, 3))
                self.assertEqual(event.colors.shape, (2, 4))
                self.assertAlmostEqual(event.radii, 0.05)

    def test_logs_only_the_solid_layer_without_intensity(self) -> None:
        """Points without an intensity column give the solid layer alone."""
        events = build_lidar_reference_events(
            np.zeros((2, 3), dtype=np.float32), root_path="scene", point_radius=0.05
        )

        self.assertEqual([event.path for event in events], ["scene/lidar/solid"])


if __name__ == "__main__":
    unittest.main()
