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

"""The raster membership test must agree with shapely point for point."""

from __future__ import annotations

import unittest

import numpy as np
import shapely
import torch
from shapely.geometry import Polygon

from autoware_ml.metrics.geometry.lanelet import LaneletMap
from autoware_ml.metrics.geometry.region_raster import RegionRaster


def _star(center: tuple[float, float], radius: float, spikes: int, rotation: float) -> Polygon:
    """A concave star polygon, edges at every angle."""
    angles = rotation + np.arange(2 * spikes) * np.pi / spikes
    radii = np.where(np.arange(2 * spikes) % 2 == 0, radius, 0.45 * radius)
    return Polygon(
        np.column_stack([center[0] + radii * np.cos(angles), center[1] + radii * np.sin(angles)])
    )


def _region():
    """A multipolygon with a hole, concave parts, thin slivers and a far-away piece."""
    road = Polygon(
        [(0, 0), (60, 0), (60, 8), (35, 8), (35, 30), (27, 30), (27, 8), (0, 8)],
        holes=[[(40, 2), (44, 2), (44, 5), (40, 5)]],
    )
    sliver = Polygon([(0, 9), (60, 9.07), (60, 9.1), (0, 9.03)])
    return shapely.union_all(
        [road, sliver, _star((80.3, 20.7), 9.0, 7, 0.3), _star((520.0, -300.0), 3.0, 5, 1.1)]
    )


def _probe_points(region, rng: np.random.Generator) -> np.ndarray:
    """Uniform points, points hugging every edge and vertex, and points on cell corners."""
    min_x, min_y, max_x, max_y = region.bounds
    uniform = rng.uniform((min_x - 5, min_y - 5), (max_x + 5, max_y + 5), size=(20000, 2))
    coords = shapely.get_coordinates(shapely.get_rings(shapely.get_parts(region)))
    near_vertices = np.repeat(coords, 20, axis=0) + rng.normal(0, 1e-3, size=(len(coords) * 20, 2))
    t = rng.uniform(0, 1, size=(len(coords) - 1, 1))
    on_edges = coords[:-1] + t * (coords[1:] - coords[:-1])
    near_edges = np.repeat(on_edges, 10, axis=0) + rng.normal(0, 1e-4, size=(len(on_edges) * 10, 2))
    grid = np.stack(np.meshgrid(np.arange(-2, 62, 0.5), np.arange(-2, 32, 0.5)), axis=-1).reshape(
        -1, 2
    )
    return np.concatenate([uniform, near_vertices, near_edges, grid])


class RegionRasterTest(unittest.TestCase):
    """Exactness against shapely.contains on NumPy and torch inputs."""

    def setUp(self) -> None:
        self.region = _region()
        shapely.prepare(self.region)
        self.points = _probe_points(self.region, np.random.default_rng(7))
        self.expected = shapely.contains_xy(self.region, self.points[:, 0], self.points[:, 1])

    def test_numpy_lookup_equals_shapely(self) -> None:
        for cell_size in (0.25, 0.5, 1.0, 3.0):
            with self.subTest(cell_size=cell_size):
                raster = RegionRaster(self.region, cell_size=cell_size, tile_cells=8)
                np.testing.assert_array_equal(raster.contains(self.points), self.expected)

    def test_tensor_lookup_equals_shapely(self) -> None:
        devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
        raster = RegionRaster(self.region, cell_size=0.5, tile_cells=16)
        for device in devices:
            with self.subTest(device=device):
                mask = raster.contains_tensor(torch.from_numpy(self.points).to(device))
                self.assertEqual(mask.device.type, device)
                np.testing.assert_array_equal(mask.cpu().numpy(), self.expected)

    def test_empty_region_and_empty_input(self) -> None:
        empty = RegionRaster(Polygon(), cell_size=0.5)
        self.assertFalse(empty.contains(self.points).any())
        self.assertFalse(empty.contains_tensor(torch.from_numpy(self.points)).any())
        raster = RegionRaster(self.region, cell_size=0.5)
        self.assertEqual(raster.contains(np.zeros((0, 2))).shape, (0,))
        self.assertEqual(
            raster.contains_tensor(torch.zeros((0, 2), dtype=torch.float64)).shape, (0,)
        )

    def test_points_far_outside_the_raster_are_outside(self) -> None:
        raster = RegionRaster(self.region, cell_size=0.5)
        far = np.array([[-1e6, 0.0], [1e6, 1e6], [30.0, -1e5], [np.nextafter(0.0, -1.0), 4.0]])
        expected = shapely.contains_xy(self.region, far[:, 0], far[:, 1])
        np.testing.assert_array_equal(raster.contains(far), expected)
        np.testing.assert_array_equal(
            raster.contains_tensor(torch.from_numpy(far)).numpy(), expected
        )

    def test_released_device_copies_are_rebuilt(self) -> None:
        raster = RegionRaster(self.region, cell_size=0.5)
        points = torch.from_numpy(self.points)
        first = raster.contains_tensor(points)
        raster.release_device_copies()
        np.testing.assert_array_equal(raster.contains_tensor(points).numpy(), first.numpy())


class LaneletMapRasterContainsTest(unittest.TestCase):
    """LaneletMap.contains keeps its margin semantics with the raster lookup."""

    def setUp(self) -> None:
        # Road y in [-10, 0] with a notch, walkway y in [0, 3], adjacent along y = 0.
        self.lanelet_map = LaneletMap(
            {
                "road": [
                    Polygon([(0, -10), (50, -10), (50, 0), (30, 0), (25, -4), (20, 0), (0, 0)])
                ],
                "walkway": [Polygon([(0, 0), (50, 0), (50, 3), (0, 3)])],
            }
        )
        rng = np.random.default_rng(3)
        self.xy = rng.uniform((-3, -13), (53, 6), size=(30000, 2))

    def _shapely_reference(self, tokens: tuple[str, ...], margin: float) -> np.ndarray:
        """The membership LaneletMap.contains defines, straight from the geometries."""
        x, y = self.xy[:, 0], self.xy[:, 1]
        region = self.lanelet_map.region_union(tokens)
        mask = shapely.contains_xy(region, x, y)
        if margin > 0:
            expanded = shapely.contains_xy(self.lanelet_map._expanded_region(tokens, margin), x, y)
            on_map = shapely.contains_xy(self.lanelet_map._full_surface(), x, y)
            return mask | (expanded & ~on_map)
        if margin < 0:
            return mask & shapely.contains_xy(self.lanelet_map._eroded_full_surface(-margin), x, y)
        return mask

    def test_contains_equals_the_shapely_definition(self) -> None:
        for tokens in (("road",), ("walkway",), ("road", "walkway")):
            for margin in (0.0, -0.2, -1.0, 0.5):
                with self.subTest(tokens=tokens, margin=margin):
                    np.testing.assert_array_equal(
                        self.lanelet_map.contains(tokens, self.xy, margin),
                        self._shapely_reference(tokens, margin),
                    )

    def test_contains_answers_tensors_on_their_device(self) -> None:
        devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
        for device in devices:
            for margin in (0.0, -0.2, 0.5):
                with self.subTest(device=device, margin=margin):
                    mask = self.lanelet_map.contains(
                        ("road",), torch.from_numpy(self.xy).to(device), margin
                    )
                    self.assertIsInstance(mask, torch.Tensor)
                    self.assertEqual(mask.device.type, device)
                    np.testing.assert_array_equal(
                        mask.cpu().numpy(), self._shapely_reference(("road",), margin)
                    )

    def test_region_absent_from_the_map_is_empty(self) -> None:
        self.assertFalse(self.lanelet_map.contains(("crosswalk",), self.xy, -0.2).any())


if __name__ == "__main__":
    unittest.main()
