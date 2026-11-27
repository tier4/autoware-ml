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
        """Test that the NumPy lookup equals shapely at every cell size.

        Input: the multipolygon from ``_region`` (a hole, concave stars, a 3 cm sliver and a
        far-away part) and the points from ``_probe_points`` (uniform, scattered about 1 mm
        around every vertex and 0.1 mm around every edge, and a 0.5 m lattice).
        Expected: ``contains`` equals shapely.contains point for point.
        Check: cell sizes 0.25, 0.5, 1 and 3 m with 8-cell tiles.
        """
        for cell_size in (0.25, 0.5, 1.0, 3.0):
            with self.subTest(cell_size=cell_size):
                raster = RegionRaster(self.region, cell_size=cell_size, tile_cells=8)
                np.testing.assert_array_equal(raster.contains(self.points), self.expected)

    def test_tensor_lookup_equals_shapely(self) -> None:
        """Test that the tensor lookup equals shapely and answers on the input's device.

        Input: the same region and points as float64 tensors on the CPU and, when available, on
        CUDA, with 0.5 m cells and 16-cell tiles.
        Expected: ``contains_tensor`` returns a mask on the input's device equal to
        shapely.contains.
        Check: the mask's device type and values on every available device.
        """
        devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
        raster = RegionRaster(self.region, cell_size=0.5, tile_cells=16)
        for device in devices:
            with self.subTest(device=device):
                mask = raster.contains_tensor(torch.from_numpy(self.points).to(device))
                self.assertEqual(mask.device.type, device)
                np.testing.assert_array_equal(mask.cpu().numpy(), self.expected)

    def test_empty_region_and_empty_input(self) -> None:
        """Test that an empty region contains nothing and no points give an empty mask.

        Input: a raster of an empty polygon probed with the usual points, and a raster of the
        test region probed with zero points.
        Expected: the empty region answers False everywhere, and zero points give shape (0,).
        Check: the NumPy and tensor lookups in both cases.
        """
        empty = RegionRaster(Polygon(), cell_size=0.5)
        self.assertFalse(empty.contains(self.points).any())
        self.assertFalse(empty.contains_tensor(torch.from_numpy(self.points)).any())
        raster = RegionRaster(self.region, cell_size=0.5)
        self.assertEqual(raster.contains(np.zeros((0, 2))).shape, (0,))
        self.assertEqual(
            raster.contains_tensor(torch.zeros((0, 2), dtype=torch.float64)).shape, (0,)
        )

    def test_points_far_outside_the_raster_are_outside(self) -> None:
        """Test that points far outside the raster, and one just outside the region, are outside.

        Input: points 1000 km and 100 km away from the region, and a point at the next float
        below x = 0, just left of the region's edge there.
        Expected: the answers equal shapely.contains, which puts all four outside.
        Check: the NumPy and tensor masks.
        """
        raster = RegionRaster(self.region, cell_size=0.5)
        far = np.array([[-1e6, 0.0], [1e6, 1e6], [30.0, -1e5], [np.nextafter(0.0, -1.0), 4.0]])
        expected = shapely.contains_xy(self.region, far[:, 0], far[:, 1])
        np.testing.assert_array_equal(raster.contains(far), expected)
        np.testing.assert_array_equal(
            raster.contains_tensor(torch.from_numpy(far)).numpy(), expected
        )

    def test_cells_whose_centre_lies_on_an_edge_are_exact(self) -> None:
        """Test that cells whose centre lies on an edge give shapely's answer in both orientations.

        Input: an L-shaped region whose inner vertical edge, x = 1.25, runs through the
        centres of a column of 0.5 m cells (the raster starts one cell left of x = 0),
        with its ring counter-clockwise and clockwise, probed on a fine grid, on that
        edge and within 1e-9 m of it.
        Expected: those cells cannot use their centre as the parity reference, so their
        points are answered by shapely, and every answer equals shapely.contains in both
        orientations. Without the fallback the clockwise ring gets 400 grid points wrong.
        Check: at least one such cell, and the NumPy and tensor masks equal shapely's.
        """
        corners = [(0, 0), (4, 0), (4, 4), (1.25, 4), (1.25, 2), (0, 2)]
        grid = np.stack(np.meshgrid(np.arange(-1, 5, 0.05), np.arange(-1, 5, 0.05)), -1)
        edge_y = np.linspace(1.9, 4.1, 221)
        points = np.concatenate(
            [
                grid.reshape(-1, 2),
                np.stack([np.full_like(edge_y, 1.25), edge_y], 1),
                np.stack([np.full_like(edge_y, 1.25 - 1e-9), edge_y], 1),
                np.stack([np.full_like(edge_y, 1.25 + 1e-9), edge_y], 1),
            ]
        )
        for orientation, ring in (("counter-clockwise", corners), ("clockwise", corners[::-1])):
            with self.subTest(orientation=orientation):
                region = Polygon(ring)
                raster = RegionRaster(region, cell_size=0.5)
                expected = shapely.contains_xy(region, points[:, 0], points[:, 1])

                self.assertGreater(int(raster.exact.sum()), 0)
                np.testing.assert_array_equal(raster.contains(points), expected)
                np.testing.assert_array_equal(
                    raster.contains_tensor(torch.from_numpy(points)).numpy(), expected
                )

    def test_released_device_copies_are_rebuilt(self) -> None:
        """Test that the tensor lookup rebuilds its tables after they are released.

        Input: the test region at 0.5 m and the points as a CPU tensor, looked up once, then
        again after ``release_device_copies``.
        Expected: the second lookup rebuilds the tables and gives the same mask as the first.
        Check: the two masks are equal.
        """
        raster = RegionRaster(self.region, cell_size=0.5)
        points = torch.from_numpy(self.points)
        first = raster.contains_tensor(points)
        raster.release_device_copies()
        np.testing.assert_array_equal(raster.contains_tensor(points).numpy(), first.numpy())


class RegionRasterGridEdgeTest(unittest.TestCase):
    """Edges in the first and last cells of the raster."""

    def _assert_equals_shapely(self, region, raster: RegionRaster, points: np.ndarray) -> None:
        expected = shapely.contains_xy(region, points[:, 0], points[:, 1])
        np.testing.assert_array_equal(raster.contains(points), expected)
        np.testing.assert_array_equal(
            raster.contains_tensor(torch.from_numpy(points)).numpy(), expected
        )

    def test_edge_in_the_last_cell_builds_and_equals_shapely(self) -> None:
        """Test that a region whose far edge lies in the raster's last cell builds and is exact.

        Input: 14 x 3 m and 3 x 14 m boxes at the origin with 1 m cells. The raster starts one
        cell before the region and is a whole number of 16-cell tiles wide, so the far side at
        14 m falls in the last column (or row), whose right-hand neighbour lies past the grid.
        Expected: the raster builds, and its answers equal shapely.contains.
        Check: a 0.1 m grid of points over the boxes and their surroundings, NumPy and tensor.
        """
        steps = np.arange(-2.0, 17.0, 0.1)
        points = np.stack(np.meshgrid(steps, steps), -1).reshape(-1, 2)
        for width, height in ((14.0, 3.0), (3.0, 14.0)):
            with self.subTest(width=width, height=height):
                region = shapely.box(0.0, 0.0, width, height)
                raster = RegionRaster(region, cell_size=1.0)
                self._assert_equals_shapely(region, raster, points)

    def test_edge_in_the_first_cell_equals_shapely(self) -> None:
        """Test that a region whose near edge rounds into the raster's first cell is exact.

        Input: a hexagon around (300 km, 300 km) with 0.1 m cells. The raster starts one cell
        left of the region, but at this magnitude ``min_x - 0.1`` rounds, so the leftmost
        vertex lands in the first column and its left-hand neighbour lies before the grid.
        Expected: answers equal shapely.contains, also in the far right column of tiles that
        a neighbour before the grid would index from the end.
        Check: a 5 cm grid of points along the right side of the hexagon, NumPy and tensor.
        """
        region = Polygon(
            [
                (299975.230, 299983.076),
                (300002.271, 299970.086),
                (300027.042, 299987.010),
                (300024.770, 300016.924),
                (299997.729, 300029.914),
                (299972.958, 300012.990),
            ]
        )
        raster = RegionRaster(region, cell_size=0.1)
        min_x, min_y, max_x, max_y = region.bounds
        xs = np.arange(max_x - 3.0, max_x + 0.5, 0.05)
        ys = np.arange(min_y - 0.5, max_y + 0.5, 0.05)
        points = np.stack(np.meshgrid(xs, ys), -1).reshape(-1, 2)
        self._assert_equals_shapely(region, raster, points)


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
        """Test that LaneletMap.contains equals its shapely definition for every margin sign.

        Input: a road with a notch and a walkway sharing the edge y = 0, 30,000 random points
        around them, the tokens road, walkway and both, and margins 0, -0.2, -1 and 0.5 m.
        Expected: the mask equals ``_shapely_reference``: the region alone at margin 0, the
        region inside the map surface eroded by the margin when it is negative, and the region
        plus its expansion beyond the map surface when it is positive.
        Check: every token set and margin, NumPy input.
        """
        for tokens in (("road",), ("walkway",), ("road", "walkway")):
            for margin in (0.0, -0.2, -1.0, 0.5):
                with self.subTest(tokens=tokens, margin=margin):
                    np.testing.assert_array_equal(
                        self.lanelet_map.contains(tokens, self.xy, margin),
                        self._shapely_reference(tokens, margin),
                    )

    def test_contains_answers_tensors_on_their_device(self) -> None:
        """Test that LaneletMap.contains answers a tensor with a tensor on the same device.

        Input: the road token with margins 0, -0.2 and 0.5 m, and the points as float64 tensors
        on the CPU and, when available, on CUDA.
        Expected: a torch.Tensor on the input's device, equal to the shapely definition.
        Check: the mask's type, device and values.
        """
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
        """Test that a region type the map has no polygon for contains no point.

        Input: the crosswalk token, absent from this map, with a -0.2 m margin.
        Expected: every point is outside.
        Check: the mask has no True value.
        """
        self.assertFalse(self.lanelet_map.contains(("crosswalk",), self.xy, -0.2).any())


if __name__ == "__main__":
    unittest.main()
