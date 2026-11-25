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

"""Exact point-in-region membership through a tiled raster of the region.

A segmentation frame asks, for a few hundred thousand points, whether each lies in a
lanelet region. Testing every point against the shapely polygon costs a quarter of a
second per frame and filter. The region is fixed per map, so this module turns it into
a raster once and answers each point with array lookups, on the device the points live
on, with the same result as ``shapely.contains``:

* A cell that no polygon edge touches lies wholly inside or wholly outside the region,
  so shapely's verdict on its centre, taken at build time, holds for all of it.
* A cell that an edge touches (a conservative superset of them) keeps the list of the
  edges that touch it, and its centre as a reference point whose membership shapely
  decides at build time. A point in that cell is inside exactly when the reference is
  inside XOR the segment from the point to the reference crosses an odd number of the
  cell's edges. The segment never leaves the convex cell, so no other edge can cross it.
  The rare cell whose centre lies on one of its edges leaves its points to shapely.

Cells are stored in square tiles. A tile without any edge-touched cell is uniform and
stored as one value, so a map several kilometres across costs memory only along the
region borders.
"""

from __future__ import annotations

from collections import OrderedDict

import numpy as np
import shapely
import torch

# Cell codes. A non-negative code is the index of an edge-touched ("border") cell.
OUTSIDE = -1
INSIDE = -2

# Rasters whose tables currently sit on a device, most recently used last. A test run
# visits scenes one after another, so only the last few maps' tables stay resident and
# device memory does not grow with the number of maps.
MAX_DEVICE_RESIDENT_RASTERS = 16
_device_resident: OrderedDict[int, "RegionRaster"] = OrderedDict()

# Relative error bound of the float orientation determinant, ten times the classic
# (3 + 16 eps) eps bound of the robust orientation predicate, below which the sign is
# not trusted.
_ORIENTATION_TOLERANCE = 1e-14

# Distance, as a fraction of the cell side, below which a cell centre counts as lying on
# an edge and cannot serve as the cell's reference point.
_REFERENCE_CLEARANCE = 1e-6

# The tables a lookup reads. NumPy lookups use them as built, tensor lookups use copies
# on the device the points arrive on.
_TABLES = (
    "tile_codes",
    "blocks",
    "edges",
    "edge_offsets",
    "edge_list",
    "reference",
    "reference_inside",
    "exact",
)


class RegionRaster:
    """Point membership in the interior of one shapely (multi)polygon.

    Args:
        geometry: The region, a (multi)polygon in the map frame. May be empty.
        cell_size: Side of a raster cell in metres. Smaller cells send fewer points
            through the exact edge test and cost more memory and build time.
        tile_cells: Side of a tile in cells.
    """

    def __init__(self, geometry, cell_size: float = 0.5, tile_cells: int = 16) -> None:
        self.cell_size = float(cell_size)
        self.tile_cells = int(tile_cells)
        self.empty = bool(geometry.is_empty)
        self._device_tensors: dict[str, dict[str, torch.Tensor]] = {}
        if self.empty:
            return
        self._geometry = geometry
        shapely.prepare(geometry)
        cell, tile = self.cell_size, self.tile_cells
        min_x, min_y, max_x, max_y = geometry.bounds
        self.origin = (min_x - cell, min_y - cell)
        self.tiles_x = int(np.ceil((max_x + cell - self.origin[0]) / (cell * tile)))
        self.tiles_y = int(np.ceil((max_y + cell - self.origin[1]) / (cell * tile)))
        self.cells_x = self.tiles_x * tile
        self.cells_y = self.tiles_y * tile

        self.edges = _ring_edges(geometry)
        border_cells, self.edge_offsets, self.edge_list = self._touched_cells()
        self._build_tiles(border_cells)
        self._build_references(border_cells)
        self._host_tables = {name: getattr(self, name) for name in _TABLES}

    # ------------------------------------------------------------------ build
    def _touched_cells(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Flat ids of the cells the edges touch, with each cell's edges in CSR form.

        Every edge is sampled at most half a cell apart. A segment that crosses a cell
        has a sample within half a cell of a point in it, so that cell lies in the 3x3
        neighbourhood of the sample's cell: the union of the neighbourhoods is a
        superset of the touched cells, which only costs exact tests.
        """
        cell = self.cell_size
        starts, ends = self.edges[:, :2], self.edges[:, 2:]
        length = np.hypot(*(ends - starts).T)
        steps = np.maximum(np.ceil(length / (0.5 * cell)).astype(np.int64), 1)
        per_edge = steps + 1
        edge_of_sample = np.repeat(np.arange(len(steps)), per_edge)
        fraction = (
            np.arange(edge_of_sample.size) - np.repeat(np.cumsum(per_edge) - per_edge, per_edge)
        ) / np.repeat(steps, per_edge)
        samples = starts[edge_of_sample] + fraction[:, None] * (ends - starts)[edge_of_sample]
        column = np.floor((samples[:, 0] - self.origin[0]) / cell).astype(np.int64)
        row = np.floor((samples[:, 1] - self.origin[1]) / cell).astype(np.int64)
        keys = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                # Points outside the grid never reach the tables (they are OUTSIDE), so a
                # neighbour outside it is not needed, and its flat id would alias another
                # cell or tile (a negative column indexes the tiles from the end) or run past
                # the tile table. The near edge can round into the first cell when
                # ``min - cell`` is inexact, and the far edge can lie in the last cell.
                neighbour_column, neighbour_row = column + dx, row + dy
                in_grid = (
                    (neighbour_column >= 0)
                    & (neighbour_column < self.cells_x)
                    & (neighbour_row >= 0)
                    & (neighbour_row < self.cells_y)
                )
                cell_id = neighbour_column * self.cells_y + neighbour_row
                keys.append((cell_id * len(self.edges) + edge_of_sample)[in_grid])
        pairs = np.unique(np.concatenate(keys))  # sorted by cell, then edge
        pair_cell, pair_edge = np.divmod(pairs, len(self.edges))
        border_cells, first = np.unique(pair_cell, return_index=True)
        offsets = np.append(first, len(pairs)).astype(np.int64)
        return border_cells, offsets, pair_edge.astype(np.int64)

    def _build_tiles(self, border_cells: np.ndarray) -> None:
        """Tile codes (uniform value or block index) and the cell blocks of mixed tiles.

        No edge touches a uniform tile or a cell that is not a border cell, so each lies
        wholly on one side of the boundary and shapely's verdict on its centre decides
        all of it.
        """
        tile = self.tile_cells
        column, row = np.divmod(border_cells, self.cells_y)
        tile_of_border = (column // tile) * self.tiles_y + (row // tile)
        mixed_tiles = np.unique(tile_of_border)

        tile_x, tile_y = np.divmod(np.arange(self.tiles_x * self.tiles_y), self.tiles_y)
        tile_codes = self._side_of_centres(tile_x, tile_y, tile)
        tile_codes[mixed_tiles] = np.arange(len(mixed_tiles), dtype=np.int32)

        local_x, local_y = np.divmod(np.arange(tile * tile), tile)
        blocks = self._side_of_centres(
            (mixed_tiles // self.tiles_y)[:, None] * tile + local_x,
            (mixed_tiles % self.tiles_y)[:, None] * tile + local_y,
            1,
        )
        block_of_border = np.searchsorted(mixed_tiles, tile_of_border)
        local = (column % tile) * tile + (row % tile)
        blocks[block_of_border, local] = np.arange(len(border_cells), dtype=np.int32)
        self.tile_codes = tile_codes
        self.blocks = blocks.ravel()

    def _side_of_centres(self, x: np.ndarray, y: np.ndarray, span: int) -> np.ndarray:
        """``INSIDE`` or ``OUTSIDE`` for the centre of each ``span``-cell square at (x, y)."""
        side = self.cell_size * span
        inside = shapely.contains_xy(
            self._geometry, self.origin[0] + (x + 0.5) * side, self.origin[1] + (y + 0.5) * side
        )
        return np.where(inside, INSIDE, OUTSIDE).astype(np.int32)

    def _build_references(self, border_cells: np.ndarray) -> None:
        """Each border cell's centre as reference point, with its membership.

        A centre on or within rounding of one of the cell's edges is no use as a
        reference, so such a cell is marked ``exact`` and its points go to shapely.
        """
        column, row = np.divmod(border_cells, self.cells_y)
        reference = np.stack(
            [
                self.origin[0] + (column + 0.5) * self.cell_size,
                self.origin[1] + (row + 0.5) * self.cell_size,
            ],
            axis=1,
        )
        owner = np.repeat(np.arange(len(border_cells)), np.diff(self.edge_offsets))
        nearest = np.full(len(border_cells), np.inf)
        np.minimum.at(
            nearest, owner, _segment_distance(reference[owner], self.edges[self.edge_list])
        )
        self.reference = reference
        self.reference_inside = shapely.contains_xy(
            self._geometry, reference[:, 0], reference[:, 1]
        )
        self.exact = nearest <= _REFERENCE_CLEARANCE * self.cell_size

    # ----------------------------------------------------------------- lookup
    def contains(self, xy: np.ndarray) -> np.ndarray:
        """Whether each map-frame point lies in the region's interior.

        Args:
            xy: Points ``(N, 2)``.

        Returns:
            Boolean mask ``(N,)``, equal to ``shapely.contains`` of the region.
        """
        xy = np.asarray(xy, dtype=np.float64)
        if self.empty or xy.shape[0] == 0:
            return np.zeros(xy.shape[0], dtype=bool)
        tables = self._host_tables
        code = self._codes(xy, tables)
        mask = code == INSIDE
        points = np.flatnonzero(code >= 0)
        if points.size:
            border = code[points].astype(np.int64)
            owner, edges = _cell_edges(border, tables)
            crossings, unsure = _crosses(xy[points[owner]], self.reference[border[owner]], edges)
            odd = np.bincount(owner, weights=crossings, minlength=points.size).astype(np.int64)
            mask[points] = self.reference_inside[border] ^ (odd % 2 == 1)
            unsure_count = np.bincount(owner, weights=unsure, minlength=points.size)
            exact_points = points[(unsure_count > 0) | self.exact[border]]
            if exact_points.size:
                mask[exact_points] = self._shapely_contains(xy[exact_points])
        return mask

    def contains_tensor(self, xy: torch.Tensor) -> torch.Tensor:
        """:meth:`contains` on the points' own device.

        Args:
            xy: Points ``(N, 2)``.

        Returns:
            Boolean mask ``(N,)`` on the device of ``xy``, equal to ``shapely.contains``.
        """
        if self.empty or xy.shape[0] == 0:
            return torch.zeros(xy.shape[0], dtype=torch.bool, device=xy.device)
        xy = xy.to(torch.float64)
        tables = self._tensors(xy.device)
        code = self._codes(xy, tables)
        mask = code == INSIDE
        points = torch.nonzero(code >= 0).squeeze(1)
        if points.numel():
            border = code[points].long()
            owner, edges = _cell_edges(border, tables)
            crossings, unsure = _crosses(
                xy[points[owner]], tables["reference"][border[owner]], edges
            )
            odd = torch.zeros_like(border).index_add_(0, owner, crossings.long())
            mask[points] = tables["reference_inside"][border] ^ (odd % 2 == 1)
            unsure_count = torch.zeros_like(border).index_add_(0, owner, unsure.long())
            exact_points = points[(unsure_count > 0) | tables["exact"][border]]
            if exact_points.numel():
                exact = self._shapely_contains(xy[exact_points].cpu().numpy())
                mask[exact_points] = torch.from_numpy(exact).to(xy.device)
        return mask

    def _shapely_contains(self, xy: np.ndarray) -> np.ndarray:
        """The exact test, for the rare points the float edge test cannot decide."""
        return np.asarray(shapely.contains_xy(self._geometry, xy[:, 0], xy[:, 1]), dtype=bool)

    def release_device_copies(self) -> None:
        """Drop the tables copied to devices, they are rebuilt on the next tensor lookup."""
        self._device_tensors.clear()
        _device_resident.pop(id(self), None)

    def _codes(self, xy, tables):
        """Cell code of each point: ``INSIDE``, ``OUTSIDE`` or a border cell index.

        Works on NumPy arrays with NumPy tables and on tensors with tensor tables.
        """
        xp = np if isinstance(xy, np.ndarray) else torch
        tile = self.tile_cells
        column = _floor_index((xy[:, 0] - self.origin[0]) / self.cell_size)
        row = _floor_index((xy[:, 1] - self.origin[1]) / self.cell_size)
        valid = (column >= 0) & (column < self.cells_x) & (row >= 0) & (row < self.cells_y)
        column = xp.where(valid, column, 0)
        row = xp.where(valid, row, 0)
        tile_code = tables["tile_codes"][(column // tile) * self.tiles_y + row // tile]
        block = xp.where(tile_code >= 0, tile_code, 0)
        cell_code = tables["blocks"][block * (tile * tile) + (column % tile) * tile + row % tile]
        code = xp.where(tile_code >= 0, cell_code, tile_code)
        return xp.where(valid, code, OUTSIDE)

    def _tensors(self, device: torch.device) -> dict[str, torch.Tensor]:
        """The lookup tables on ``device``, copied on first use (shared with NumPy on the CPU)."""
        _device_resident[id(self)] = self
        _device_resident.move_to_end(id(self))
        while len(_device_resident) > MAX_DEVICE_RESIDENT_RASTERS:
            _device_resident.popitem(last=False)[1].release_device_copies()
        key = str(device)
        if key not in self._device_tensors:
            self._device_tensors[key] = {
                name: torch.from_numpy(np.ascontiguousarray(getattr(self, name))).to(device)
                for name in _TABLES
            }
        return self._device_tensors[key]


def _ring_edges(geometry) -> np.ndarray:
    """Every ring segment of a (multi)polygon, rows ``x1 y1 x2 y2``, float64."""
    rings = shapely.get_rings(shapely.get_parts(geometry))
    coords, ring = shapely.get_coordinates(rings, return_index=True)
    same_ring = ring[1:] == ring[:-1]
    return np.concatenate([coords[:-1][same_ring], coords[1:][same_ring]], axis=1)


def _floor_index(values):
    """``floor`` of an array or tensor of coordinates, as int64 indices."""
    if isinstance(values, np.ndarray):
        return np.floor(values).astype(np.int64)
    return torch.floor(values).long()


def _cell_edges(border, tables):
    """Pairs (index into ``border``, edge row) of every edge of the given border cells.

    Works on a NumPy ``border`` with NumPy tables and on a tensor with tensor tables.
    """
    start = tables["edge_offsets"][border]
    count = tables["edge_offsets"][border + 1] - start
    if isinstance(border, np.ndarray):
        owner = np.repeat(np.arange(border.size), count)
        rank = np.arange(owner.size) - np.repeat(np.cumsum(count) - count, count)
    else:
        owner = torch.repeat_interleave(torch.arange(border.numel(), device=border.device), count)
        rank = torch.arange(owner.numel(), device=border.device) - torch.repeat_interleave(
            torch.cumsum(count, 0) - count, count
        )
    return owner, tables["edges"][tables["edge_list"][start[owner] + rank]]


def _segment_distance(points: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Pairwise distance from ``points`` (N, 2) to segments ``edges`` (N, 4)."""
    start, delta = edges[:, :2], edges[:, 2:] - edges[:, :2]
    length2 = np.einsum("ij,ij->i", delta, delta)
    t = np.clip(
        np.einsum("ij,ij->i", points - start, delta) / np.where(length2 > 0, length2, 1.0), 0, 1
    )
    return np.hypot(*(points - (start + t[:, None] * delta)).T)


def _crosses(points, references, edges):
    """Whether the segment point -> reference crosses each edge, pairwise, and whether
    that answer is unsure.

    The sides are half-open (zero counts as negative), so a segment through a vertex
    shared by two edges counts one crossing when the ring passes through and none
    when it only touches: both edges compute that vertex's side with the very same
    arithmetic, so they agree on it. The one side that float rounding can get wrong in
    a way that matters is the point's own side of an edge line, when the point lies
    on or within rounding of the line (a point on the boundary is outside for
    ``shapely.contains``). Such pairs are flagged unsure, with a generous error bound
    of the orientation determinant, for the caller to decide exactly.
    """
    ax, ay, bx, by = edges[:, 0], edges[:, 1], edges[:, 2], edges[:, 3]
    px, py = points[:, 0], points[:, 1]
    rx, ry = references[:, 0], references[:, 1]
    point_left = (bx - ax) * (py - ay)
    point_right = (by - ay) * (px - ax)
    point_side = point_left - point_right
    reference_side = (bx - ax) * (ry - ay) - (by - ay) * (rx - ax)
    start_side = (rx - px) * (ay - py) - (ry - py) * (ax - px)
    end_side = (rx - px) * (by - py) - (ry - py) * (bx - px)
    crosses = ((point_side > 0) != (reference_side > 0)) & ((start_side > 0) != (end_side > 0))
    unsure = abs(point_side) <= _ORIENTATION_TOLERANCE * (abs(point_left) + abs(point_right))
    return crosses, unsure
