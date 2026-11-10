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

"""Synthetic model inputs at set voxel counts and densities.

The voxel layout pools stage by stage as a hierarchical encoder does, and the raw files it
writes are the inputs of a trtexec latency benchmark.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import functools
import itertools
import logging
import math
from pathlib import Path
from typing import Mapping, Sequence

from jaxtyping import Float32, Int, Int64
import torch

from autoware_ml.utils.deploy import ExportSpec
from autoware_ml.utils.point_cloud.structures import pooled_sparse_shape

logger = logging.getLogger(__name__)

_NEIGHBOR_OFFSETS = torch.tensor([d for d in itertools.product((-1, 0, 1), repeat=3) if any(d)])

Cells = Int64[torch.Tensor, "num_cells 3"]


@dataclass(frozen=True)
class VoxelLayoutSpec:
    """Describe a pooled voxel layout, one entry per encoder stage.

    Attributes:
        voxel_counts: Voxels per stage, finest first.
        pooling_strides: Pooling stride between consecutive stages.
        grid_shape: Finest grid size per axis.
        target_neighbors: Target mean of occupied cells around a voxel per stage, finest first.
        center: Point the voxels cluster around, in finest grid cells.
    """

    voxel_counts: Sequence[int]
    pooling_strides: Sequence[int]
    grid_shape: Int64[torch.Tensor, "3"]
    target_neighbors: Sequence[float]
    center: Float32[torch.Tensor, "3"]

    def __post_init__(self) -> None:
        """Refuse a spec whose stages do not line up.

        Raises:
            ValueError: If the counts, strides and targets disagree on the number of stages.
        """
        num_stages = len(self.voxel_counts)
        if not num_stages == len(self.pooling_strides) + 1 == len(self.target_neighbors):
            raise ValueError(
                f"{num_stages} voxel counts and {len(self.target_neighbors)} neighbor "
                f"targets for {len(self.pooling_strides)} pooling stages"
            )


def mean_neighbors(coords: Int[torch.Tensor, "num_voxels 3"]) -> float:
    """Return how many of the 26 cells around a voxel are occupied, on average.

    Args:
        coords: ``[num_voxels, 3]`` integer grid coordinates of the occupied voxels.

    Returns:
        Mean number of occupied neighbor cells per voxel, between 0 and 26.
    """
    cells = coords.long() + 1
    neighbor_cells = cells[:, None] + _NEIGHBOR_OFFSETS.to(cells.device)
    span = int(neighbor_cells.max()) + 1
    weights = torch.tensor([span * span, span, 1], device=cells.device)
    keys = (cells * weights).sum(dim=-1).sort().values
    neighbor_keys = (neighbor_cells * weights).sum(dim=-1)
    found = keys[torch.searchsorted(keys, neighbor_keys).clamp(max=len(keys) - 1)] == neighbor_keys
    return float(found.sum()) / len(coords)


def _best_cells(cells: Cells, count: int, score: Float32[torch.Tensor, " num_cells"]) -> Cells:
    """Return the ``count`` best-scoring cells."""
    return cells[score.argsort()[:count]]


def _best_children(
    children: Cells, num_parents: int, count: int, score: Float32[torch.Tensor, " num_cells"]
) -> Cells:
    """Return the ``count`` best-scoring children, the best child of every parent among them."""
    score = score.reshape(num_parents, -1)
    keep = torch.zeros_like(score, dtype=torch.bool)
    keep[torch.arange(num_parents), score.argmin(dim=1)] = True
    rest = torch.where(keep, math.inf, score).reshape(-1)
    keep.view(-1)[rest.argsort()[: count - num_parents]] = True
    return children[keep.view(-1)]


def _fit_neighbors(
    cells: Cells,
    grid_shape: Int64[torch.Tensor, "3"],
    center: Float32[torch.Tensor, "3"],
    depth: int,
    target_neighbors: float,
    generator: torch.Generator,
    keep: Callable[[Float32[torch.Tensor, " num_cells"]], Cells],
) -> Cells:
    """Return the cells ``keep`` picks by distance to ``center`` plus noise.

    The noise weight is bisected until the picked cells have ``target_neighbors`` on average.
    """
    distance = (cells + 0.5 - center / (1 << depth)).norm(dim=1)
    noise = torch.rand(len(cells), generator=generator) * distance.max()
    distance[~(cells < pooled_sparse_shape(grid_shape, depth)).all(dim=1)] = math.inf
    low_weight, high_weight, tolerance = 0.0, 64.0, 1e-3
    while high_weight - low_weight > tolerance:
        noise_weight = (low_weight + high_weight) / 2
        if mean_neighbors(keep(distance + noise_weight * noise)) > target_neighbors:
            low_weight = noise_weight
        else:
            high_weight = noise_weight
    return keep(distance + low_weight * noise)


def voxel_layout(
    spec: VoxelLayoutSpec, generator: torch.Generator
) -> Int64[torch.Tensor, "num_voxels 3"]:
    """Pick voxels whose stages hold the voxel counts and neighbor targets of ``spec``.

    Stage by stage, coarsest first, keeps the cells nearest ``spec.center``, ranked with noise
    whose weight is bisected until the stage's mean neighbors reach the target. Every
    stage's grid is the finest one pooled by ``2 ** depth``, rounded up as the model pools it.

    Args:
        spec: Voxel counts, pooling strides, grid, neighbor targets and center.
        generator: Source of the ranking noise.

    Returns:
        ``[spec.voxel_counts[0], 3]`` grid coordinates.

    Raises:
        ValueError: If a stage cannot pool to the next.
    """
    voxel_counts, pooling_strides = spec.voxel_counts, spec.pooling_strides
    grid_shape, target_neighbors = spec.grid_shape, spec.target_neighbors
    depth = sum(stride.bit_length() - 1 for stride in pooling_strides)
    cells = torch.cartesian_prod(
        *(torch.arange(int(n)) for n in pooled_sparse_shape(grid_shape, depth))
    )
    if voxel_counts[-1] > len(cells):
        raise ValueError(f"{voxel_counts[-1]} voxels cannot pool to a grid of {len(cells)} cells")
    layout = _fit_neighbors(
        cells,
        grid_shape,
        spec.center,
        depth,
        target_neighbors[-1],
        generator,
        functools.partial(_best_cells, cells, voxel_counts[-1]),
    )
    for stage in reversed(range(len(voxel_counts) - 1)):
        stride = pooling_strides[stage]
        depth -= stride.bit_length() - 1
        offsets = torch.tensor(list(itertools.product(range(stride), repeat=3)))
        cells = (layout[:, None] * stride + offsets[None]).reshape(-1, 3)
        inside = int((cells < pooled_sparse_shape(grid_shape, depth)).all(dim=1).sum())
        if not len(layout) <= voxel_counts[stage] <= inside:
            raise ValueError(
                f"{voxel_counts[stage]} voxels cannot pool to {len(layout)} with stride {stride}"
            )
        layout = _fit_neighbors(
            cells,
            grid_shape,
            spec.center,
            depth,
            target_neighbors[stage],
            generator,
            functools.partial(_best_children, cells, len(layout), voxel_counts[stage]),
        )
    return layout


def write_export_inputs(specs: Mapping[str, ExportSpec], out: Path) -> None:
    """Write every input of every spec as a raw ``.bin`` under ``out/<module>``.

    ``shapes.txt`` next to them lists ``name:AxB`` per input, the ``--shapes`` argument of
    trtexec.

    Args:
        specs: Export spec per module; every tensor of ``args`` is written as
            ``<input name>.bin``.
        out: Root directory, one subdirectory per module, created when missing. Files
            already there are overwritten.
    """
    for module, spec in specs.items():
        module_dir = out / module
        module_dir.mkdir(parents=True, exist_ok=True)
        for name, tensor in zip(spec.input_names, spec.args):
            tensor.detach().cpu().contiguous().numpy().tofile(module_dir / f"{name}.bin")
        shapes = ",".join(
            f"{name}:{'x'.join(map(str, tensor.shape))}"
            for name, tensor in zip(spec.input_names, spec.args)
        )
        (module_dir / "shapes.txt").write_text(shapes + "\n")
        logger.info("%s: %d inputs in %s", module, len(spec.input_names), module_dir)
