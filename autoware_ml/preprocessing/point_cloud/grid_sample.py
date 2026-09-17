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

"""Grid subsampling of a batched point cloud."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from jaxtyping import Int64
from torch import Tensor


class GridSamplePreprocessor:
    """Keep one representative point per occupied voxel of every sample of the batch.

    The encoder consumes one point per voxel, and the points a voxel holds keep the label of
    their representative through the inverse index, which scatters the voxel predictions back
    onto the points the metrics are computed on.
    """

    def __init__(self, grid_size: float, point_cloud_range: Sequence[float]) -> None:
        """Initialize the grid subsampling.

        Args:
            grid_size: Edge length of a voxel in meters.
            point_cloud_range: Range [x_min, y_min, z_min, x_max, y_max, z_max] in meters, its
                minimum fixes the origin of the voxel grid.
        """
        self.grid_size = float(grid_size)
        self.point_cloud_range = tuple(float(bound) for bound in point_cloud_range)

    def __call__(self, batch_inputs_dict: dict[str, Any], *, is_training: bool) -> dict[str, Any]:
        """Subsample the points of the batch to one point per voxel.

        Args:
            batch_inputs_dict: Model inputs holding ``coord``, ``feat``, ``offset`` and the
                per point ``batch_indices``, and ``segment`` when segmentation is active.
            is_training: Whether the owning model is in training mode. Training picks a
                random point of every voxel, evaluation and deployment the first one.

        Returns:
            The subsampled model inputs together with ``grid_coord``, the ``inverse`` index
            of every input point, and the ``origin_coord`` and ``origin_segment`` of the
            points before the subsampling.
        """
        coord: Tensor = batch_inputs_dict["coord"]
        batch_indices: Tensor = batch_inputs_dict["batch_indices"]
        grid_coord = self.voxel_coords(coord)
        voxel_keys = self.voxel_keys(grid_coord, batch_indices)

        # Group the points of one voxel together, unique runs over the sorted keys
        sort_indices = torch.argsort(voxel_keys)
        _, sorted_inverse, counts = torch.unique_consecutive(
            voxel_keys[sort_indices], return_inverse=True, return_counts=True
        )
        first_of_voxel = torch.cumsum(counts, dim=0) - counts
        if is_training:
            offset_in_voxel = (torch.rand_like(counts, dtype=torch.float32) * counts).long()
            representatives = sort_indices[first_of_voxel + offset_in_voxel]
        else:
            representatives = sort_indices[first_of_voxel]

        inverse = torch.empty_like(sorted_inverse)
        inverse[sort_indices] = sorted_inverse

        outputs: dict[str, Any] = {
            "origin_coord": coord,
            "inverse": inverse,
            "coord": coord[representatives],
            "feat": batch_inputs_dict["feat"][representatives],
            "grid_coord": grid_coord[representatives].to(torch.int32),
            "batch_indices": batch_indices[representatives],
            "offset": self.voxel_offset(batch_indices[representatives], batch_inputs_dict),
        }
        if "segment" in batch_inputs_dict:
            outputs["origin_segment"] = batch_inputs_dict["segment"]
            outputs["segment"] = batch_inputs_dict["segment"][representatives]
        return outputs

    def voxel_coords(self, coord: Tensor) -> Int64[Tensor, "num_points 3"]:
        """Discretize the point coordinates into voxel coordinates.

        Args:
            coord: Coordinates of every point.

        Returns:
            Int64[Tensor, "num_points 3"]: Voxel coordinate of every point, counted from the
                minimum of the configured range.
        """
        minimum = torch.tensor(self.point_cloud_range[:3], dtype=torch.float32, device=coord.device)
        # A host scalar divisor lets CUDA multiply by its rounded reciprocal, which pushes
        # points just below the maximum one voxel past the grid. A device tensor divides exactly.
        grid_size = torch.tensor(self.grid_size, dtype=torch.float32, device=coord.device)
        origin = torch.floor(minimum / grid_size)
        return (torch.floor(coord / grid_size) - origin).long()

    @staticmethod
    def voxel_keys(
        grid_coord: Int64[Tensor, "num_points 3"], batch_indices: Tensor
    ) -> Int64[Tensor, " num_points"]:
        """Give every voxel of every sample of the batch one key.

        Args:
            grid_coord: Voxel coordinate of every point.
            batch_indices: Sample every point belongs to.

        Returns:
            Int64[Tensor, " num_points"]: Key of the voxel every point falls into.
        """
        keyed = torch.cat([batch_indices.long().unsqueeze(1), grid_coord], dim=1)
        extent = keyed.max(dim=0).values - keyed.min(dim=0).values + 1
        keys = torch.zeros(keyed.shape[0], dtype=torch.int64, device=keyed.device)
        shifted = keyed - keyed.min(dim=0).values
        for dimension in range(keyed.shape[1]):
            keys = keys * extent[dimension] + shifted[:, dimension]
        return keys

    @staticmethod
    def voxel_offset(batch_indices: Tensor, batch_inputs_dict: dict[str, Any]) -> Tensor:
        """Split the retained voxels per sample the way the encoder reads them.

        Args:
            batch_indices: Sample every retained voxel belongs to.
            batch_inputs_dict: Model inputs holding the sample count of the batch.

        Returns:
            Tensor: Cumulative voxel count of every sample.
        """
        counts = torch.bincount(batch_indices.long(), minlength=batch_inputs_dict["sample_count"])
        return torch.cumsum(counts, dim=0)
