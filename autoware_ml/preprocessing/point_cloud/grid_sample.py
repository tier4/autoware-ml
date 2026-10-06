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

import torch
from jaxtyping import Int64
from torch import Tensor

from autoware_ml.dataclasses.geometry.grid_sample import GridSampleData
from autoware_ml.dataclasses.models.model_batch_inputs import ModelBatchInputs


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

    def __call__(self, batch_inputs: ModelBatchInputs, *, is_training: bool) -> ModelBatchInputs:
        """Subsample the points of the batch to one point per voxel.

        Args:
            batch_inputs: Model inputs holding the point cloud of the batch.
            is_training: Whether the owning model is in training mode. Training picks a
                random point of every voxel, evaluation and deployment the first one.

        Returns:
            The model inputs with ``grid_sample_data`` holding the voxel coordinate and the
            row of every representative, the inverse index of every input point and the
            representative count of every sample.

        Raises:
            ValueError: If the batch carries no point cloud.
        """
        point_batch = batch_inputs.multi_task_gt_batch.point_cloud_gt_batch
        if point_batch is None:
            raise ValueError("GridSamplePreprocessor needs the point cloud of the batch.")
        grid_coord = self.voxel_coords(point_batch.points[:, :3])
        voxel_keys = self.voxel_keys(grid_coord, point_batch.batch_indices)

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

        voxel_counts = torch.bincount(
            point_batch.batch_indices[representatives].long(), minlength=point_batch.batch_size
        )
        grid_sample_data = GridSampleData(
            grid_coords=grid_coord[representatives].to(torch.int32),
            representative_indices=representatives,
            inverse=inverse,
            offsets=torch.cumsum(voxel_counts, dim=0),
        )
        return batch_inputs.replace(grid_sample_data=grid_sample_data)

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
