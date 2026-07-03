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

"""PointPillars preprocessing for Detection3D models."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from autoware_ml.ops.voxelization import hard_voxelize


class PointPillarPreprocessor(nn.Module):
    """Convert batched point clouds into padded pillars for PointPillars models.

    The preprocessor voxelizes each point cloud using
    :func:`~autoware_ml.ops.voxelization.hard_voxelize`, pads variable-size
    pillars to ``max_num_points``, and packages the tensors expected by
    PointPillars-style detectors.

    Args:
        voxel_size: Voxel size along each axis ``[dx, dy, dz]`` in meters.
        point_cloud_range: Spatial range ``[x_min, y_min, z_min, x_max, y_max, z_max]``
            in meters.
        max_num_points: Maximum number of points kept per pillar.
        max_voxels: Maximum number of pillars retained per sample.
    """

    def __init__(
        self,
        voxel_size: list[float],
        point_cloud_range: list[float],
        max_num_points: int,
        max_voxels: int,
    ) -> None:
        super().__init__()
        self.register_buffer("voxel_size", torch.tensor(voxel_size, dtype=torch.float32))
        self.register_buffer(
            "point_cloud_range", torch.tensor(point_cloud_range, dtype=torch.float32)
        )
        self.max_num_points = max_num_points
        self.max_voxels = max_voxels

    def forward(self, batch_inputs_dict: dict[str, Any]) -> dict[str, Any]:
        """Voxelize batched point clouds and append pillar tensors.

        Args:
            batch_inputs_dict: Batch dictionary containing a ``"points"`` key
                with a list of ``(N_i, C)`` point tensors.

        Returns:
            Updated batch dictionary with the following additional keys:

            - ``"voxels"`` - padded pillar features ``(total_pillars, max_num_points, C)``.
            - ``"num_points"`` - per-pillar point counts ``(total_pillars,)``.
            - ``"voxel_coords"`` - pillar coordinates ``(total_pillars, 4)`` in
              ``[batch, z, y, x]`` order, ``dtype=torch.int32``.
        """
        device = batch_inputs_dict["points"][0].device
        voxel_size = self.voxel_size.to(device=device)
        point_cloud_range = self.point_cloud_range.to(device=device)

        batch_voxels: list[torch.Tensor] = []
        batch_num_points: list[torch.Tensor] = []
        batch_coords: list[torch.Tensor] = []

        for batch_index, points in enumerate(batch_inputs_dict["points"]):
            voxels, coords, num_points = hard_voxelize(
                points, voxel_size, point_cloud_range, self.max_num_points, self.max_voxels
            )
            batch_column = torch.full(
                (coords.shape[0], 1), batch_index, device=device, dtype=torch.int32
            )
            coords = torch.cat([batch_column, coords], dim=1)
            batch_voxels.append(voxels)
            batch_num_points.append(num_points)
            batch_coords.append(coords)

        outputs = dict(batch_inputs_dict)
        outputs["voxels"] = (
            torch.cat(batch_voxels, dim=0) if batch_voxels else torch.zeros((0, 0, 0))
        )
        outputs["num_points"] = (
            torch.cat(batch_num_points, dim=0)
            if batch_num_points
            else torch.zeros((0,), dtype=torch.int32)
        )
        outputs["voxel_coords"] = (
            torch.cat(batch_coords, dim=0)
            if batch_coords
            else torch.zeros((0, 4), dtype=torch.int32)
        )
        return outputs
