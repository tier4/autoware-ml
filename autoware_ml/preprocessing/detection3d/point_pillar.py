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

from jaxtyping import Float32
import torch

from autoware_ml.ops.voxelization.voxelization import hard_voxelize


class PointPillarPreprocessor:
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
        max_voxels: Maximum number of pillars retained per sample during training.
        eval_max_voxels: Maximum number of pillars retained per sample during
            evaluation and inference. Required before the preprocessor runs in
            evaluation mode.
        voxelization_z_order_first: If ``True``, this preprocessor will transpose [x, y, z]
            coordinates to [z, y, x] in coords from voxelization.
            This is used for backward-compatible, and will be removed very soon.
        default_point_channels: Default number of point channels to be used when no points
            are provided in the batch. Default is 4, which corresponds to (x, y, z, intensity).
    """

    # Add class attributes for type checking
    voxel_size: Float32[torch.Tensor, " 3"]
    point_cloud_range: Float32[torch.Tensor, " 6"]

    def __init__(
        self,
        voxel_size: list[float],
        point_cloud_range: list[float],
        max_num_points: int,
        max_voxels: int,
        eval_max_voxels: int | None = None,
        voxelization_z_order_first: bool = True,
        default_point_channels: int = 4,
    ) -> None:
        self.voxel_size = torch.tensor(voxel_size, dtype=torch.float32)
        self.point_cloud_range = torch.tensor(point_cloud_range, dtype=torch.float32)
        self.max_num_points = max_num_points
        self.max_voxels = max_voxels
        self.eval_max_voxels = eval_max_voxels
        self.voxelization_z_order_first = voxelization_z_order_first
        self._default_point_channels = default_point_channels

    def __call__(self, batch_inputs_dict: dict[str, Any], *, is_training: bool) -> dict[str, Any]:
        """Voxelize batched point clouds and append pillar tensors.

        Args:
            batch_inputs_dict: Batch dictionary containing a ``"points"`` key
                with a list of ``(N_i, C)`` point tensors.
            is_training: Whether the owning model is in training mode. Selects
                between the ``max_voxels`` (training) and ``eval_max_voxels``
                (evaluation) pillar budgets.

        Returns:
            Updated batch dictionary with the following additional keys:

            - ``"voxels"`` - padded pillar features ``(total_pillars, max_num_points, C)``.
            - ``"num_points"`` - per-pillar point counts ``(total_pillars,)``.
            - ``"voxel_coords"`` - pillar coordinates ``(total_pillars, 4)`` in
              ``[batch, z, y, x]`` order, ``dtype=torch.int32``.
            - ``"point_voxel_indices"`` - pillar row of every input point in the
              concatenated batch order ``(total_points,)``, ``-1`` when the point
              was not assigned to a retained pillar.
            - ``"num_dropped_voxels"`` - occupied pillars discarded by the pillar budget.
        """
        if not is_training and self.eval_max_voxels is None:
            raise ValueError(
                "PointPillarPreprocessor is running in evaluation mode but 'eval_max_voxels' "
                "is not set. Set 'eval_max_voxels' in the data_preprocessing config (use the "
                "same value as 'max_voxels' to keep the training-time budget)."
            )
        points_list = batch_inputs_dict["points"]
        outputs = dict(batch_inputs_dict)
        if not points_list:
            outputs["voxels"] = self.voxel_size.new_zeros(
                (0, self.max_num_points, self._default_point_channels)
            )
            outputs["num_points"] = torch.zeros(
                (0,), device=self.voxel_size.device, dtype=torch.int32
            )
            outputs["voxel_coords"] = torch.zeros(
                (0, 4), device=self.voxel_size.device, dtype=torch.int32
            )
            outputs["point_voxel_indices"] = torch.zeros(
                (0,), device=self.voxel_size.device, dtype=torch.int64
            )
            outputs["num_dropped_voxels"] = torch.zeros(
                (), device=self.voxel_size.device, dtype=torch.int64
            )
            return outputs

        device = points_list[0].device
        voxel_size = self.voxel_size.to(device=device)
        point_cloud_range = self.point_cloud_range.to(device=device)

        # Concat all points across a batch size to a single tensor for voxelization, but keep track of the batch index
        # (N*B, point dimension)
        points = torch.cat(points_list, dim=0)
        # (N*B,) where each point has a batch index
        points_batch_indices = torch.cat(
            [
                torch.full((p.shape[0],), i, device=device, dtype=torch.int32)
                for i, p in enumerate(points_list)
            ],
            dim=0,
        )
        voxels_data = hard_voxelize(
            points,
            points_batch_indices=points_batch_indices,
            voxel_size=voxel_size,
            point_cloud_range=point_cloud_range,
            max_num_points=self.max_num_points,
            max_voxels=self.max_voxels if is_training else self.eval_max_voxels,
        )

        # Handle the case where no voxels are generated
        if not len(voxels_data.voxels):
            outputs["voxels"] = points.new_zeros((0, self.max_num_points, points.shape[1]))
            outputs["num_points"] = torch.zeros((0,), device=points.device, dtype=torch.int32)
            outputs["voxel_coords"] = torch.zeros((0, 4), device=points.device, dtype=torch.int32)
            outputs["point_voxel_indices"] = voxels_data.point_voxel_indices
            outputs["num_dropped_voxels"] = voxels_data.num_dropped_voxels
            return outputs

        # Concat batch column to the voxel coordinates
        batch_coords = torch.cat(
            [voxels_data.batch_indices.unsqueeze(1), voxels_data.coords], dim=1
        )
        batch_voxels = voxels_data.voxels
        batch_num_points = voxels_data.num_points

        # TODO (KokSeang): Remove this backward compatibility code in the future
        if self.voxelization_z_order_first:
            # Transpose [x, y, z] to [z, y, x] for backward compatibility
            batch_coords = batch_coords[:, [0, 3, 2, 1]].contiguous()

        outputs["voxels"] = batch_voxels
        outputs["num_points"] = batch_num_points
        outputs["voxel_coords"] = batch_coords
        outputs["point_voxel_indices"] = voxels_data.point_voxel_indices
        outputs["num_dropped_voxels"] = voxels_data.num_dropped_voxels
        return outputs
