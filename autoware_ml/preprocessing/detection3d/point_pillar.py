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

from jaxtyping import Float32
import torch

from autoware_ml.dataclasses.models.model_batch_inputs import ModelBatchInputs
from autoware_ml.ops.voxelization.voxelization import hard_voxelize


class PointPillarPreprocessor:
    """Convert the batched point cloud into padded pillars for PointPillars models.

    The preprocessor voxelizes the points of the batch using
    :func:`~autoware_ml.ops.voxelization.hard_voxelize`, pads every voxel to
    ``max_num_points`` points, and adds the voxels to the model inputs.

    Args:
        voxel_size: Voxel size along each axis ``[dx, dy, dz]`` in meters.
        point_cloud_range: Spatial range ``[x_min, y_min, z_min, x_max, y_max, z_max]``
            in meters.
        max_num_points: Maximum number of points kept per pillar.
        max_voxels: Maximum number of pillars retained per sample during training.
        eval_max_voxels: Maximum number of pillars retained per sample during
            evaluation and inference. Required before the preprocessor runs in
            evaluation mode.
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
    ) -> None:
        self.voxel_size = torch.tensor(voxel_size, dtype=torch.float32)
        self.point_cloud_range = torch.tensor(point_cloud_range, dtype=torch.float32)
        self.max_num_points = max_num_points
        self.max_voxels = max_voxels
        self.eval_max_voxels = eval_max_voxels

    def __call__(self, batch_inputs: ModelBatchInputs, *, is_training: bool) -> ModelBatchInputs:
        """Voxelize batched point clouds and append pillar tensors.

        Args:
            batch_inputs: Model inputs holding the point cloud batch.
            is_training: Whether the owning model is in training mode. Selects
                between the ``max_voxels`` (training) and ``eval_max_voxels``
                (evaluation) pillar budgets.

        Returns:
            The model inputs with ``voxels_data`` holding the padded pillars, their point counts,
            their coordinates in (x, y, z) order and their sample indices.

        Raises:
            ValueError: If the batch carries no point cloud, or the preprocessor runs in
                evaluation mode without an evaluation budget.
        """
        if not is_training and self.eval_max_voxels is None:
            raise ValueError(
                "PointPillarPreprocessor is running in evaluation mode but 'eval_max_voxels' "
                "is not set. Set 'eval_max_voxels' in the data_preprocessing config (use the "
                "same value as 'max_voxels' to keep the training-time budget)."
            )
        point_batch = batch_inputs.multi_task_gt_batch.point_cloud_gt_batch
        if point_batch is None:
            raise ValueError("PointPillarPreprocessor needs the point cloud of the batch.")

        device = point_batch.points.device
        voxels_data = hard_voxelize(
            point_batch.points,
            points_batch_indices=point_batch.batch_indices,
            voxel_size=self.voxel_size.to(device=device),
            point_cloud_range=self.point_cloud_range.to(device=device),
            max_num_points=self.max_num_points,
            max_voxels=self.max_voxels if is_training else self.eval_max_voxels,
        )
        return batch_inputs.replace(voxels_data=voxels_data)
