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

from typing import Sequence

import torch


from autoware_ml.dataclasses.models.model_batch_inputs import ModelBatchInputs
from autoware_ml.preprocessing.data_preprocessor_modules import DataPreprocessorModule
from autoware_ml.ops.voxelization.voxelization import hard_voxelize


class PointPillarPreprocessor(DataPreprocessorModule):
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
        eval_max_voxels: Maximum number of pillars retained per sample during
            evaluation and inference.
    """

    def __init__(
        self,
        voxel_size: Sequence[float],
        point_cloud_range: Sequence[float],
        max_num_points: int,
        max_voxels: int,
        eval_max_voxels: int,
    ) -> None:
        super().__init__()
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range
        self.max_num_points = max_num_points
        self.max_voxels = max_voxels
        self.eval_max_voxels = eval_max_voxels

    def __call__(
        self,
        multi_task_batch_inputs: ModelBatchInputs,
        *,
        is_training: bool,
    ) -> ModelBatchInputs:
        """
        Process batch data and convert to multi_task_input_features for downstream tasks.

        Args:
            multi_task_batch_inputs (ModelBatchInputs): Batch data containing ground truths and
            input features.
            is_training (bool): Flag indicating whether the model is in training mode.

        Returns:
            ModelBatchInputs: The processed input features for downstream tasks
            generating voxelization with VoxelData.
        """
        multi_task_gt_batch = multi_task_batch_inputs.multi_task_gt_batch
        if multi_task_gt_batch.point_cloud_gt_batch is None:
            raise ValueError("ModelGTBatch must contain point cloud data for voxelization.")

        points = multi_task_gt_batch.point_cloud_gt_batch.points
        device = points.device
        voxel_size = torch.tensor(self.voxel_size, device=device)
        point_cloud_range = torch.tensor(self.point_cloud_range, device=device)
        points_batch_indices = multi_task_gt_batch.point_cloud_gt_batch.batch_indices

        voxels_data = hard_voxelize(
            points,
            points_batch_indices=points_batch_indices,
            voxel_size=voxel_size,
            point_cloud_range=point_cloud_range,
            max_num_points=self.max_num_points,
            max_voxels=self.max_voxels if is_training else self.eval_max_voxels,
        )

        return multi_task_batch_inputs.replace(voxels_data=voxels_data)
