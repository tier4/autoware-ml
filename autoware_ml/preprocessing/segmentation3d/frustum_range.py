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

"""Frustum-range preprocessing for Segmentation3D models."""

from __future__ import annotations

import math

import torch

from autoware_ml.dataclasses.geometry.range_view import RangeViewData
from autoware_ml.dataclasses.models.model_batch_inputs import ModelBatchInputs
from autoware_ml.preprocessing.segmentation3d.range_view import project_range


class FrustumRangePreprocessor:
    """Convert batched points into FRNet frustum and range-view tensors.

    The preprocessor projects points into range-view bins, groups them into
    frustum voxels, and assembles the tensors expected by FRNet. It is
    stateless and operates as a plain callable on the batch dictionary.
    """

    def __init__(
        self,
        height: int,
        width: int,
        fov_up: float,
        fov_down: float,
        ignore_index: int,
        num_classes: int,
    ) -> None:
        """Initialize the frustum range preprocessor.

        Args:
            height: Range-image height.
            width: Range-image width.
            fov_up: Upward field-of-view limit in degrees.
            fov_down: Downward field-of-view limit in degrees.
            ignore_index: Ignore label used for segmentation targets.
            num_classes: Number of trainable semantic classes.
        """
        self.height = int(height)
        self.width = int(width)
        self.fov_up = math.radians(float(fov_up))
        self.fov_down = math.radians(float(fov_down))
        self.ignore_index = int(ignore_index)
        self.num_classes = int(num_classes)

    def __call__(self, batch_inputs: ModelBatchInputs, *, is_training: bool) -> ModelBatchInputs:
        """Project the points of the batch into FRNet range-view tensors.

        Every point is assigned the range-view cell of its sample. When the batch carries
        labels, a dense semantic target image is computed by majority vote for each sample.

        Args:
            batch_inputs: Model inputs holding the point cloud and, when labelled, its labels.
            is_training: Whether the model is in training mode. The projection is the same
                in both modes.

        Returns:
            The model inputs with ``range_view_data`` holding the range-view cell of every
            point, the occupied cells, the cell index of every point and, for a labelled
            batch, the dense semantic target image of every sample.

        Raises:
            ValueError: If the batch carries no point cloud.
        """
        del is_training
        point_batch = batch_inputs.multi_task_gt_batch.point_cloud_gt_batch
        if point_batch is None:
            raise ValueError("FrustumRangePreprocessor needs the point cloud of the batch.")
        points = point_batch.points

        proj_y, proj_x = project_range(points, self.height, self.width, self.fov_up, self.fov_down)
        coors = torch.stack([point_batch.batch_indices.long(), proj_y, proj_x], dim=1)
        voxel_coors, inverse_map = torch.unique(coors, return_inverse=True, dim=0)

        label_batch = batch_inputs.multi_task_gt_batch.segmentation3d_gt_batch
        semantic_labels = (
            self._range_view_targets(
                coors,
                label_batch.gt_semantic_masks.long(),
                point_batch.batch_size,
                points.device,
            )
            if label_batch is not None
            else None
        )
        range_view_data = RangeViewData(
            coors=coors,
            voxel_coors=voxel_coors,
            inverse_map=inverse_map,
            semantic_labels=semantic_labels,
        )
        return batch_inputs.replace(range_view_data=range_view_data)

    def _range_view_targets(
        self,
        coors: torch.Tensor,
        labels: torch.Tensor,
        sample_count: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Build per-sample dense range-view label maps via majority vote.

        The computation is vectorized across the whole batch: per-point
        ``(batch, row, col, class)`` votes accumulate into one 4D tensor and
        the per-cell argmax produces the dense target image. Cells with no
        valid (non-ignored) points keep ``ignore_index``.

        Args:
            coors: Per-point range-view coordinates of shape ``(N, 3)`` with
                columns ``(batch_index, row, col)``.
            labels: Concatenated per-point semantic labels of shape ``(N,)``.
            sample_count: Number of samples in the batch.
            device: Target device for the output tensor.

        Returns:
            A tensor of shape ``(sample_count, height, width)`` containing
            the dense range-view semantic targets.
        """
        seg_label = torch.full(
            (sample_count, self.height, self.width),
            fill_value=self.ignore_index,
            dtype=torch.long,
            device=device,
        )

        valid = labels != self.ignore_index
        if not valid.any():
            return seg_label

        valid_batch = coors[valid, 0]
        valid_row = coors[valid, 1]
        valid_col = coors[valid, 2]
        valid_labels = labels[valid]

        counts = torch.zeros(
            (sample_count, self.height, self.width, self.num_classes),
            dtype=torch.float32,
            device=device,
        )
        counts.index_put_(
            (valid_batch, valid_row, valid_col, valid_labels),
            torch.ones_like(valid_labels, dtype=torch.float32),
            accumulate=True,
        )

        has_vote = counts.sum(dim=-1) > 0
        majority = counts.argmax(dim=-1)
        seg_label[has_vote] = majority[has_vote]
        return seg_label
