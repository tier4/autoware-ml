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
from typing import Any

import torch


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
        self.fov = abs(self.fov_down) + abs(self.fov_up)
        self.ignore_index = int(ignore_index)
        self.num_classes = int(num_classes)

    def __call__(self, batch_inputs_dict: dict[str, Any]) -> dict[str, Any]:
        """Project concatenated point clouds into FRNet range-view tensors.

        Reads the concatenated batch produced by :meth:`DataModule.collate_fn`,
        derives per-point batch indices from ``offset``, projects every point
        into a 2D range-view cell, and returns the tensors expected by FRNet's
        voxel encoder and backbone.
        When per-point labels are provided, a dense semantic target image is
        computed via majority vote for each sample.

        Args:
            batch_inputs_dict: Batch dictionary containing the concatenated
                ``points`` tensor, the cumulative per-sample ``offset``
                tensor, and an optional concatenated ``pts_semantic_mask``.

        Returns:
            Dictionary with:
                * ``points``: the input point cloud, unchanged.
                * ``coors``: per-point ``(batch_index, row, col)`` range-view
                  coordinates.
                * ``voxel_coors``: unique range-view coordinates.
                * ``inverse_map``: index mapping from each point to its
                  ``voxel_coors`` entry.
                * ``sample_count``: number of samples in the batch.
                * ``pts_semantic_mask`` and ``semantic_seg`` when labels were
                  provided.
        """
        points: torch.Tensor = batch_inputs_dict["points"]
        offset: torch.Tensor = batch_inputs_dict["offset"]
        labels: torch.Tensor | None = batch_inputs_dict.get("pts_semantic_mask")
        device = points.device

        sample_count = int(offset.numel())
        lengths = torch.cat([offset[:1], offset[1:] - offset[:-1]])
        batch_index = torch.repeat_interleave(
            torch.arange(sample_count, device=device, dtype=torch.long), lengths
        )

        depth = torch.linalg.norm(points[:, :3], dim=1).clamp_min(1e-6)
        yaw = -torch.atan2(points[:, 1], points[:, 0])
        pitch = torch.arcsin(torch.clamp(points[:, 2] / depth, -1.0, 1.0))

        proj_x = torch.floor(0.5 * (yaw / torch.pi + 1.0) * self.width)
        proj_y = torch.floor((1.0 - (pitch + abs(self.fov_down)) / self.fov) * self.height)
        proj_x = proj_x.clamp(0, self.width - 1).long()
        proj_y = proj_y.clamp(0, self.height - 1).long()

        coors = torch.stack([batch_index, proj_y, proj_x], dim=1)
        voxel_coors, inverse_map = torch.unique(coors, return_inverse=True, dim=0)

        outputs: dict[str, Any] = {
            "points": points,
            "coors": coors,
            "voxel_coors": voxel_coors,
            "inverse_map": inverse_map,
            "sample_count": sample_count,
        }

        if labels is not None:
            labels = labels.long()
            outputs["pts_semantic_mask"] = labels
            outputs["semantic_seg"] = self._range_view_targets(coors, labels, sample_count, device)

        return outputs

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
