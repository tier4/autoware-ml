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

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class FrustumRangePreprocessor(nn.Module):
    """Convert batched points into FRNet frustum and range-view tensors.

    The preprocessor projects points into range-view bins, groups them into
    frustum voxels, and assembles the tensors expected by FRNet.
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
        super().__init__()
        self.height = height
        self.width = width
        # Store FOV limits as plain Python floats so they work transparently
        # on any device without requiring register_buffer or .to() calls.
        self.fov_up = float(torch.deg2rad(torch.tensor(float(fov_up))))
        self.fov_down = float(torch.deg2rad(torch.tensor(float(fov_down))))
        self.fov = abs(self.fov_down) + abs(self.fov_up)
        self.ignore_index = ignore_index
        self.num_classes = num_classes

    def forward(self, batch_inputs_dict: dict[str, Any]) -> dict[str, Any]:
        """Project batched point clouds into FRNet range-view tensors.

        Args:
            batch_inputs_dict: Batch dictionary containing ``points`` and optional labels.

        Returns:
            Updated batch dictionary with range-view coordinates and labels.
        """
        points_batch: list[torch.Tensor] = batch_inputs_dict["points"]
        labels_batch: list[torch.Tensor] | None = batch_inputs_dict.get("pts_semantic_mask")

        all_points = []
        all_point_labels = []
        all_coors = []
        semantic_seg = []

        for batch_index, points in enumerate(points_batch):
            device = points.device
            depth = torch.linalg.norm(points[:, :3], dim=1).clamp_min(1e-6)
            yaw = -torch.atan2(points[:, 1], points[:, 0])
            pitch = torch.arcsin(torch.clamp(points[:, 2] / depth, -1.0, 1.0))

            proj_x = torch.floor(0.5 * (yaw / torch.pi + 1.0) * self.width)
            proj_y = torch.floor((1.0 - (pitch + abs(self.fov_down)) / self.fov) * self.height)
            proj_x = proj_x.clamp(0, self.width - 1).long()
            proj_y = proj_y.clamp(0, self.height - 1).long()

            sample_coors = torch.stack([proj_y, proj_x], dim=1)
            batch_column = torch.full(
                (sample_coors.size(0), 1), batch_index, device=device, dtype=torch.long
            )
            sample_coors = torch.cat([batch_column, sample_coors], dim=1)

            all_points.append(points)
            all_coors.append(sample_coors)

            if labels_batch is not None:
                point_labels = labels_batch[batch_index].long()
                all_point_labels.append(point_labels)
                semantic_seg.append(self._majority_vote(sample_coors, point_labels, device))

        points = torch.cat(all_points, dim=0)
        coors = torch.cat(all_coors, dim=0)
        voxel_coors, inverse_map = torch.unique(coors, return_inverse=True, dim=0)

        outputs: dict[str, Any] = {
            "points": points,
            "coors": coors,
            "voxel_coors": voxel_coors,
            "inverse_map": inverse_map,
            "batch_size": len(points_batch),
        }

        if all_point_labels:
            outputs["pts_semantic_mask"] = torch.cat(all_point_labels, dim=0)
            outputs["semantic_seg"] = torch.stack(semantic_seg, dim=0)

        return outputs

    def _majority_vote(
        self, coors: torch.Tensor, labels: torch.Tensor, device: torch.device
    ) -> torch.Tensor:
        """Assign one semantic label to each projected range-view cell.

        Args:
            coors: Projected point coordinates with batch, row, and column indices.
            labels: Point-wise semantic labels.
            device: Target device for the output tensor.

        Returns:
            Dense semantic label map for one range-view sample.
        """
        seg_label = torch.full(
            (self.height, self.width), fill_value=self.ignore_index, dtype=torch.long, device=device
        )

        if coors.numel() == 0 or labels.numel() == 0:
            return seg_label

        unique_coors, inverse = torch.unique(coors[:, 1:], return_inverse=True, dim=0)
        valid = labels != self.ignore_index
        if not valid.any():
            return seg_label

        counts = torch.zeros(
            (unique_coors.size(0), self.num_classes), dtype=torch.float32, device=device
        )
        counts.scatter_add_(
            dim=0,
            index=inverse[valid].unsqueeze(1).expand(-1, self.num_classes),
            src=F.one_hot(labels[valid], num_classes=self.num_classes).float(),
        )
        valid_cells = counts.sum(dim=1) > 0
        majority = counts[valid_cells].argmax(dim=1)
        seg_label[unique_coors[valid_cells, 0], unique_coors[valid_cells, 1]] = majority
        return seg_label
