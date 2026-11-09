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

"""Lidar branch of the Sparse4D detector."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from jaxtyping import Float32, Int
import torch
import torch.nn as nn

from autoware_ml.models.detection3d.heads.sparse4d import SparseStage


class Sparse4DLidar(nn.Module):
    """Encode voxelized lidar points into the sparse stages the Sparse4D head samples.

    The branch owns the whole lidar path: the voxel encoder, the sparse middle encoder, the
    optional backbone refining its stages and the neck that turns them into the levels the head
    samples. That mirrors the BEVFusion lidar branch, so the detector only runs the head and a
    camera branch can join with its own path.
    """

    def __init__(
        self,
        pts_voxel_encoder: nn.Module,
        pts_middle_encoder: nn.Module,
        pts_backbone: nn.Module | None,
        pts_neck: nn.Module,
    ) -> None:
        """Initialize the lidar branch.

        Args:
            pts_voxel_encoder: Lidar voxel encoder describing every voxel with one feature vector.
            pts_middle_encoder: Sparse middle encoder encoding the voxel features and their
                coordinates into a hierarchy of stages, grouping its parameters with
                ``build_optimizer_groups``.
            pts_backbone: Optional sparse backbone refining the encoder stages before the neck,
                the stages go straight to the neck without one.
            pts_neck: Neck turning the stages into the ``num_levels`` levels the head samples.
        """
        super().__init__()
        self.pts_voxel_encoder = pts_voxel_encoder
        self.pts_middle_encoder = pts_middle_encoder
        self.pts_backbone = pts_backbone
        self.pts_neck = pts_neck

    @property
    def num_levels(self) -> int:
        """Return the number of levels the branch returns, which the head samples."""
        return self.pts_neck.num_levels

    def build_optimizer_groups(self) -> Mapping[str, Sequence[torch.nn.Parameter]]:
        """Group the middle encoder parameters apart from the newly initialized backbone and neck.

        Returns:
            The optimizer groups of the middle encoder and ``pts_backbone_neck``, the trainable
            parameters of the backbone and the neck.
        """
        groups = dict(self.pts_middle_encoder.build_optimizer_groups())
        refining = (
            [self.pts_neck] if self.pts_backbone is None else [self.pts_backbone, self.pts_neck]
        )
        groups["pts_backbone_neck"] = [
            parameter
            for module in refining
            for parameter in module.parameters()
            if parameter.requires_grad
        ]
        return groups

    def forward(
        self,
        voxels: Float32[torch.Tensor, "num_voxels max_num_points C"],
        coords: Int[torch.Tensor, "num_voxels 4"],
        num_points: Int[torch.Tensor, " num_voxels"],
        batch_size: int,
    ) -> list[SparseStage]:
        """Encode lidar voxels into the sparse levels the head samples.

        Args:
            voxels: Lidar voxel features, the padded points of every voxel.
            coords: Voxel coordinates in ``(batch, x, y, z)`` order, as the sparse middle
                encoder consumes them, with the voxels grouped by batch index.
            num_points: Number of points in each voxel.
            batch_size: Explicit batch size.

        Returns:
            Features, voxel coordinates and batch offsets of every level, from the finest to the
            coarsest.
        """
        voxel_features = self.pts_voxel_encoder(voxels, num_points)
        point = self.pts_middle_encoder(
            voxel_features=voxel_features, coords=coords, batch_size=batch_size
        )
        if self.pts_backbone is not None:
            point = self.pts_backbone(point)
        return self.pts_neck(point)
