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

from autoware_ml.models.detection3d.backbones.sparse_backbone import SparseBackbone
from autoware_ml.models.detection3d.heads.sparse4d import SparseStage
from autoware_ml.models.detection3d.necks.sparse_neck import SparseNeck
from autoware_ml.models.segmentation3d.encoders.voxel import VoxelFeatureEncoder


class Sparse4DLidar(nn.Module):
    """Encode voxelized lidar points into the sparse stages the Sparse4D head samples.

    The branch owns the whole lidar path: the voxel encoder, the sparse backbone and the neck
    that turns the backbone stages into the levels the head samples. That mirrors the BEVFusion
    lidar branch, so the detector only runs the head and a camera branch can join with its own
    path.
    """

    def __init__(
        self,
        pts_voxel_encoder: VoxelFeatureEncoder,
        pts_backbone: SparseBackbone,
        pts_neck: SparseNeck,
    ) -> None:
        """Initialize the lidar branch.

        Args:
            pts_voxel_encoder: Lidar voxel encoder describing every voxel with one feature vector.
            pts_backbone: Sparse backbone encoding the voxel features into a hierarchy of stages.
            pts_neck: Neck turning the backbone stages into the levels the head samples.

        Raises:
            TypeError: If the backbone does not implement :class:`SparseBackbone` or the neck
                :class:`SparseNeck`.
            ValueError: If the voxel encoder and the backbone disagree on the voxel features.
        """
        super().__init__()
        if not isinstance(pts_backbone, SparseBackbone):
            raise TypeError(
                f"The lidar backbone must be a SparseBackbone, got {type(pts_backbone).__name__}."
            )
        if not isinstance(pts_neck, SparseNeck):
            raise TypeError(f"The lidar neck must be a SparseNeck, got {type(pts_neck).__name__}.")
        if pts_backbone.in_channels != pts_voxel_encoder.out_channels:
            raise ValueError(
                f"The backbone reads {pts_backbone.in_channels} features per voxel but the voxel "
                f"encoder describes a voxel with {pts_voxel_encoder.out_channels}."
            )
        self.pts_voxel_encoder = pts_voxel_encoder
        self.pts_backbone = pts_backbone
        self.pts_neck = pts_neck

    @property
    def num_levels(self) -> int:
        """Return the number of levels the branch returns, which the head samples."""
        return self.pts_neck.num_levels

    def build_optimizer_groups(self) -> Mapping[str, Sequence[torch.nn.Parameter]]:
        """Group the backbone parameters apart from the newly initialized neck.

        Returns:
            The optimizer groups of the backbone and ``pts_neck``, the trainable neck parameters.
        """
        groups = dict(self.pts_backbone.build_optimizer_groups())
        groups["pts_neck"] = [
            parameter for parameter in self.pts_neck.parameters() if parameter.requires_grad
        ]
        return groups

    def forward(
        self,
        voxels: Float32[torch.Tensor, "num_voxels max_num_points C"],
        coords: Int[torch.Tensor, "num_voxels 4"],
        num_points: Int[torch.Tensor, " num_voxels"],
        time_lag_column: int,
    ) -> list[SparseStage]:
        """Encode lidar voxels into the sparse levels the head samples.

        Args:
            voxels: Lidar voxel features, the padded points of every voxel.
            coords: Voxel coordinates in ``(batch, x, y, z)`` order, as the sparse backbone
                consumes them, with the voxels grouped by batch index.
            num_points: Number of points in each voxel.
            time_lag_column: Point column holding the time lag, ``-1`` when the points have
                none; the voxel encoder checks it against the layout it reads.

        Returns:
            Features, voxel coordinates and batch offsets of every level, from the finest to the
            coarsest.

        Raises:
            ValueError: If the points do not follow the layout the voxel encoder reads.
        """
        self.pts_voxel_encoder.check_points(voxels, time_lag_column)
        voxel_features = self.pts_voxel_encoder(voxels, num_points)
        point = self.pts_backbone(voxel_features=voxel_features, coords=coords)
        return self.pts_neck(point)
