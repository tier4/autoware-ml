# Copyright 2023 OpenMMLab.
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

"""Pillar-based encoders for LiDAR detection.

This module contains the pillar encoders used by PointPillars-style models.
"""

from typing import Sequence

from jaxtyping import Float32
import torch
from torch import nn

from autoware_ml.models.detection3d.encoders.pillars.pfn_layer import PFNLayer
from autoware_ml.ops.voxelization.voxelization import VoxelsData


class PillarFeatureNet(nn.Module):
    """Decorate pillars and encode them into per-pillar descriptors.

    The encoder augments raw point features with pillar-relative offsets and
    applies stacked PFN layers to produce one descriptor per pillar.
    """

    def __init__(
        self,
        in_channels: int,
        feat_channels: Sequence[int],
        voxel_size: Sequence[float],
        point_cloud_range: Sequence[float],
        with_distance: bool = False,
        with_cluster_center: bool = True,
        with_voxel_center: bool = True,
    ) -> None:
        """Initialize the pillar feature network.

        Args:
            in_channels: Raw point feature dimension.
            feat_channels: PFN output channel widths.
            voxel_size: Voxel size in meters.
            point_cloud_range: Point cloud range used for voxelization.
            with_distance: Whether to append point distance from the origin.
            with_cluster_center: Whether to append cluster-center offsets.
            with_voxel_center: Whether to append voxel-center offsets.
        """
        super().__init__()
        self.with_distance = with_distance
        self.with_cluster_center = with_cluster_center
        self.with_voxel_center = with_voxel_center
        self.vx = float(voxel_size[0])
        self.vy = float(voxel_size[1])
        self.vz = float(voxel_size[2])
        self.x_offset = self.vx / 2 + float(point_cloud_range[0])
        self.y_offset = self.vy / 2 + float(point_cloud_range[1])
        self.z_offset = self.vz / 2 + float(point_cloud_range[2])

        feature_channels = in_channels
        if with_cluster_center:
            feature_channels += 3
        if with_voxel_center:
            feature_channels += 3
        if with_distance:
            feature_channels += 1
        self.feature_channels = feature_channels

        pfn_layers: list[nn.Module] = []
        layer_channels = [feature_channels] + list(feat_channels)
        for index in range(len(layer_channels) - 1):
            last_layer = index == len(layer_channels) - 2
            pfn_layers.append(
                PFNLayer(layer_channels[index], layer_channels[index + 1], last_layer=last_layer)
            )
        self.pfn_layers = nn.ModuleList(pfn_layers)

    def encode(
        self,
        voxels_data: VoxelsData,
    ) -> Float32[torch.Tensor, "num_pillars num_output_channels"]:
        """Build PointPillars decorated point features.

        Args:
            voxels_data: VoxelsData object containing padded voxel tensor,
                number of points, voxel coordinates, and batch indices.

        Returns:
            Encoded pillar feature tensor.
        """
        features = [voxels_data.voxels]
        voxels = voxels_data.voxels
        num_points = voxels_data.num_points
        coords = voxels_data.coords

        points_mean = voxels[:, :, :3].sum(dim=1, keepdim=True) / num_points.clamp_min(1).view(
            -1, 1, 1
        ).to(voxels.dtype)
        if self.with_cluster_center:
            features.append(voxels[:, :, :3] - points_mean)

        if self.with_voxel_center:
            center_offset = voxels.new_zeros((*voxels.shape[:2], 3))
            center_offset[:, :, 0] = voxels[:, :, 0] - (
                coords[:, 0].to(voxels.dtype).unsqueeze(1) * self.vx + self.x_offset
            )
            center_offset[:, :, 1] = voxels[:, :, 1] - (
                coords[:, 1].to(voxels.dtype).unsqueeze(1) * self.vy + self.y_offset
            )
            center_offset[:, :, 2] = voxels[:, :, 2] - (
                coords[:, 2].to(voxels.dtype).unsqueeze(1) * self.vz + self.z_offset
            )
            features.append(center_offset)

        if self.with_distance:
            features.append(torch.norm(voxels[:, :, :3], dim=2, keepdim=True))

        decorated = torch.cat(features, dim=-1)
        mask = torch.arange(voxels.shape[1], device=voxels.device).unsqueeze(
            0
        ) < num_points.unsqueeze(1)
        decorated = decorated * mask.unsqueeze(-1)

        if decorated.shape[-1] != self.feature_channels:
            raise ValueError(
                f"Decorated pillar features have {decorated.shape[-1]} channels, "
                f"expected {self.feature_channels}."
            )
        return decorated

    def encode_decorated(
        self,
        input_features: Float32[torch.Tensor, "num_pillars max_num_points feature_channels"],
    ) -> Float32[torch.Tensor, "num_pillars 1 num_output_channels"]:
        """Run the PFN layers over already-decorated pillar features.

        The singleton point dimension produced by the last PFN layer is kept so
        deployment can export this stage with the runtime pillar-feature ABI.

        Args:
            input_features: Decorated pillar features produced by :meth:`encode`.

        Returns:
            Per-pillar feature tensor with the singleton point dimension kept.
        """
        features = input_features
        for layer in self.pfn_layers:
            features = layer(features)
        return features

    def forward(
        self,
        voxels_data: VoxelsData,
    ) -> Float32[torch.Tensor, "num_pillars num_output_channels"]:
        """Encode padded voxel pillars into BEV pillar features.

        Args:
            voxels_data: VoxelsData object containing padded voxel tensor, number of points,
                voxel coordinates, and batch indices.

        Returns:
            Encoded pillar feature tensor after sequence of PFN layers.
        """
        return self.encode_decorated(self.encode(voxels_data)).squeeze(1)
