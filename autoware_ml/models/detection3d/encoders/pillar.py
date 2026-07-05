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

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class PFNLayer(nn.Module):
    """Implement one PointPillars PFN layer.

    The layer applies a linear projection, normalization, and pooling over the
    points that belong to each pillar.
    """

    def __init__(self, in_channels: int, out_channels: int, last_layer: bool) -> None:
        """Initialize one PFN layer.

        Args:
            in_channels: Input feature dimension.
            out_channels: Output feature dimension.
            last_layer: Whether this is the final PFN stage.
        """
        super().__init__()
        units = out_channels if last_layer else out_channels // 2
        self.last_layer = last_layer
        self.linear = nn.Linear(in_channels, units, bias=False)
        self.norm = nn.BatchNorm1d(units, eps=1e-3, momentum=0.01)

    def forward(self, inputs: torch.Tensor, num_points: torch.Tensor) -> torch.Tensor:
        """Encode one PFN stage.

        Args:
            inputs: Decorated pillar feature tensor.
            num_points: Number of points in each pillar.

        Returns:
            Encoded pillar features.
        """
        x = self.linear(inputs)
        x = self.norm(x.reshape(-1, x.shape[-1])).reshape_as(x)
        x = F.relu(x, inplace=True)
        x_max = x.max(dim=1, keepdim=True).values
        if self.last_layer:
            return x_max
        x_repeat = x_max.repeat(1, inputs.shape[1], 1)
        return torch.cat([x, x_repeat], dim=2)


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
        self.x_offset = self.vx / 2 + float(point_cloud_range[0])
        self.y_offset = self.vy / 2 + float(point_cloud_range[1])

        feature_channels = in_channels
        if with_cluster_center:
            feature_channels += 3
        if with_voxel_center:
            feature_channels += 2
        if with_distance:
            feature_channels += 1

        pfn_layers: list[nn.Module] = []
        layer_channels = [feature_channels] + list(feat_channels)
        for index in range(len(layer_channels) - 1):
            last_layer = index == len(layer_channels) - 2
            pfn_layers.append(
                PFNLayer(layer_channels[index], layer_channels[index + 1], last_layer=last_layer)
            )
        self.pfn_layers = nn.ModuleList(pfn_layers)

    def forward(
        self, voxels: torch.Tensor, num_points: torch.Tensor, coords: torch.Tensor
    ) -> torch.Tensor:
        """Encode padded voxel pillars into BEV pillar features.

        Args:
            voxels: Padded voxel tensor.
            num_points: Number of points in each voxel.
            coords: Voxel coordinates with batch indices.

        Returns:
            Encoded pillar feature tensor.
        """
        features = [voxels]
        points_mean = voxels[:, :, :3].sum(dim=1, keepdim=True) / num_points.clamp_min(1).view(
            -1, 1, 1
        ).to(voxels.dtype)
        if self.with_cluster_center:
            features.append(voxels[:, :, :3] - points_mean)

        if self.with_voxel_center:
            center_offset = voxels.new_zeros((*voxels.shape[:2], 2))
            center_offset[:, :, 0] = voxels[:, :, 0] - (
                coords[:, 3].to(voxels.dtype).unsqueeze(1) * self.vx + self.x_offset
            )
            center_offset[:, :, 1] = voxels[:, :, 1] - (
                coords[:, 2].to(voxels.dtype).unsqueeze(1) * self.vy + self.y_offset
            )
            features.append(center_offset)

        if self.with_distance:
            features.append(torch.norm(voxels[:, :, :3], dim=2, keepdim=True))

        decorated = torch.cat(features, dim=-1)
        mask = torch.arange(voxels.shape[1], device=voxels.device).unsqueeze(
            0
        ) < num_points.unsqueeze(1)
        decorated = decorated * mask.unsqueeze(-1)

        for layer in self.pfn_layers:
            decorated = layer(decorated, num_points)
        return decorated.squeeze(1)


class PointPillarsScatter(nn.Module):
    """Scatter sparse pillar features to a dense BEV feature map.

    The scatter step converts sparse pillar descriptors into a dense 2D BEV map
    consumed by downstream convolutional backbones.
    """

    def __init__(self, in_channels: int, output_shape: Sequence[int]) -> None:
        """Initialize the dense BEV scatter module.

        Args:
            in_channels: Input feature dimension.
            output_shape: Output BEV shape as ``(height, width)``.
        """
        super().__init__()
        self.in_channels = in_channels
        self.output_shape = tuple(output_shape)

    def forward(
        self, pillar_features: torch.Tensor, coords: torch.Tensor, batch_size: int
    ) -> torch.Tensor:
        """Scatter pillar features into a dense BEV canvas.

        Args:
            pillar_features: Encoded pillar features.
            coords: Voxel coordinates with batch indices.
            batch_size: Batch size of the current sample set.

        Returns:
            Dense BEV feature map.
        """
        batch_indices = coords[:, 0].long()
        y_indices = coords[:, 2].long()
        x_indices = coords[:, 3].long()
        height, width = self.output_shape
        flat_indices = batch_indices * (height * width) + y_indices * width + x_indices

        canvas = pillar_features.new_zeros((batch_size * height * width, self.in_channels))
        scatter_indices = flat_indices.unsqueeze(1).expand(-1, self.in_channels)
        canvas = canvas.scatter(0, scatter_indices, pillar_features)
        return (
            canvas.view(batch_size, height, width, self.in_channels)
            .permute(0, 3, 1, 2)
            .contiguous()
        )
