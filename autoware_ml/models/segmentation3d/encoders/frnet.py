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

"""FRNet encoder modules.

This module contains reusable feature encoding blocks used by FRNet-style range
and frustum segmentation models.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn

from autoware_ml.models.segmentation3d.norm import build_norm_1d
from autoware_ml.models.segmentation3d.structures import FRNetFeatureDict, FRNetInputs


class FrustumFeatureEncoder(nn.Module):
    """Encode point features into voxel and point feature pyramids.

    The encoder augments raw point features, applies per-point MLP layers, and
    aggregates voxel-level features for the FRNet backbone.
    """

    def __init__(
        self,
        in_channels: int,
        feat_channels: Sequence[int],
        with_distance: bool,
        with_cluster_center: bool,
        norm_eps: float,
        norm_momentum: float,
        with_pre_norm: bool,
        feat_compression: int | None,
    ) -> None:
        """Initialize the frustum feature encoder.

        Args:
            in_channels: Number of input point feature channels.
            feat_channels: Hidden dimensions for the point-wise MLP stack.
            with_distance: Whether to append radial distance as an input feature.
            with_cluster_center: Whether to append point-to-cluster-center offsets.
            norm_eps: Batch-normalization epsilon.
            norm_momentum: Batch-normalization momentum.
            with_pre_norm: Whether to normalize the augmented input features.
            feat_compression: Optional output compression dimension for voxel features.
        """
        super().__init__()
        feature_dim = in_channels
        if with_distance:
            feature_dim += 1
        if with_cluster_center:
            feature_dim += 3

        self.with_distance = with_distance
        self.with_cluster_center = with_cluster_center
        self.pre_norm = (
            build_norm_1d(feature_dim, norm_eps, norm_momentum) if with_pre_norm else None
        )

        layers: list[nn.Module] = []
        current_dim = feature_dim
        for layer_index, hidden_dim in enumerate(feat_channels):
            is_last = layer_index == len(feat_channels) - 1
            if is_last:
                layers.append(nn.Linear(current_dim, hidden_dim))
            else:
                layers.append(
                    nn.Sequential(
                        nn.Linear(current_dim, hidden_dim, bias=False),
                        build_norm_1d(hidden_dim, norm_eps, norm_momentum),
                        nn.ReLU(inplace=True),
                    )
                )
            current_dim = hidden_dim
        self.layers = nn.ModuleList(layers)
        self.compression = (
            nn.Sequential(nn.Linear(current_dim, feat_compression), nn.ReLU(inplace=True))
            if feat_compression is not None
            else None
        )

    def forward(self, voxel_dict: FRNetInputs) -> FRNetFeatureDict:
        """Encode raw points into voxel and point feature pyramids.

        Args:
            voxel_dict: Dictionary containing points, voxel coordinates, and inverse mapping.

        Returns:
            Updated voxel dictionary with point and voxel feature tensors.
        """
        points = voxel_dict["points"]
        inverse_map = voxel_dict["inverse_map"]
        num_voxels = inverse_map.max() + 1
        voxel_dict["voxel_coors"] = voxel_dict["voxel_coors"][:num_voxels]
        feature_list = [points]
        features = points

        if self.with_distance:
            feature_list.append(torch.norm(points[:, :3], dim=1, keepdim=True))

        if self.with_cluster_center:
            voxel_xyz_sum = torch.zeros(
                (num_voxels, features.shape[1]),
                device=features.device,
                dtype=features.dtype,
            )
            voxel_count = torch.zeros_like(voxel_xyz_sum)
            ones_tensor = torch.ones_like(features)
            index = torch.clamp(
                inverse_map.unsqueeze(-1).expand(-1, features.shape[1]),
                0,
                voxel_xyz_sum.shape[0] - 1,
            )
            voxel_xyz_sum.scatter_add_(dim=0, index=index, src=features)
            voxel_count.scatter_add_(dim=0, index=index, src=ones_tensor)
            voxel_mean = voxel_xyz_sum / voxel_count
            point_mean = voxel_mean[inverse_map]
            feature_list.append(features[:, :3] - point_mean[:, :3])

        point_features = torch.cat(feature_list, dim=1)
        if self.pre_norm is not None:
            point_features = self.pre_norm(point_features)

        point_feature_pyramid = []
        for layer in self.layers:
            point_features = layer(point_features)
            point_feature_pyramid.append(point_features)

        voxel_features = torch.full(
            (num_voxels, point_features.shape[1]),
            fill_value=torch.finfo(point_features.dtype).min,
            device=point_features.device,
            dtype=point_features.dtype,
        )
        voxel_features.scatter_reduce_(
            dim=0,
            index=inverse_map.unsqueeze(1).expand_as(point_features),
            src=point_features,
            reduce="amax",
            include_self=True,
        )

        if self.compression is not None:
            voxel_features = self.compression(voxel_features)

        voxel_dict["voxel_feats"] = voxel_features
        voxel_dict["point_feats"] = point_feature_pyramid
        return voxel_dict
