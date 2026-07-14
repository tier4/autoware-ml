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

"""Reusable feature extractors shared by native detection3d models.

This module groups the common lidar-to-BEV and multiview-image feature
encoding paths reused across lidar, camera, and fusion detectors.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn

from autoware_ml.utils.point_cloud.batching import infer_batch_size_from_voxel_coords


class LidarBEVFeatureExtractor(nn.Module):
    """Encode voxelized lidar inputs into BEV feature maps.

    The extractor composes the shared PointPillars-style voxel encoder, middle
    encoder, optional BEV backbone, and optional BEV neck used by lidar and
    fusion detectors.
    """

    def __init__(
        self,
        pts_voxel_encoder: nn.Module,
        pts_middle_encoder: nn.Module,
        pts_backbone: nn.Module | None = None,
        pts_neck: nn.Module | None = None,
    ) -> None:
        """Initialize the shared lidar BEV extractor."""
        super().__init__()
        self.pts_voxel_encoder = pts_voxel_encoder
        self.pts_middle_encoder = pts_middle_encoder
        self.pts_backbone = pts_backbone
        self.pts_neck = pts_neck

    def forward(
        self,
        voxels: torch.Tensor,
        num_points: torch.Tensor,
        voxel_coords: torch.Tensor,
        batch_size: int | None = None,
    ) -> torch.Tensor:
        """Encode voxelized lidar inputs into BEV features."""
        if batch_size is None:
            batch_size = infer_batch_size_from_voxel_coords(voxel_coords)
        pillar_features = self.pts_voxel_encoder(voxels, num_points, voxel_coords)
        bev_features = self.pts_middle_encoder(pillar_features, voxel_coords, batch_size=batch_size)
        if self.pts_backbone is not None:
            bev_features = self.pts_backbone(bev_features)
        if self.pts_neck is not None:
            bev_features = self.pts_neck(bev_features)
        return bev_features


class MultiviewImageFeatureExtractor(nn.Module):
    """Encode multiview image batches into shared neck features."""

    def __init__(self, img_backbone: nn.Module, img_neck: nn.Module) -> None:
        """Initialize the multiview image feature extractor."""
        super().__init__()
        self.img_backbone = img_backbone
        self.img_neck = img_neck

    def forward(self, image_batch: torch.Tensor | Sequence[torch.Tensor]) -> torch.Tensor:
        """Encode multiview image batches into neck features.

        Args:
            image_batch: Multiview image tensor or per-sample image sequence
                with shape ``(B, N, C, H, W)`` after stacking.

        Returns:
            Primary image neck feature tensor with shape
            ``(B, N, C, H, W)``.
        """
        image_batch = (
            torch.stack(list(image_batch), dim=0).float()
            if isinstance(image_batch, (list, tuple))
            else image_batch.float()
        )
        if image_batch.dim() != 5:
            raise ValueError(
                f"Expected image batch with shape (B, N, C, H, W), got {tuple(image_batch.shape)}"
            )
        batch_size, num_cams = image_batch.shape[:2]
        flat_images = image_batch.view(batch_size * num_cams, *image_batch.shape[2:])
        image_features = self.img_backbone(flat_images)
        if isinstance(image_features, torch.Tensor):
            image_features = (image_features,)
        image_features = self.img_neck(image_features)
        primary_feature = (
            image_features[0] if isinstance(image_features, (list, tuple)) else image_features
        )
        return primary_feature.view(batch_size, num_cams, *primary_feature.shape[1:])
