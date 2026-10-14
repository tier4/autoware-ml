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

"""Lidar branch of the native BEVFusion detector."""

from __future__ import annotations

from typing import Sequence

from jaxtyping import Float32, Int32
import torch
import torch.nn as nn

from autoware_ml.models.detection3d.main_modules.bevfusions.fuser import ConvFuser


class BEVFusionLidar(nn.Module):
    """Encode voxelized lidar points into the BEV feature map BEVFusion fuses.

    The branch owns the whole lidar path: the voxel encoder, the BEV middle encoder, and the BEV
    backbone and neck that refine the resulting map. That mirrors the image branch, which carries
    its own backbone and neck, so each modality is complete before the two are fused.
    """

    def __init__(
        self,
        pts_voxel_encoder: nn.Module,
        pts_middle_encoder: nn.Module,
        pts_backbone: nn.Module,
        pts_neck: nn.Module,
        fuser: ConvFuser | None,
    ) -> None:
        """Initialize the lidar branch.

        Args:
            pts_voxel_encoder: Lidar voxel encoder.
            pts_middle_encoder: Lidar BEV middle encoder.
            pts_backbone: BEV backbone applied to the encoded map.
            pts_neck: BEV neck applied after the backbone.
            fuser: Optional fuser to fuse bev features from other modality.
        """
        super().__init__()
        self.pts_voxel_encoder = pts_voxel_encoder
        self.pts_middle_encoder = pts_middle_encoder
        self.pts_backbone = pts_backbone
        self.pts_neck = pts_neck
        self.fuser = fuser

    @property
    def expected_bev_shape(self) -> tuple[int, int]:
        """Return the expected ``(height, width)`` of lidar BEV features."""
        if not hasattr(self.pts_middle_encoder, "bev_output_shape"):
            raise AttributeError(
                "The lidar middle encoder does not have the attribute `bev_output_shape`. "
                "Please ensure that the middle encoder is properly configured to provide "
                "the expected BEV output shape."
            )
        return self.pts_middle_encoder.bev_output_shape  # type: ignore

    def forward(
        self,
        voxels: Float32[torch.Tensor, "num_voxels max_num_points C"],
        coords: Int32[torch.Tensor, "num_voxels 4"],
        num_points: Int32[torch.Tensor, " num_voxels"],
        batch_size: int,
        other_bev_features: Sequence[Float32[torch.Tensor, "batch_size num_channels height width"]]
        | None,
    ) -> Float32[torch.Tensor, "batch_size channels height width"]:
        """Encode lidar voxels into a BEV feature map.

        Args:
            voxels: Lidar voxel features.
            num_points: Number of points in each voxel.
            coords: Voxel coordinates in ``(batch, z, y, x)`` order, as the sparse middle
                encoder consumes them.
            batch_size: Explicit batch size.
            other_bev_features: Optional bev features from other modalities.

        Returns:
            BEV feature maps from lidar if fusion is disable, otherwise, BEV feature maps fused
            with other modalities.
        """
        voxel_features = self.pts_voxel_encoder(
            voxels=voxels,
            num_points=num_points,
            coords=coords,
        )
        bev_features = self.pts_middle_encoder(
            voxel_features=voxel_features, coords=coords, batch_size=batch_size
        )

        if other_bev_features is not None and self.fuser is not None:
            concat_bev_features = [*other_bev_features, bev_features]
            bev_features = self.fuser(concat_bev_features)

        bev_features = self.pts_neck(self.pts_backbone(bev_features))
        return bev_features

    def prepare_for_export(self) -> nn.Module:
        """Swap the middle encoder for its ONNX-exportable variant, in place.

        Returns:
            The exportable middle encoder that replaced the original, or None when the encoder has
            no exportable variant.
        """
        self.pts_middle_encoder = self.pts_middle_encoder.prepare_for_export()  # type: ignore
        return self.pts_middle_encoder
