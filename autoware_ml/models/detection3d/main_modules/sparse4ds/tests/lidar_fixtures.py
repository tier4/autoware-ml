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

"""Fixtures building a small Sparse4D lidar branch on the 8 m test scene of the PTv3 fixtures.

The test scene is voxelized on a 1 m grid over ``[0, 8] x [0, 8]`` m, so the stub middle encoder
scatters the voxels onto an 8x8 BEV canvas with one metre cells.
"""

from __future__ import annotations

from jaxtyping import Float32, Int32
import torch
import torch.nn as nn

from autoware_ml.models.detection3d.backbones.second import SECONDBackbone
from autoware_ml.models.detection3d.encoders.bev_middle_encoder import BEVMiddleEncoder
from autoware_ml.models.detection3d.main_modules.sparse4ds.sparse4d_lidar import Sparse4DLidar
from autoware_ml.models.detection3d.necks.second_fpn import SECONDFPN

POINT_CHANNELS = 5
MIDDLE_CHANNELS = 16
NECK_CHANNELS = 32
BEV_SHAPE = (8, 8)


class StubVoxelEncoder(nn.Module):
    """Mean-pool the points of each voxel and project them to the middle encoder channels."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels)

    def forward(
        self,
        voxels: Float32[torch.Tensor, "num_voxels max_points channels"],
        num_points: Int32[torch.Tensor, " num_voxels"],
        coords: Int32[torch.Tensor, "num_voxels 4"],
    ) -> Float32[torch.Tensor, "num_voxels out_channels"]:
        voxel_mean = voxels.sum(dim=1) / num_points.clamp(min=1).unsqueeze(1).to(voxels.dtype)
        return self.linear(voxel_mean)


class StubMiddleEncoder(BEVMiddleEncoder):
    """Scatter voxel features onto a dense ``(B, C, H, W)`` canvas at their ``(y, x)`` cells."""

    def __init__(self, bev_shape: tuple[int, int], exportable: bool = False) -> None:
        super().__init__()
        self._bev_shape = bev_shape
        self.exportable = exportable

    @property
    def bev_output_shape(self) -> tuple[int, int]:
        return self._bev_shape

    def forward(
        self,
        voxel_features: Float32[torch.Tensor, "num_voxels channels"],
        coords: Int32[torch.Tensor, "num_voxels 4"],
        batch_size: int,
    ) -> Float32[torch.Tensor, "batch_size channels height width"]:
        height, width = self._bev_shape
        canvas = voxel_features.new_zeros(batch_size, voxel_features.shape[1], height, width)
        batch_indices, y, x = coords[:, 0].long(), coords[:, 2].long(), coords[:, 3].long()
        canvas[batch_indices, :, y, x] = voxel_features
        return canvas

    def prepare_for_export(self) -> StubMiddleEncoder:
        return StubMiddleEncoder(self._bev_shape, exportable=True)


def build_lidar_network() -> Sparse4DLidar:
    """Build a lidar branch from the stub encoders and a small SECOND backbone and neck."""
    return Sparse4DLidar(
        pts_voxel_encoder=StubVoxelEncoder(POINT_CHANNELS, MIDDLE_CHANNELS),
        pts_middle_encoder=StubMiddleEncoder(BEV_SHAPE),
        pts_backbone=SECONDBackbone(
            in_channels=MIDDLE_CHANNELS,
            out_channels=[16, 32],
            layer_nums=[1, 1],
            layer_strides=[1, 2],
        ),
        pts_neck=SECONDFPN(
            in_channels=[16, 32],
            out_channels=[NECK_CHANNELS // 2, NECK_CHANNELS // 2],
            upsample_strides=[1, 2],
        ),
    )
