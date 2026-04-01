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

"""FRNet backbone modules.

This module contains the residual frustum-range backbone used by FRNet.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from autoware_ml.models.segmentation3d.norm import build_norm_1d, build_norm_2d
from autoware_ml.models.segmentation3d.structures import FRNetFeatureDict


class BasicBlock(nn.Module):
    """Implement the residual block used by :class:`FRNetBackbone`.

    The block applies two 2D convolutions with a residual shortcut in the
    range-view backbone.
    """

    def __init__(
        self, inplanes: int, planes: int, stride: int, dilation: int, eps: float, momentum: float
    ) -> None:
        """Initialize the FRNet residual block.

        Args:
            inplanes: Input channel count.
            planes: Output channel count.
            stride: Stride of the first convolution.
            dilation: Dilation factor for the main branch.
            eps: BatchNorm epsilon.
            momentum: BatchNorm momentum.
        """
        super().__init__()
        self.conv1 = nn.Conv2d(
            inplanes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=dilation,
            dilation=dilation,
            bias=False,
        )
        self.bn1 = build_norm_2d(planes, eps, momentum)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = build_norm_2d(planes, eps, momentum)
        self.relu = nn.Hardswish(inplace=True)
        self.downsample = None
        if stride != 1 or inplanes != planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False),
                build_norm_2d(planes, eps, momentum),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the residual block.

        Args:
            x: Input feature map.

        Returns:
            Output feature map after residual fusion.
        """
        identity = x if self.downsample is None else self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)


class FRNetBackbone(nn.Module):
    """Encode frustum and range-view features with a fusion backbone.

    The backbone processes range-view tensors, fuses them with point features,
    and returns multi-scale features for FRNet heads.
    """

    stage_blocks = {18: (2, 2, 2, 2), 34: (3, 4, 6, 3)}

    def __init__(
        self,
        in_channels: int,
        point_in_channels: int,
        output_shape: Sequence[int],
        depth: int,
        stem_channels: int,
        out_channels: Sequence[int],
        strides: Sequence[int],
        dilations: Sequence[int],
        fuse_channels: Sequence[int],
        norm_eps: float,
        norm_momentum: float,
    ) -> None:
        """Initialize the FRNet backbone.

        Args:
            in_channels: Input frustum feature channels.
            point_in_channels: Input point feature channels.
            output_shape: Spatial shape of the frustum grid.
            depth: Backbone depth identifier.
            stem_channels: Stem channel count.
            out_channels: Stage output channel counts.
            strides: Stage strides.
            dilations: Stage dilations.
            fuse_channels: Final fusion channel widths.
            norm_eps: BatchNorm epsilon.
            norm_momentum: BatchNorm momentum.
        """
        super().__init__()
        if depth not in self.stage_blocks:
            raise ValueError(f"Unsupported FRNet depth: {depth}")

        self.output_shape = tuple(output_shape)
        self.strides = []
        self.stem = self._make_stem_layer(in_channels, stem_channels, norm_eps, norm_momentum)
        self.point_stem = self._make_point_layer(
            point_in_channels, stem_channels, norm_eps, norm_momentum
        )
        self.fusion_stem = self._make_fusion_layer(
            stem_channels * 2, stem_channels, norm_eps, norm_momentum
        )

        current_channels = stem_channels
        residual_layers = []
        point_fusion_layers = []
        pixel_fusion_layers = []
        attention_layers = []
        overall_stride = 1
        for num_blocks, out_channel, stride, dilation in zip(
            self.stage_blocks[depth], out_channels, strides, dilations
        ):
            blocks = [
                BasicBlock(current_channels, out_channel, stride, dilation, norm_eps, norm_momentum)
            ]
            blocks.extend(
                BasicBlock(out_channel, out_channel, 1, dilation, norm_eps, norm_momentum)
                for _ in range(1, num_blocks)
            )
            residual_layers.append(nn.Sequential(*blocks))
            point_fusion_layers.append(
                self._make_point_layer(
                    current_channels + out_channel, out_channel, norm_eps, norm_momentum
                )
            )
            pixel_fusion_layers.append(
                self._make_fusion_layer(out_channel * 2, out_channel, norm_eps, norm_momentum)
            )
            attention_layers.append(
                self._make_attention_layer(out_channel, norm_eps, norm_momentum)
            )
            current_channels = out_channel
            overall_stride *= stride
            self.strides.append(overall_stride)

        self.residual_layers = nn.ModuleList(residual_layers)
        self.point_fusion_layers = nn.ModuleList(point_fusion_layers)
        self.pixel_fusion_layers = nn.ModuleList(pixel_fusion_layers)
        self.attention_layers = nn.ModuleList(attention_layers)

        fused_in_channels = stem_channels + sum(out_channels)
        self.fuse_layers = nn.ModuleList()
        self.point_fuse_layers = nn.ModuleList()
        for fuse_channel in fuse_channels:
            self.fuse_layers.append(
                self._make_fusion_layer(fused_in_channels, fuse_channel, norm_eps, norm_momentum)
            )
            self.point_fuse_layers.append(
                self._make_point_layer(fused_in_channels, fuse_channel, norm_eps, norm_momentum)
            )
            fused_in_channels = fuse_channel

    def _make_stem_layer(
        self, in_channels: int, out_channels: int, eps: float, momentum: float
    ) -> nn.Module:
        """Build the initial frustum stem.

        Args:
            in_channels: Input channel count.
            out_channels: Output channel count.
            eps: BatchNorm epsilon.
            momentum: BatchNorm momentum.

        Returns:
            Sequential frustum stem module.
        """
        layers = []
        channel_plan = [out_channels // 2, out_channels, out_channels]
        current = in_channels
        for next_channel in channel_plan:
            layers.extend(
                [
                    nn.Conv2d(current, next_channel, kernel_size=3, padding=1, bias=False),
                    build_norm_2d(next_channel, eps, momentum),
                    nn.Hardswish(inplace=True),
                ]
            )
            current = next_channel
        return nn.Sequential(*layers)

    def _make_point_layer(
        self, in_channels: int, out_channels: int, eps: float, momentum: float
    ) -> nn.Module:
        """Build a point feature projection layer.

        Args:
            in_channels: Input channel count.
            out_channels: Output channel count.
            eps: BatchNorm epsilon.
            momentum: BatchNorm momentum.

        Returns:
            Sequential point projection layer.
        """
        return nn.Sequential(
            nn.Linear(in_channels, out_channels, bias=False),
            build_norm_1d(out_channels, eps, momentum),
            nn.ReLU(inplace=True),
        )

    def _make_fusion_layer(
        self, in_channels: int, out_channels: int, eps: float, momentum: float
    ) -> nn.Module:
        """Build a frustum fusion layer.

        Args:
            in_channels: Input channel count.
            out_channels: Output channel count.
            eps: BatchNorm epsilon.
            momentum: BatchNorm momentum.

        Returns:
            Sequential frustum fusion layer.
        """
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            build_norm_2d(out_channels, eps, momentum),
            nn.Hardswish(inplace=True),
        )

    def _make_attention_layer(self, channels: int, eps: float, momentum: float) -> nn.Module:
        """Build a spatial attention layer.

        Args:
            channels: Feature channel count.
            eps: BatchNorm epsilon.
            momentum: BatchNorm momentum.

        Returns:
            Spatial attention module.
        """
        return nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            build_norm_2d(channels, eps, momentum),
            nn.Hardswish(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            build_norm_2d(channels, eps, momentum),
            nn.Sigmoid(),
        )

    def forward(self, voxel_dict: FRNetFeatureDict) -> FRNetFeatureDict:
        """Apply FRNet backbone and update the voxel dictionary.

        Args:
            voxel_dict: FRNet feature dictionary containing voxel and point tensors.

        Returns:
            Updated feature dictionary with fused voxel and point features.
        """
        point_feats = voxel_dict["point_feats"][-1]
        voxel_feats = voxel_dict["voxel_feats"]
        voxel_coors = voxel_dict["voxel_coors"]
        point_coors = voxel_dict["coors"]
        inverse_map = voxel_dict["inverse_map"]
        sample_count = voxel_dict["sample_count"]

        x = self._frustum_to_pixel(voxel_feats, voxel_coors, sample_count, stride=1)
        x = self.stem(x)
        point_feats = self.point_stem(
            torch.cat([self._pixel_to_point(x, point_coors, 1), point_feats], dim=1)
        )
        stride_voxel_coors, frustum_feats = self._point_to_frustum(
            point_feats, voxel_coors, inverse_map, 1
        )
        x = self.fusion_stem(
            torch.cat(
                [self._frustum_to_pixel(frustum_feats, stride_voxel_coors, sample_count, 1), x],
                dim=1,
            )
        )

        pixel_outputs = [x]
        point_outputs = [point_feats]
        for layer, point_fusion, pixel_fusion, attention, stride in zip(
            self.residual_layers,
            self.point_fusion_layers,
            self.pixel_fusion_layers,
            self.attention_layers,
            self.strides,
        ):
            x = layer(x)
            point_feats = point_fusion(
                torch.cat([self._pixel_to_point(x, point_coors, stride), point_feats], dim=1)
            )
            stride_voxel_coors, frustum_feats = self._point_to_frustum(
                point_feats, voxel_coors, inverse_map, stride
            )
            fused_pixel = pixel_fusion(
                torch.cat(
                    [
                        self._frustum_to_pixel(
                            frustum_feats, stride_voxel_coors, sample_count, stride
                        ),
                        x,
                    ],
                    dim=1,
                )
            )
            x = fused_pixel * attention(fused_pixel) + x
            pixel_outputs.append(x)
            point_outputs.append(point_feats)

        target_size = pixel_outputs[0].shape[2:]
        pixel_outputs = [
            F.interpolate(output, size=target_size, mode="bilinear", align_corners=True)
            for output in pixel_outputs
        ]
        fused_pixels = torch.cat(pixel_outputs, dim=1)
        fused_points = torch.cat(point_outputs, dim=1)

        for pixel_fuse, point_fuse in zip(self.fuse_layers, self.point_fuse_layers):
            fused_pixels = pixel_fuse(fused_pixels)
            fused_points = point_fuse(fused_points)

        voxel_dict["voxel_feats"] = [fused_pixels, *pixel_outputs[1:]]
        voxel_dict["point_feats_backbone"] = [fused_points, *point_outputs[1:]]
        return voxel_dict

    def _frustum_to_pixel(
        self,
        frustum_features: torch.Tensor,
        coors: torch.Tensor,
        sample_count: int,
        stride: int,
    ) -> torch.Tensor:
        """Scatter frustum features into a dense pixel grid.

        Args:
            frustum_features: Sparse frustum feature tensor.
            coors: Frustum coordinates.
            sample_count: Number of samples represented by the batch.
            stride: Spatial stride applied to the target grid.

        Returns:
            Dense pixel feature map.
        """
        height = self.output_shape[0] // stride
        width = self.output_shape[1] // stride
        pixel_features = torch.zeros(
            (sample_count, height, width, frustum_features.shape[1]),
            device=frustum_features.device,
            dtype=frustum_features.dtype,
        )
        pixel_features[coors[:, 0], coors[:, 1], coors[:, 2]] = frustum_features
        return pixel_features.permute(0, 3, 1, 2).contiguous()

    def _pixel_to_point(
        self, pixel_features: torch.Tensor, coors: torch.Tensor, stride: int
    ) -> torch.Tensor:
        """Sample dense pixel features back to sparse point locations.

        Args:
            pixel_features: Dense pixel feature map.
            coors: Point coordinates.
            stride: Spatial stride used when indexing the pixel grid.

        Returns:
            Point-aligned feature tensor.
        """
        pixel_features = pixel_features.permute(0, 2, 3, 1).contiguous()
        return pixel_features[coors[:, 0], coors[:, 1] // stride, coors[:, 2] // stride]

    def _point_to_frustum(
        self,
        point_features: torch.Tensor,
        voxel_coors: torch.Tensor,
        inverse_map: torch.Tensor,
        stride: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Aggregate sparse point features into frustum voxels.

        Args:
            point_features: Point feature tensor.
            voxel_coors: Voxel coordinates.
            inverse_map: Point-to-voxel index mapping.
            stride: Spatial stride used for the target voxel grid.

        Returns:
            Tuple of downsampled voxel coordinates and aggregated voxel features.
        """
        voxel_features = torch.full(
            (inverse_map.max() + 1, point_features.shape[1]),
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
        return voxel_coors // stride, voxel_features
