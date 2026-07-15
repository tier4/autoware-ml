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

"""Image-to-BEV view transforms for detection3d models.

This module contains lift-splat view transforms used by camera-based 3D
detectors and fusion models.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn

from autoware_ml.ops.bev_pool.bev_pool import bev_pool


def _gen_dx_bx(
    xbound: Sequence[float], ybound: Sequence[float], zbound: Sequence[float]
) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int, int]]:
    """Derive voxel sizes, voxel origins, and grid shape from bounds.

    Args:
        xbound: X-axis bounds in ``[min, max, step]`` format.
        ybound: Y-axis bounds in ``[min, max, step]`` format.
        zbound: Z-axis bounds in ``[min, max, step]`` format.

    Returns:
        Tuple of voxel size, voxel origin, and grid shape.
    """
    dx = torch.tensor([row[2] for row in (xbound, ybound, zbound)], dtype=torch.float32)
    bx = torch.tensor(
        [row[0] + row[2] / 2.0 for row in (xbound, ybound, zbound)], dtype=torch.float32
    )
    nx = tuple(int((row[1] - row[0]) / row[2]) for row in (xbound, ybound, zbound))
    return dx, bx, nx


class DownSampleNet(nn.Module):
    """Downsample camera BEV features after frustum pooling.

    The module reduces the BEV feature resolution produced by the view
    transform before fusion with lidar features.
    """

    def __init__(self, downsample: int, channels: int) -> None:
        """Initialize the BEV downsampling network.

        Args:
            downsample: Spatial downsampling factor.
            channels: Feature channel count.
        """
        super().__init__()
        if downsample == 1:
            self.net = nn.Identity()
        elif downsample == 2:
            self.net = nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True),
            )
        else:
            raise ValueError(f"Unsupported downsample factor: {downsample}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Downsample BEV features.

        Args:
            x: BEV feature map.

        Returns:
            Downsampled BEV feature map.
        """
        return self.net(x)


class LidarDepthImageNet(nn.Module):
    """Encode sparse lidar depth maps into depth features.

    The network strides the image-resolution depth map down to the image
    feature resolution so it can be concatenated with camera features.
    """

    def __init__(self, in_channels: int = 1, out_channels: int = 64, last_stride: int = 2) -> None:
        """Initialize the lidar depth-map encoder.

        Args:
            in_channels: Depth-map channel count.
            out_channels: Output feature channel count.
            last_stride: Stride of the final convolution. The total stride is
                ``4 * last_stride`` and must match the image-to-feature ratio.
        """
        super().__init__()
        self.out_channels = out_channels
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 32, kernel_size=5, stride=4, padding=2, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, out_channels, kernel_size=5, stride=last_stride, padding=2, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode lidar depth maps.

        Args:
            x: Depth maps of shape ``(B * N, 1, H, W)``.

        Returns:
            Depth features of shape ``(B * N, C, H / stride, W / stride)``.
        """
        return self.net(x)


class DepthLSSNet(nn.Module):
    """Fuse camera and lidar depth features into depth logits and context.

    The network consumes concatenated camera features and lidar depth
    features and predicts the per-pixel depth distribution plus the context
    features lifted into the frustum.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        """Initialize the depth fusion network.

        Args:
            in_channels: Concatenated camera and depth feature channels.
            out_channels: Output channels (depth bins plus context channels).
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict depth logits and context features.

        Args:
            x: Concatenated feature tensor of shape ``(B * N, C, H, W)``.

        Returns:
            Tensor of shape ``(B * N, depth_bins + context, H, W)``.
        """
        return self.net(x)


class DepthLSSTransform(nn.Module):
    """Implement a Lift-Splat-Shoot view transform with lidar-guided depth.

    The module projects lidar points onto each camera to build sparse depth
    maps, fuses them with image features to predict the depth distribution,
    lifts the features into frustum space, pools them into BEV, and exposes
    export helpers for BEVFusion deployment.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        image_size: Sequence[int],
        feature_size: Sequence[int],
        xbound: Sequence[float],
        ybound: Sequence[float],
        zbound: Sequence[float],
        dbound: Sequence[float],
        downsample: int = 1,
        lidar_depth_channels: int = 64,
    ) -> None:
        """Initialize the Depth-LSS view transform.

        Args:
            in_channels: Input image feature channels.
            out_channels: Output BEV feature channels.
            image_size: Input image size.
            feature_size: Backbone feature-map size.
            xbound: X-axis bounds in ``[min, max, step]`` format.
            ybound: Y-axis bounds in ``[min, max, step]`` format.
            zbound: Z-axis bounds in ``[min, max, step]`` format.
            dbound: Depth bounds in ``[min, max, step]`` format.
            downsample: Output BEV downsampling factor.
            lidar_depth_channels: Output channels of the lidar depth encoder.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.image_size = tuple(image_size)
        self.feature_size = tuple(feature_size)
        self.xbound = tuple(xbound)
        self.ybound = tuple(ybound)
        self.zbound = tuple(zbound)
        self.dbound = tuple(dbound)
        self.downsample = downsample

        dx, bx, nx = _gen_dx_bx(self.xbound, self.ybound, self.zbound)
        self.register_buffer("dx", dx, persistent=False)
        self.register_buffer("bx", bx, persistent=False)
        self.nx = nx

        self.frustum = nn.Parameter(self._create_frustum(), requires_grad=False)
        self.depth_bins = self.frustum.shape[0]

        stride_height = self.image_size[0] // self.feature_size[0]
        stride_width = self.image_size[1] // self.feature_size[1]
        if stride_height != stride_width or stride_height % 4 != 0:
            raise ValueError(
                "DepthLSSTransform requires a uniform image-to-feature stride divisible by 4, "
                f"got image_size={self.image_size} and feature_size={self.feature_size}."
            )
        self.dtransform = LidarDepthImageNet(
            in_channels=1, out_channels=lidar_depth_channels, last_stride=stride_height // 4
        )
        self.depthnet = DepthLSSNet(
            in_channels + lidar_depth_channels, self.depth_bins + out_channels
        )
        self.downsample_net = DownSampleNet(downsample, out_channels)

    @property
    def expected_bev_shape(self) -> tuple[int, int]:
        """Return the expected ``(height, width)`` of image BEV features."""
        height = self.nx[1] // self.downsample
        width = self.nx[0] // self.downsample
        return height, width

    def _create_frustum(self) -> torch.Tensor:
        """Create the frustum grid used for lift-splat projection.

        Returns:
            Frustum grid in image coordinates with depth bins.
        """
        image_height, image_width = self.image_size
        feature_height, feature_width = self.feature_size

        depths = (
            torch.arange(*self.dbound, dtype=torch.float32)
            .view(-1, 1, 1)
            .expand(-1, feature_height, feature_width)
        )
        grid_y = torch.linspace(0, image_height - 1, feature_height, dtype=torch.float32).view(
            1, feature_height, 1
        )
        grid_y = grid_y.expand(depths.shape[0], feature_height, feature_width)
        grid_x = torch.linspace(0, image_width - 1, feature_width, dtype=torch.float32).view(
            1, 1, feature_width
        )
        grid_x = grid_x.expand(depths.shape[0], feature_height, feature_width)
        return torch.stack((grid_x, grid_y, depths), dim=-1)

    def camera_to_lidar_geometry(
        self,
        camera2lidar: torch.Tensor,
        camera_intrinsics: torch.Tensor,
        lidar_aug_matrix: torch.Tensor,
        img_aug_matrix: torch.Tensor,
    ) -> torch.Tensor:
        """Project image frustum points into lidar coordinates.

        Args:
            camera2lidar: Camera-to-lidar extrinsics.
            camera_intrinsics: Camera intrinsic matrices.
            lidar_aug_matrix: Lidar augmentation matrices.
            img_aug_matrix: Image augmentation matrices.

        Returns:
            Frustum points expressed in lidar coordinates.
        """
        batch_size, num_cams = camera2lidar.shape[:2]

        camera2lidar_rots = camera2lidar[..., :3, :3]
        camera2lidar_trans = camera2lidar[..., :3, 3]
        intrinsic_inverse = torch.inverse(camera_intrinsics[..., :3, :3])
        post_rot_inverse = torch.inverse(img_aug_matrix[..., :3, :3])
        post_trans = img_aug_matrix[..., :3, 3]

        points = self.frustum.to(camera2lidar.device) - post_trans.view(
            batch_size, num_cams, 1, 1, 1, 3
        )
        points = post_rot_inverse.view(batch_size, num_cams, 1, 1, 1, 3, 3).matmul(
            points.unsqueeze(-1)
        )
        points = torch.cat([points[..., :2, :] * points[..., 2:3, :], points[..., 2:3, :]], dim=-2)

        camera_to_lidar = camera2lidar_rots.matmul(intrinsic_inverse)
        points = (
            camera_to_lidar.view(batch_size, num_cams, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)
        )
        points += camera2lidar_trans.view(batch_size, num_cams, 1, 1, 1, 3)

        extra_rots = lidar_aug_matrix[..., :3, :3]
        extra_trans = lidar_aug_matrix[..., :3, 3]
        points = (
            extra_rots.view(batch_size, 1, 1, 1, 1, 3, 3)
            .repeat(1, num_cams, 1, 1, 1, 1, 1)
            .matmul(points.unsqueeze(-1))
            .squeeze(-1)
        )
        points += extra_trans.view(batch_size, 1, 1, 1, 1, 3).repeat(1, num_cams, 1, 1, 1, 1)
        return points

    def _get_cam_feats(self, x: torch.Tensor, depth_maps: torch.Tensor) -> torch.Tensor:
        """Predict per-depth camera features guided by lidar depth maps.

        Args:
            x: Multiview image feature tensor of shape ``(B, N, C, fH, fW)``.
            depth_maps: Sparse lidar depth maps of shape ``(B, N, 1, H, W)``.

        Returns:
            Depth-weighted camera feature tensor.
        """
        batch_size, num_cams, channels, feature_height, feature_width = x.shape
        x = x.view(batch_size * num_cams, channels, feature_height, feature_width)
        depth_features = self.dtransform(
            depth_maps.view(batch_size * num_cams, *depth_maps.shape[2:])
        )
        x = self.depthnet(torch.cat([depth_features, x], dim=1))
        depth = x[:, : self.depth_bins].softmax(dim=1)
        feats = depth.unsqueeze(1) * x[
            :, self.depth_bins : self.depth_bins + self.out_channels
        ].unsqueeze(2)
        feats = feats.view(
            batch_size, num_cams, self.out_channels, self.depth_bins, feature_height, feature_width
        )
        return feats.permute(0, 1, 3, 4, 5, 2)

    def build_depth_maps(
        self,
        points: Sequence[torch.Tensor],
        lidar2image: torch.Tensor,
        img_aug_matrix: torch.Tensor,
    ) -> torch.Tensor:
        """Project lidar points onto each camera into sparse depth maps.

        Both matrices must map into the same image plane the depth map is
        built for: when ``lidar2image`` already contains the image
        augmentation (training pipeline), pass an identity ``img_aug_matrix``;
        at deployment the runtime provides the raw projection and the
        augmentation separately.

        The per-camera scatter is vectorized so the traced graph supports a
        dynamic number of cameras; duplicate pixel hits resolve to an
        arbitrary point among the duplicates.

        Args:
            points: Per-sample lidar points with at least XYZ columns.
            lidar2image: Lidar-to-image projections of shape ``(B, N, 4, 4)``.
            img_aug_matrix: Image augmentation matrices of shape ``(B, N, 4, 4)``.

        Returns:
            Sparse depth maps of shape ``(B, N, 1, H, W)``.
        """
        batch_size = len(points)
        num_cams = lidar2image.shape[1]
        height, width = self.image_size
        depth_maps = lidar2image.new_zeros(batch_size, num_cams, 1, height, width)

        for batch_index in range(batch_size):
            coords = points[batch_index][:, :3].transpose(0, 1)  # (3, P)
            projected = lidar2image[batch_index][:, :3, :3].matmul(coords)
            projected = projected + lidar2image[batch_index][:, :3, 3].reshape(-1, 3, 1)

            distances = projected[:, 2, :]
            valid_distance = distances > 0

            projected = torch.cat(
                [
                    projected[:, :2, :] / torch.clamp(projected[:, 2:3, :], 1e-5, 1e5),
                    projected[:, 2:3, :],
                ],
                dim=1,
            )
            projected = img_aug_matrix[batch_index][:, :3, :3].matmul(projected)
            projected = projected + img_aug_matrix[batch_index][:, :3, 3].reshape(-1, 3, 1)
            pixel_coords = projected[:, :2, :].transpose(1, 2)[..., [1, 0]]  # (N, P, [y, x])

            on_image = (
                (pixel_coords[..., 0] >= 0)
                & (pixel_coords[..., 0] < height)
                & (pixel_coords[..., 1] >= 0)
                & (pixel_coords[..., 1] < width)
                & valid_distance
            )
            hits = torch.nonzero(on_image, as_tuple=False)
            camera_indices = hits[:, 0]
            point_indices = hits[:, 1]
            hit_coords = pixel_coords[camera_indices, point_indices].long()
            hit_distances = distances[camera_indices, point_indices]

            flat_indices = (
                camera_indices * height * width + hit_coords[:, 0] * width + hit_coords[:, 1]
            )
            flat_depth = torch.zeros(
                num_cams * height * width, device=depth_maps.device, dtype=depth_maps.dtype
            )
            flat_depth.scatter_(dim=0, index=flat_indices, src=hit_distances)
            depth_maps[batch_index] = flat_depth.view(num_cams, 1, height, width)

        return depth_maps

    def bev_pool_aux(
        self, geom_feats: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Precompute sorted BEV pooling metadata from projected frustum points.

        Args:
            geom_feats: Projected frustum coordinates in lidar space.

        Returns:
            Tuple of filtered geometry features, keep mask, ranks, and sorting indices.
        """
        batch_size, num_cams, depth_bins, height, width, channels = geom_feats.shape
        geom_feats = ((geom_feats - (self.bx - self.dx / 2.0)) / self.dx).long()
        geom_feats = geom_feats.view(batch_size * num_cams * depth_bins * height * width, channels)
        batch_indices = torch.cat(
            [
                torch.full(
                    (geom_feats.shape[0] // batch_size, 1),
                    batch_index,
                    device=geom_feats.device,
                    dtype=torch.long,
                )
                for batch_index in range(batch_size)
            ],
            dim=0,
        )
        geom_feats = torch.cat((geom_feats, batch_indices), dim=1)

        kept = (
            (geom_feats[:, 0] >= 0)
            & (geom_feats[:, 0] < self.nx[0])
            & (geom_feats[:, 1] >= 0)
            & (geom_feats[:, 1] < self.nx[1])
            & (geom_feats[:, 2] >= 0)
            & (geom_feats[:, 2] < self.nx[2])
        )
        geom_feats = geom_feats[kept]

        ranks = (
            geom_feats[:, 0] * (self.nx[1] * self.nx[2] * batch_size)
            + geom_feats[:, 1] * (self.nx[2] * batch_size)
            + geom_feats[:, 2] * batch_size
            + geom_feats[:, 3]
        )
        indices = ranks.argsort()
        ranks = ranks[indices]
        geom_feats = geom_feats[indices]
        return geom_feats, kept, ranks, indices

    def bev_pool_precomputed(
        self,
        feats: torch.Tensor,
        geom_feats: torch.Tensor,
        kept: torch.Tensor,
        ranks: torch.Tensor,
        indices: torch.Tensor,
    ) -> torch.Tensor:
        """Pool camera features into BEV using precomputed geometry metadata.

        Args:
            feats: Depth-weighted camera features.
            geom_feats: Filtered geometry features.
            kept: Keep mask produced by :meth:`bev_pool_aux`.
            ranks: Sorted BEV ranks.
            indices: Sorting indices aligned with ``ranks``.

        Returns:
            BEV feature map of shape ``(B, C * Z, Y, X)``.
        """
        batch_size, num_cams, depth_bins, height, width, channels = feats.shape
        feats = feats.reshape(batch_size * num_cams * depth_bins * height * width, channels)
        feats = feats[kept]
        feats = feats[indices]
        bev = bev_pool(
            feats, geom_feats, ranks, batch_size, self.nx[2], self.nx[0], self.nx[1], self.training
        )
        # The pooling metadata is x-major (geometry column 0 is the X index),
        # so the pooled grid comes out as (X, Y); transpose to the (Y, X) BEV
        # layout shared with the lidar branch and the detection head.
        return torch.cat(bev.unbind(dim=2), dim=1).transpose(-2, -1).contiguous()

    def _bev_pool(self, feats: torch.Tensor, geom_feats: torch.Tensor) -> torch.Tensor:
        """Pool camera features into BEV using on-the-fly metadata generation.

        Args:
            feats: Depth-weighted camera features.
            geom_feats: Projected frustum coordinates in lidar space.

        Returns:
            BEV feature map of shape ``(B, C * Z, Y, X)``.
        """
        pooled_geom_feats, kept, ranks, indices = self.bev_pool_aux(geom_feats)
        return self.bev_pool_precomputed(feats, pooled_geom_feats, kept, ranks, indices)

    def forward_precomputed(
        self,
        x: torch.Tensor,
        points: Sequence[torch.Tensor],
        lidar2image: torch.Tensor,
        img_aug_matrix: torch.Tensor,
        geom_feats: torch.Tensor,
        kept: torch.Tensor,
        ranks: torch.Tensor,
        indices: torch.Tensor,
    ) -> torch.Tensor:
        """Project multiview image features into BEV using precomputed pooling metadata.

        Args:
            x: Multiview image feature tensor.
            points: Per-sample lidar points used for depth guidance.
            lidar2image: Lidar-to-image projection matrices.
            img_aug_matrix: Image augmentation matrices.
            geom_feats: Filtered geometry features.
            kept: Keep mask produced by :meth:`bev_pool_aux`.
            ranks: Sorted BEV ranks.
            indices: Sorting indices aligned with ``ranks``.

        Returns:
            BEV feature map.
        """
        depth_maps = self.build_depth_maps(points, lidar2image, img_aug_matrix)
        bev = self.bev_pool_precomputed(
            self._get_cam_feats(x, depth_maps), geom_feats, kept, ranks, indices
        )
        return self.downsample_net(bev)

    def forward(
        self,
        x: torch.Tensor,
        points: Sequence[torch.Tensor],
        lidar2image: torch.Tensor,
        camera_intrinsics: torch.Tensor,
        camera2lidar: torch.Tensor,
        img_aug_matrix: torch.Tensor,
        lidar_aug_matrix: torch.Tensor,
        geom_feats_precomputed: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        | None = None,
    ) -> torch.Tensor:
        """Project multiview image features into the BEV plane.

        Args:
            x: Multiview image feature tensor.
            points: Per-sample lidar points used for depth guidance.
            lidar2image: Lidar-to-image projection matrices. Must map into the
                same image plane as ``x`` (image augmentation included when the
                pipeline bakes it into the calibration).
            camera_intrinsics: Camera intrinsic matrices.
            camera2lidar: Camera-to-lidar extrinsics.
            img_aug_matrix: Image augmentation matrices.
            lidar_aug_matrix: Lidar augmentation matrices.
            geom_feats_precomputed: Optional precomputed BEV pooling metadata.

        Returns:
            BEV feature map.
        """
        depth_maps = self.build_depth_maps(points, lidar2image, img_aug_matrix)
        feats = self._get_cam_feats(x, depth_maps)
        if geom_feats_precomputed is not None:
            geom_feats, kept, ranks, indices = geom_feats_precomputed
            bev = self.bev_pool_precomputed(feats, geom_feats, kept, ranks, indices)
        else:
            geom_feats = self.camera_to_lidar_geometry(
                camera2lidar, camera_intrinsics, lidar_aug_matrix, img_aug_matrix
            )
            bev = self._bev_pool(feats, geom_feats)
        return self.downsample_net(bev)
