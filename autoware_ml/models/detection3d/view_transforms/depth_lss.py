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

from typing import NamedTuple, Sequence

from jaxtyping import Bool, Float32, Int64
import torch
import torch.nn as nn

from autoware_ml.ops.bev_pool.bev_pool import bev_pool


class BEVPoolResult(NamedTuple):
    """Precomputed BEV pooling metadata produced by `DepthLSSTransform.bev_pool_aux`.

    ``num_points`` is the number of frustum points, ``batch_size * num_cams * depth_bins *
    height * width``. ``num_kept`` is the number of those points that fall inside the BEV
    grid, i.e. ``kept.sum()``.

    Attributes:
        geom_feats: Integer voxel coordinates ``(x_idx, y_idx, z_idx, batch_idx)`` of the kept
            points, sorted by ``ranks``.
        kept: Boolean mask over all frustum points; ``True`` where the point lies inside the
            BEV grid.
        ranks: Sorted flat BEV grid rank of each kept point.
        indices: Permutation that sorts the kept points by rank; apply it to features after
            masking with ``kept``.
    """

    geom_feats: Int64[torch.Tensor, "num_kept 4"]
    kept: Bool[torch.Tensor, " num_frustum_points"]
    ranks: Int64[torch.Tensor, " num_kept"]
    indices: Int64[torch.Tensor, " num_kept"]


def _gen_dx_bx(
    xbound: Sequence[float], ybound: Sequence[float], zbound: Sequence[float]
) -> tuple[Float32[torch.Tensor, " 3"], Float32[torch.Tensor, " 3"], tuple[int, ...]]:
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

    def forward(
        self, bev_features: Float32[torch.Tensor, "batch_size num_channels height width"]
    ) -> Float32[torch.Tensor, "batch_size num_channels height width"]:
        """Downsample BEV features.

        Args:
            bev_features: BEV feature map.

        Returns:
            Downsampled BEV feature map, where height and width are divided by the downsampling factor.
        """
        return self.net(bev_features)


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

    def forward(
        self, lidar_depth_maps: Float32[torch.Tensor, "batch_size num_channels height width"]
    ) -> Float32[torch.Tensor, "batch_size num_channels height width"]:
        """Encode lidar depth maps.

        Args:
            lidar_depth_maps: Lidar depth maps of shape ``(batch_size * num_cameras, 1, H, W)``.

        Returns:
            Depth features of shape ``(batch_size * num_cameras, C, H / stride, W / stride)``.
        """
        return self.net(lidar_depth_maps)


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

    def forward(
        self, fused_features: Float32[torch.Tensor, "batch_size num_channels height width"]
    ) -> Float32[torch.Tensor, "batch_size num_channels height width"]:
        """Predict depth logits and context features.

        Args:
            fused_features: Concatenated feature tensor of shape ``(batch_size * num_cameras, C, H, W)``.

        Returns:
            Tensor of shape ``(batch_size * num_cameras, depth_bins + context, H, W)``.
        """
        return self.net(fused_features)


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

    def _create_frustum(self) -> Float32[torch.Tensor, "depth_bins feature_height feature_width 3"]:
        """Create the frustum grid used for lift-splat projection.

        Returns:
            Frustum grid in image coordinates with depth bins, where depth_bins is the number of
            discrete depth bins. For example, 0.5m resolution from 0m to 50m would yield 100 depth
            bins. 3 represents the (x, y, depth) coordinates in image space.
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
        camera2aug_lidar: Float32[torch.Tensor, "batch_size num_cameras 4 4"],
        camera_intrinsics: Float32[torch.Tensor, "batch_size num_cameras 3 3"],
        img_aug_matrix: Float32[torch.Tensor, "batch_size num_cameras 4 4"],
    ) -> Float32[torch.Tensor, "batch_size num_cameras depth_bins feature_height feature_width 3"]:
        """Project the image frustum into the lidar frame the BEV grid is built in.

        The frustum lives in the coordinates of the (augmented) image the features were
        computed on. It is first undone back to the raw image plane with ``img_aug_matrix``,
        lifted into the camera frame with ``camera_intrinsics``, and moved into the lidar
        frame with ``camera2aug_lidar``. The output frame must be the one the point cloud,
        the boxes, and the BEV grid live in, so the arguments have to be kept consistent.

        Args:
            camera2aug_lidar: Camera-to-lidar extrinsics mapping the camera into the lidar
                frame the BEV grid is built in. During training that is the augmented lidar
                frame, so this must be the inverse of a ``lidar2cam`` into which the pipeline
                already composed the inverse of the lidar augmentation (as
                ``update_lidar_transformation_matrices`` does). No lidar augmentation is
                applied here, so an extrinsic left in the raw lidar frame would put the
                frustum in the wrong frame.
            camera_intrinsics: Intrinsics of the camera, mapping 3D points in the camera frame to
                2D points in the image plane. It takes the raw intrinsics and thus doesn't take
                image augmentation into account. The frustum is in the augmented image plane,
                so it has to be undone back to the raw image plane with ``img_aug_matrix`` before
                applying the intrinsics.
            img_aug_matrix: 4x4 image augmentation mapping the raw image plane described by
                ``camera_intrinsics`` onto the image the features were computed on, with the
                2D rotation in the top-left 2x2 block and the pixel translation in the last
                column, see ``camera_intrinsics``.

        Returns:
            Frustum points expressed in the lidar frame the BEV grid is built in.
        """
        batch_size, num_cams = camera2aug_lidar.shape[:2]

        camera2lidar_rots = camera2aug_lidar[..., :3, :3]
        camera2lidar_trans = camera2aug_lidar[..., :3, 3]
        intrinsic_inverse = torch.inverse(camera_intrinsics[..., :3, :3])
        post_rot_inverse = torch.inverse(img_aug_matrix[..., :3, :3])
        post_trans = img_aug_matrix[..., :3, 3]

        # First transform the frustum points from augmented image coordinates to raw image
        # coordinates by subtracting the translation and applying the inverse rotation.
        # (batch_size, num_cams, depth_bins, feature_height, feature_width, 3) - (batch_size, num_cams, 1, 1, 1, 3)
        # = (batch_size, num_cams, depth_bins, feature_height, feature_width, 3)
        points = self.frustum.to(camera2aug_lidar.device) - post_trans.view(
            batch_size, num_cams, 1, 1, 1, 3
        )
        # Apply the inverse rotation to the points to get them into raw image coordinates.
        # (batch_size, num_cams, 1, 1, 1, 3, 3) @ (batch_size, num_cams, depth_bins, feature_height, feature_width, 3, 1) =
        # (batch_size, num_cams, depth_bins, feature_height, feature_width, 3, 1)
        # It's now in the raw image coordinates.
        points = post_rot_inverse.view(batch_size, num_cams, 1, 1, 1, 3, 3).matmul(
            points.unsqueeze(-1)
        )
        # Multiply x and y by z to do un-normalization, so that the points are in 3D space in raw image coordinates.
        # Concat the z coordinate back to the points, so that we have un-normalized (x, y, z) in raw image coordinates.
        # (batch_size, num_cams, depth_bins, feature_height, feature_width, 3)
        points = torch.cat([points[..., :2, :] * points[..., 2:3, :], points[..., 2:3, :]], dim=-2)

        # Now get the rotation transformation from camera to lidar
        # camera_to_lidar_rotation @ camera_intrinsic_inverse (from camera to image -> image to camera)
        # (batch_size, num_cams, 3, 3) @ (batch_size, num_cams, 3, 3)
        camera_to_lidar = camera2lidar_rots.matmul(intrinsic_inverse)
        # Now apply the camera to lidar transformation to the points.
        # (batch_size, num_cams, 1, 1, 1, 3, 3) @ (batch_size, num_cams, depth_bins, feature_height, feature_width, 3, 1) =
        # (batch_size, num_cams, depth_bins, feature_height, feature_width, 3, 1) -> squeeze(-1) ->
        # (batch_size, num_cams, depth_bins, feature_height, feature_width, 3)
        # The points are now in lidar coordinates.
        points = (
            camera_to_lidar.view(batch_size, num_cams, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)
        )
        # Add the translation from camera to lidar to the points, so that they are now in lidar coordinates.
        # (batch_size, num_cams, depth_bins, feature_height, feature_width, 3) + (batch_size, num_cams, 1, 1, 1, 3) =
        # (batch_size, num_cams, depth_bins, feature_height, feature_width, 3)
        points += camera2lidar_trans.view(batch_size, num_cams, 1, 1, 1, 3)
        return points

    def _get_cam_feats(
        self,
        image_features: Float32[
            torch.Tensor, "batch_size num_cams channels feature_height feature_width"
        ],
        depth_maps: Float32[torch.Tensor, "batch_size num_cams 1 height width"],
    ) -> Float32[
        torch.Tensor, "batch_size num_cams depth_bins feature_height feature_width channels"
    ]:
        """Predict per-depth camera features guided by lidar depth maps.

        Args:
            image_features: Multiview image feature tensor of shape ``(B, N, C, fH, fW)``.
            depth_maps: Sparse lidar depth maps of shape ``(B, N, 1, H, W)``.

        Returns:
            Depth-weighted camera feature tensor.
        """
        batch_size, num_cams, channels, feature_height, feature_width = image_features.shape
        x = image_features.view(batch_size * num_cams, channels, feature_height, feature_width)
        depth_features = self.dtransform(
            depth_maps.view(batch_size * num_cams, *depth_maps.shape[2:])
        )
        x = self.depthnet(torch.cat([depth_features, x], dim=1))
        depth = x[:, : self.depth_bins].softmax(dim=1)
        feats = depth.unsqueeze(1) * x[
            :, self.depth_bins : self.depth_bins + self.out_channels
        ].unsqueeze(2)
        # Reshape the features to have shape
        # (batch_size, num_cams, out_channels, depth_bins, feature_height, feature_width)
        feats = feats.view(
            batch_size, num_cams, self.out_channels, self.depth_bins, feature_height, feature_width
        )
        # Permute the features to have shape
        # (batch_size, num_cams, depth_bins, feature_height, feature_width, out_channels)
        return feats.permute(0, 1, 3, 4, 5, 2)

    def bev_pool_aux(
        self,
        geom_feats: Float32[torch.Tensor, "batch_size num_cams depth_bins height width 3"],
    ) -> BEVPoolResult:
        """Precompute sorted BEV pooling metadata from projected frustum points.

        Args:
            geom_feats: Projected frustum coordinates in lidar space.

        Returns:
            :class:`BEVPoolResult` holding the sorted integer voxel coordinates of the kept
            points, the keep mask over all frustum points, the sorted ranks, and the sorting
            indices.
        """
        batch_size, num_cams, depth_bins, height, width, channels = geom_feats.shape
        geom_feats = ((geom_feats - (self.bx - self.dx / 2.0)) / self.dx).long()  # type: ignore
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
        return BEVPoolResult(geom_feats=geom_feats, kept=kept, ranks=ranks, indices=indices)

    def bev_pool_precomputed(
        self,
        feats: Float32[torch.Tensor, "batch_size num_cams depth_bins height width channels"],
        geom_feats: Int64[torch.Tensor, "num_frustum_points 4"],
        kept: Bool[torch.Tensor, " num_frustum_points"],
        ranks: Int64[torch.Tensor, " num_kept"],
        indices: Int64[torch.Tensor, " num_kept"],
    ) -> Float32[torch.Tensor, "batch_size channels * Z height width"]:
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

    def _bev_pool(
        self,
        feats: Float32[torch.Tensor, "batch_size num_cams depth_bins height width channels"],
        geom_feats: Float32[torch.Tensor, "batch_size num_cams depth_bins height width 3"],
    ) -> Float32[torch.Tensor, "batch_size channels * Z height width"]:
        """Pool camera features into BEV using on-the-fly metadata generation.

        Args:
            feats: Depth-weighted camera features.
            geom_feats: Projected frustum coordinates in lidar space.

        Returns:
            BEV feature map of shape ``(B, C * Z, Y, X)``.
        """
        bev_pool_result = self.bev_pool_aux(geom_feats)
        return self.bev_pool_precomputed(
            feats,
            bev_pool_result.geom_feats,
            bev_pool_result.kept,
            bev_pool_result.ranks,
            bev_pool_result.indices,
        )

    def forward_precomputed(
        self,
        image_features: Float32[
            torch.Tensor, "batch_size num_cams channels feature_height feature_width"
        ],
        depth_maps: Float32[torch.Tensor, "batch_size num_cams 1 height width"],
        geom_feats: Int64[torch.Tensor, "num_frustum_points 4"],
        kept: Bool[torch.Tensor, " num_frustum_points"],
        ranks: Int64[torch.Tensor, " num_kept"],
        indices: Int64[torch.Tensor, " num_kept"],
    ) -> Float32[torch.Tensor, "batch_size channels * Z height width"]:
        """Project multiview image features into BEV using precomputed pooling metadata.

        Args:
            image_features: Multiview image feature tensor.
            depth_maps: Pre-computed depth maps for each camera.
            geom_feats: Filtered geometry features.
            kept: Keep mask produced by :meth:`bev_pool_aux`.
            ranks: Sorted BEV ranks.
            indices: Sorting indices aligned with ``ranks``.

        Returns:
            BEV feature map, where height and width are divided by the downsampling factor.
        """
        feats = self._get_cam_feats(image_features, depth_maps)
        bev = self.bev_pool_precomputed(feats, geom_feats, kept, ranks, indices)
        return self.downsample_net(bev)

    def forward(
        self,
        image_features: Float32[
            torch.Tensor, "batch_size num_cams channels feature_height feature_width"
        ],
        depth_maps: Float32[torch.Tensor, "batch_size num_cams 1 height width"],
        camera_intrinsics: Float32[torch.Tensor, "batch_size num_cams 3 3"],
        camera2aug_lidar: Float32[torch.Tensor, "batch_size 4 4"],
        img_aug_matrix: Float32[torch.Tensor, "batch_size 4 4"],
        geom_feats_precomputed: BEVPoolResult | None = None,
    ) -> Float32[torch.Tensor, "batch_size channels * Z height width"]:
        """Project multiview image features into the BEV plane.

        Args:
            image_features: Multiview image feature tensor.
            depth_maps: Pre-computed depth maps for each camera.
            camera_intrinsics: Intrinsics of the image plane ``img_aug_matrix`` maps
                into, see :meth:`camera_to_lidar_geometry`.
            camera2aug_lidar: Camera-to-lidar extrinsics mapping into the lidar frame
                the BEV grid is built in, i.e. the augmented lidar frame during
                training, see :meth:`camera_to_lidar_geometry`.
            img_aug_matrix: 4x4 image augmentation matrices paired with
                ``camera_intrinsics``. Also applied on top of ``lidar2image`` for the
                depth maps, so it must be the identity when ``lidar2image`` already
                includes the augmentation.
            geom_feats_precomputed: Optional precomputed BEV pooling metadata.

        Returns:
            BEV feature map, where height and width are divided by the downsampling factor.
        """
        feats = self._get_cam_feats(image_features, depth_maps)
        if geom_feats_precomputed is not None:
            bev = self.bev_pool_precomputed(
                feats=feats,
                geom_feats=geom_feats_precomputed.geom_feats,
                kept=geom_feats_precomputed.kept,
                ranks=geom_feats_precomputed.ranks,
                indices=geom_feats_precomputed.indices,
            )
        else:
            geom_feats = self.camera_to_lidar_geometry(
                camera2aug_lidar, camera_intrinsics, img_aug_matrix
            )
            bev = self._bev_pool(feats, geom_feats)
        return self.downsample_net(bev)
