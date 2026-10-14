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

"""Camera branch of the native BEVFusion detector."""

from __future__ import annotations


from jaxtyping import Bool, Float32, Int64
import torch
import torch.nn as nn

from autoware_ml.models.detection3d.view_transforms.depth_lss import (
    BEVPoolResult,
    DepthLSSTransform,
)


class BEVFusionCamera(nn.Module):
    """Encode multiview images into the BEV feature map BEVFusion fuses.

    The branch owns the whole camera path: the image backbone and neck that encode the multiview
    images, and the view transform that lifts the resulting features into BEV space. That mirrors
    the lidar branch, which carries its own encoders, backbone and neck, so each modality is
    complete before the two are fused.
    """

    def __init__(
        self,
        img_backbone: nn.Module,
        img_neck: nn.Module,
        view_transform: DepthLSSTransform,
    ) -> None:
        """Initialize the camera branch.

        Args:
            img_backbone: Image backbone.
            img_neck: Image neck applied after the backbone.
            view_transform: View transform lifting image features into BEV.
        """
        super().__init__()
        self.img_backbone = img_backbone
        self.img_neck = img_neck
        self.view_transform = view_transform

    @property
    def expected_bev_shape(self) -> tuple[int, int]:
        """Return the expected ``(height, width)`` of image BEV features."""
        return self.view_transform.expected_bev_shape

    def extract_image_features(
        self, image_batch: Float32[torch.Tensor, "batch_size num_cams 3 height width"]
    ) -> Float32[torch.Tensor, "batch_size num_cams channels feature_height feature_width"]:
        """Encode multiview images into the neck features expected by the view transform.

        Args:
            image_batch: Image batch with shape ``(B, N, C, H, W)``.

        Returns:
            Neck feature tensor consumed by the view transform.
        """
        batch_size, num_cams, channels, image_height, image_width = image_batch.shape
        flat_images = image_batch.view(batch_size * num_cams, channels, image_height, image_width)
        image_features = self.img_backbone(flat_images)
        if isinstance(image_features, torch.Tensor):
            image_features = (image_features,)
        image_features = self.img_neck(image_features)
        primary_feature = (
            image_features[0] if isinstance(image_features, (list, tuple)) else image_features
        )
        _, primary_feature_channels, primary_feature_height, primary_feature_width = (
            primary_feature.shape
        )
        return primary_feature.view(
            batch_size,
            num_cams,
            primary_feature_channels,
            primary_feature_height,
            primary_feature_width,
        )

    def forward(
        self,
        image_batch: Float32[torch.Tensor, "batch_size num_cams 3 height width"],
        depth_maps: Float32[torch.Tensor, "batch_size num_cams height width"],
        camera_intrinsics: Float32[torch.Tensor, "batch_size num_cams 3 3"],
        aug_lidar2cam: Float32[torch.Tensor, "batch_size num_cams 4 4"],
        geom_feats_precomputed: BEVPoolResult | None = None,
        image_features: Float32[
            torch.Tensor, "batch_size num_cams channels feature_height feature_width"
        ]
        | None = None,
        img_aug_matrix: Float32[torch.Tensor, "batch_size num_cams 4 4"] | None = None,
    ) -> Float32[torch.Tensor, "batch_size channels height width"]:
        """Encode multiview images into a BEV feature map.

        Args:
            image_batch: Multiview image tensors.
            depth_maps: Depth maps for each camera.
            camera_intrinsics: Camera intrinsic matrices.
            aug_lidar2cam: Augmented lidar-to-camera extrinsics expressed in the augmented lidar frame the BEV grid
                is built in. The pipeline composes the inverse of the lidar augmentation into
                these, so their inverse maps the camera into the augmented lidar frame.
            geom_feats_precomputed: Optional precomputed BEV-pool metadata.
            image_feature: Optional precomputed image feature tensor.
            img_aug_matrix: Optional image augmentation matrices.

        Returns:
            Image BEV feature map.
        """
        if image_features is None:
            image_features = self.extract_image_features(image_batch)
        batch_size, num_cams = image_features.shape[:2]

        camera2aug_lidar = torch.inverse(aug_lidar2cam)
        if img_aug_matrix is None:
            img_aug_matrix = (
                torch.eye(4, device=image_features.device)
                .view(1, 1, 4, 4)
                .repeat(batch_size, num_cams, 1, 1)
            )
        return self.view_transform(
            image_features=image_features,
            depth_maps=depth_maps,
            camera_intrinsics=camera_intrinsics,
            camera2aug_lidar=camera2aug_lidar,
            img_aug_matrix=img_aug_matrix,
            geom_feats_precomputed=geom_feats_precomputed,
        )

    def forward_export(
        self,
        image_features: Float32[torch.Tensor, "num_cams channels feature_height feature_width"],
        depth_maps: Float32[torch.Tensor, "num_cams 1 height width"],
        geom_feats: Float32[torch.Tensor, "num_frustum_points 4"]
        | Int64[torch.Tensor, "num_frustum_points 4"],
        kept: Bool[torch.Tensor, " num_frustum_points"],
        ranks: Int64[torch.Tensor, " num_kept"],
        indices: Int64[torch.Tensor, " num_kept"],
    ) -> Float32[torch.Tensor, "1 channels height width"]:
        """Encode a single sample into a BEV feature map with precomputed BEV-pool metadata.

        The exported main body is a single-sample graph, so the runtime passes the camera inputs
        without a batch dimension; they are unsqueezed back into the batched layout the view
        transform expects.

        Args:
            image_features: Precomputed image features for each camera.
            depth_maps: Depth maps for each camera.
            geom_feats: Precomputed BEV-pool geometry features; cast to ``long`` here so
                the runtime may pass them as float.
            kept: Keep mask for pooled features.
            ranks: Sorted BEV ranks.
            indices: Sorting indices aligned with ``ranks``.

        Returns:
            Image BEV feature map for the single exported sample.
        """
        return self.view_transform.forward_precomputed(
            image_features=image_features.unsqueeze(0),
            depth_maps=depth_maps.unsqueeze(0),
            geom_feats=geom_feats.long(),
            kept=kept,
            ranks=ranks,
            indices=indices,
        )

    def build_export_geometry(
        self,
        camera_intrinsics: Float32[torch.Tensor, "1 num_cams 4 4"],
        aug_lidar2cam: Float32[torch.Tensor, "1 num_cams 4 4"],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Precompute the BEV-pool metadata baked into the exported graph.

        The runtime resolves the pooling geometry once from the calibration, so the export sample
        uses an identity image augmentation to keep the metadata free of training-time augmentation.

        Args:
            camera_intrinsics: Single-sample camera intrinsic matrices.
            aug_lidar2cam: Single-sample augmented lidar-to-camera extrinsics.

        Returns:
            Tuple of geometry features, keep mask, sorted ranks, and sorting indices.
        """
        device = camera_intrinsics.device
        num_cams = camera_intrinsics.shape[1]
        camera2aug_lidar = torch.inverse(aug_lidar2cam)
        identity_img_aug = torch.eye(4, device=device).view(1, 1, 4, 4).repeat(1, num_cams, 1, 1)
        geom = self.view_transform.camera_to_lidar_geometry(
            camera2aug_lidar,
            camera_intrinsics,
            identity_img_aug,
        )
        return self.view_transform.bev_pool_aux(geom)


class BEVFusionImageBackboneExportWrapper(nn.Module):
    """Wrap the multiview image backbone export.

    The wrapper consumes raw ``uint8`` images and bakes the training-time ``1 / 255``
    normalization into the graph.
    """

    def __init__(self, bevfusion_camera: BEVFusionCamera) -> None:
        """Initialize the image backbone export wrapper.

        Args:
            bevfusion_camera: BEVFusion camera branch instance.
        """
        super().__init__()
        self.bevfusion_camera = bevfusion_camera

    def forward(
        self, imgs: Float32[torch.Tensor, "batch_size num_cameras 3 height width"]
    ) -> Float32[torch.Tensor, "num_cameras channels feature_height feature_width"]:
        """Encode raw multiview images into neck features.

        Args:
            imgs: Raw multiview images of shape ``(N, 3, H, W)`` with ``uint8`` values.

        Returns:
            Image neck features of shape ``(N, C, fH, fW)``.
        """
        images = imgs.float() / 255.0
        return self.bevfusion_camera.extract_image_features(images.unsqueeze(0)).squeeze(0)
