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

"""Native BEVFusion detector.

This module contains the high-level BEVFusion detector wrapper and export ABI.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from autoware_ml.metrics.base import MetricSuite
from autoware_ml.metrics.detection3d.eval_output import detection_eval_output
from autoware_ml.models.base import BaseModel
from autoware_ml.models.detection3d.feature_extractors import (
    LidarBEVFeatureExtractor,
    MultiviewImageFeatureExtractor,
)
from autoware_ml.utils.deploy import ExportSpec
from autoware_ml.utils.point_cloud.batching import infer_batch_size_from_voxel_coords


def _runtime_coors_to_voxel_coords(coors: torch.Tensor) -> torch.Tensor:
    """Convert runtime voxel coordinates into the internal layout.

    The deployment runtime (``autoware_bevfusion`` voxelizes with spconv's
    ``Point2Voxel``) provides per-voxel coordinates as ``(z, y, x)`` without a
    batch column; the sparse encoder consumes ``(batch, z, y, x)``.

    Args:
        coors: Runtime voxel coordinates of shape ``(N, 3)`` in ``(z, y, x)``
            order.

    Returns:
        Voxel coordinates of shape ``(N, 4)`` in ``(batch, z, y, x)`` order
        with a zero batch column.
    """
    batch_column = torch.zeros((coors.shape[0], 1), dtype=coors.dtype, device=coors.device)
    return torch.cat((batch_column, coors), dim=1)


def _export_detection_outputs(
    head: nn.Module, outputs: dict[str, torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pack raw head outputs into the runtime detection interface.

    The runtime consumes the raw regression channels and decodes them with
    its own parameters, so no metric-space decoding happens in the graph.

    Args:
        head: TransFusion detection head producing the output dictionary.
        outputs: Raw prediction tensors from the head forward pass.

    Returns:
        Tuple of ``bbox_pred`` with the concatenated regression channels of
        shape ``(10, num_proposals)``, ``score`` of shape ``(num_proposals,)``,
        and ``label_pred`` of shape ``(num_proposals,)``.
    """
    num_proposals = head.num_proposals
    query_labels = outputs["query_labels"]
    heatmap = outputs["heatmap"][..., -num_proposals:].sigmoid()
    one_hot = (
        F.one_hot(query_labels, num_classes=head.num_classes).permute(0, 2, 1).to(heatmap.dtype)
    )
    score = (heatmap * outputs["query_heatmap_score"] * one_hot)[0].max(dim=0).values

    if outputs.get("vel") is None:
        raise ValueError("BEVFusion export requires a velocity branch in the detection head.")
    bbox_pred = torch.cat(
        [outputs[key][0, :, -num_proposals:] for key in ("center", "height", "dim", "rot", "vel")],
        dim=0,
    )
    return bbox_pred, score, query_labels[0]


class _BEVFusionExportWrapper(nn.Module):
    """Wrap the camera-lidar main body export.

    The wrapper exposes the exact tensor-only signature expected by the
    deployment runtime and returns runtime-decodable detection outputs.
    """

    def __init__(self, model: BEVFusionDetectionModel) -> None:
        """Initialize the export wrapper.

        Args:
            model: BEVFusion model instance.
        """
        super().__init__()
        self.model = model

    def forward(
        self,
        voxels: torch.Tensor,
        coors: torch.Tensor,
        num_points_per_voxel: torch.Tensor,
        points: torch.Tensor,
        lidar2image: torch.Tensor,
        img_aug_matrix: torch.Tensor,
        geom_feats: torch.Tensor,
        kept: torch.Tensor,
        ranks: torch.Tensor,
        indices: torch.Tensor,
        image_feats: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run export-time inference with precomputed BEV-pool metadata.

        Args:
            voxels: Lidar voxel tensor.
            coors: Voxel coordinates in ``(z, y, x)`` order without batch column.
            num_points_per_voxel: Number of points per voxel.
            points: Raw point tensor used for lidar depth guidance.
            lidar2image: Raw lidar-to-image projection matrices.
            img_aug_matrix: Image augmentation matrices.
            geom_feats: Precomputed BEV-pool geometry features.
            kept: Keep mask for pooled features.
            ranks: Sorted BEV ranks.
            indices: Sorting indices aligned with ``ranks``.
            image_feats: Precomputed image features.

        Returns:
            Tuple of ``bbox_pred``, ``score``, and ``label_pred``.
        """
        outputs = self.model._forward_export(
            voxels=voxels,
            coors=coors,
            num_points_per_voxel=num_points_per_voxel,
            points=points,
            lidar2image=lidar2image,
            img_aug_matrix=img_aug_matrix,
            geom_feats=geom_feats,
            kept=kept,
            ranks=ranks,
            indices=indices,
            image_feats=image_feats,
        )
        return _export_detection_outputs(self.model.bbox_head, outputs)


class _BEVFusionImageBackboneExportWrapper(nn.Module):
    """Wrap the multiview image backbone export.

    The wrapper consumes raw ``uint8`` images and bakes the training-time
    ``1 / 255`` normalization into the graph.
    """

    def __init__(self, model: BEVFusionDetectionModel) -> None:
        """Initialize the image backbone export wrapper.

        Args:
            model: BEVFusion model instance.
        """
        super().__init__()
        self.model = model

    def forward(self, imgs: torch.Tensor) -> torch.Tensor:
        """Encode raw multiview images into neck features.

        Args:
            imgs: Raw multiview images of shape ``(N, 3, H, W)`` with
                ``uint8`` values.

        Returns:
            Image neck features of shape ``(N, C, fH, fW)``.
        """
        images = imgs.float() / 255.0
        return self.model.get_image_backbone_features(images.unsqueeze(0)).squeeze(0)


class _BEVFusionLidarExportWrapper(nn.Module):
    """Wrap the lidar-only BEVFusion main body export.

    Used when the image branch is disabled. The wrapper exposes the same
    single-sample tensor interface as the camera-lidar main body without the
    image inputs.
    """

    def __init__(self, model: BEVFusionDetectionModel) -> None:
        """Initialize the lidar export wrapper.

        Args:
            model: BEVFusion model instance.
        """
        super().__init__()
        self.model = model

    def forward(
        self,
        voxels: torch.Tensor,
        coors: torch.Tensor,
        num_points_per_voxel: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run export-time inference on lidar voxel inputs.

        Args:
            voxels: Voxel features.
            coors: Voxel coordinates in ``(z, y, x)`` order without batch column.
            num_points_per_voxel: Number of points in each voxel.

        Returns:
            Tuple of ``bbox_pred``, ``score``, and ``label_pred``.
        """
        outputs = self.model._forward_with_batch_size(
            voxels=voxels,
            num_points=num_points_per_voxel,
            voxel_coords=_runtime_coors_to_voxel_coords(coors),
            batch_size=1,
        )
        return _export_detection_outputs(self.model.bbox_head, outputs)


class BEVFusionDetectionModel(BaseModel):
    """Compose a BEVFusion detector with camera and lidar branches.

    The model fuses image and lidar features in BEV space and exposes the
    shared Autoware-ML training, prediction, and export interfaces.
    """

    def __init__(
        self,
        pts_voxel_encoder: nn.Module | None,
        pts_middle_encoder: nn.Module | None,
        pts_backbone: nn.Module | None,
        pts_neck: nn.Module | None,
        bbox_head: nn.Module,
        img_backbone: nn.Module | None = None,
        img_neck: nn.Module | None = None,
        view_transform: nn.Module | None = None,
        fusion_layer: nn.Module | None = None,
        optimizer: Callable[..., Optimizer] | None = None,
        scheduler: Callable[[Optimizer], LRScheduler] | None = None,
        metrics: list[MetricSuite] | None = None,
    ) -> None:
        """Initialize BEVFusion.

        Args:
            pts_voxel_encoder: Lidar voxel encoder.
            pts_middle_encoder: Lidar BEV encoder.
            pts_backbone: Shared BEV backbone.
            pts_neck: Shared BEV neck.
            bbox_head: Detection head.
            img_backbone: Image backbone.
            img_neck: Image neck.
            view_transform: View transform from image features to BEV.
            fusion_layer: BEV fusion layer for multi-branch inputs.
            optimizer: Optimizer factory.
            scheduler: Scheduler factory.
            metrics: Detection metrics accumulated during validation and test.
        """
        super().__init__(optimizer=optimizer, scheduler=scheduler, metrics=metrics)
        self.pts_voxel_encoder = pts_voxel_encoder
        self.pts_middle_encoder = pts_middle_encoder
        self.pts_backbone = pts_backbone
        self.pts_neck = pts_neck
        self.bbox_head = bbox_head
        self.img_backbone = img_backbone
        self.img_neck = img_neck
        self.view_transform = view_transform
        self.fusion_layer = fusion_layer
        self.lidar_feature_extractor = (
            LidarBEVFeatureExtractor(
                pts_voxel_encoder=pts_voxel_encoder,
                pts_middle_encoder=pts_middle_encoder,
                pts_backbone=None,
                pts_neck=None,
            )
            if pts_voxel_encoder is not None and pts_middle_encoder is not None
            else None
        )
        self.image_feature_extractor = (
            MultiviewImageFeatureExtractor(img_backbone=img_backbone, img_neck=img_neck)
            if img_backbone is not None and img_neck is not None
            else None
        )
        self._validate_geometry_contract()

    def _validate_geometry_contract(self) -> None:
        """Validate static geometry contracts between camera and lidar branches.

        Raises:
            ValueError: If configured lidar and image BEV branches do not
                agree on the BEV spatial shape.
        """
        if self.view_transform is None or self.pts_middle_encoder is None:
            return
        if not hasattr(self.view_transform, "expected_bev_shape"):
            return
        if not hasattr(self.pts_middle_encoder, "output_shape"):
            return

        image_bev_shape = tuple(int(value) for value in self.view_transform.expected_bev_shape)
        lidar_bev_shape = tuple(int(value) for value in self.pts_middle_encoder.output_shape)
        if image_bev_shape != lidar_bev_shape:
            raise ValueError(
                "BEVFusion image and lidar branches must share the same BEV shape. "
                f"Got image BEV shape {image_bev_shape} and lidar BEV shape {lidar_bev_shape}."
            )

    def _validate_runtime_bev_shapes(self, bev_features: Sequence[torch.Tensor]) -> None:
        """Validate runtime BEV tensor shapes before multi-branch fusion.

        Args:
            bev_features: Sequence of BEV feature maps to fuse.

        Raises:
            ValueError: If the BEV branches do not share the same spatial shape.
        """
        if len(bev_features) < 2:
            return
        reference_shape = tuple(bev_features[0].shape[-2:])
        for feature in bev_features[1:]:
            feature_shape = tuple(feature.shape[-2:])
            if feature_shape != reference_shape:
                raise ValueError(
                    "BEVFusion branches must share the same runtime BEV shape before fusion. "
                    f"Expected {reference_shape}, got {feature_shape}."
                )

    def _build_lidar_bev(
        self,
        voxels: torch.Tensor,
        num_points: torch.Tensor,
        voxel_coords: torch.Tensor,
        batch_size: int | None = None,
    ) -> torch.Tensor:
        """Encode lidar voxels into a BEV feature map.

        Args:
            voxels: Lidar voxel tensor.
            num_points: Number of points per voxel.
            voxel_coords: Voxel coordinates with batch indices.
            batch_size: Optional explicit batch size.

        Returns:
            Lidar BEV feature map.
        """
        if self.lidar_feature_extractor is None:
            raise ValueError("Lidar branch is not configured.")
        return self.lidar_feature_extractor(voxels, num_points, voxel_coords, batch_size=batch_size)

    def _build_image_bev(
        self,
        img: Sequence[torch.Tensor],
        points: Sequence[torch.Tensor],
        lidar2img: Sequence[torch.Tensor],
        camera_intrinsics: Sequence[torch.Tensor],
        lidar2cam: Sequence[torch.Tensor],
        geom_feats_precomputed: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        | None = None,
        image_feature: torch.Tensor | None = None,
        img_aug_matrix: torch.Tensor | None = None,
        lidar_aug_matrix: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode multiview images into a BEV feature map.

        Args:
            img: Multiview image tensors.
            points: Per-sample lidar points used for depth guidance.
            lidar2img: Lidar-to-image projection matrices. The training
                pipeline bakes image augmentation into these, so the default
                identity ``img_aug_matrix`` keeps the projection consistent.
            camera_intrinsics: Camera intrinsic matrices.
            lidar2cam: Lidar-to-camera extrinsics.
            geom_feats_precomputed: Optional precomputed BEV-pool metadata.
            image_feature: Optional precomputed image feature tensor.
            img_aug_matrix: Optional image augmentation matrices.
            lidar_aug_matrix: Optional lidar augmentation matrices.

        Returns:
            Image BEV feature map.
        """
        if self.image_feature_extractor is None or self.view_transform is None:
            raise ValueError("Image branch is not configured.")
        if image_feature is None:
            image_batch = (
                torch.stack(list(img), dim=0).float()
                if isinstance(img, (list, tuple))
                else img.float()
            )
            if image_batch.dim() != 5:
                raise ValueError(
                    "Expected image batch with shape (B, N, C, H, W), "
                    f"got {tuple(image_batch.shape)}"
                )
            batch_size, num_cams = image_batch.shape[:2]
            image_feature = self.image_feature_extractor(image_batch)
        else:
            batch_size, num_cams = image_feature.shape[:2]

        intrinsics = (
            torch.stack(list(camera_intrinsics), dim=0).float()
            if isinstance(camera_intrinsics, (list, tuple))
            else camera_intrinsics.float()
        )
        lidar2cam_tensor = (
            torch.stack(list(lidar2cam), dim=0).float()
            if isinstance(lidar2cam, (list, tuple))
            else lidar2cam.float()
        )
        lidar2image = (
            torch.stack(list(lidar2img), dim=0).float()
            if isinstance(lidar2img, (list, tuple))
            else lidar2img.float()
        )
        camera2lidar = torch.inverse(lidar2cam_tensor)
        if img_aug_matrix is None:
            img_aug_matrix = (
                torch.eye(4, device=image_feature.device)
                .view(1, 1, 4, 4)
                .repeat(batch_size, num_cams, 1, 1)
            )
        if lidar_aug_matrix is None:
            lidar_aug_matrix = (
                torch.eye(4, device=image_feature.device).view(1, 4, 4).repeat(batch_size, 1, 1)
            )
        return self.view_transform(
            image_feature,
            points,
            lidar2image,
            intrinsics,
            camera2lidar,
            img_aug_matrix,
            lidar_aug_matrix,
            geom_feats_precomputed=geom_feats_precomputed,
        )

    def get_image_backbone_features(self, image_batch: torch.Tensor) -> torch.Tensor:
        """Encode multiview images into neck features expected by the view transform.

        Args:
            image_batch: Image batch with shape ``(B, N, C, H, W)``.

        Returns:
            Neck feature tensor consumed by the view transform.
        """
        if self.image_feature_extractor is None:
            raise ValueError("Image branch is not configured.")
        return self.image_feature_extractor(image_batch)

    def _forward_export(
        self,
        voxels: torch.Tensor,
        coors: torch.Tensor,
        num_points_per_voxel: torch.Tensor,
        points: torch.Tensor,
        lidar2image: torch.Tensor,
        img_aug_matrix: torch.Tensor,
        geom_feats: torch.Tensor,
        kept: torch.Tensor,
        ranks: torch.Tensor,
        indices: torch.Tensor,
        image_feats: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Run the export-time main body with runtime-compatible inputs.

        Args:
            voxels: Lidar voxel tensor.
            coors: Voxel coordinates in ``(z, y, x)`` order without batch column.
            num_points_per_voxel: Number of points per voxel.
            points: Raw point tensor used for lidar depth guidance.
            lidar2image: Raw lidar-to-image projection matrices.
            img_aug_matrix: Image augmentation matrices.
            geom_feats: Precomputed BEV-pool geometry features.
            kept: Keep mask for pooled features.
            ranks: Sorted BEV ranks.
            indices: Sorting indices aligned with ``ranks``.
            image_feats: Precomputed image features.

        Returns:
            Detection head outputs produced by the export path.
        """
        if self.view_transform is None:
            raise ValueError("Image branch is not configured.")
        image_bev = self.view_transform.forward_precomputed(
            image_feats.unsqueeze(0),
            [points],
            lidar2image.unsqueeze(0),
            img_aug_matrix.unsqueeze(0),
            geom_feats.long(),
            kept,
            ranks,
            indices,
        )

        return self._forward_with_batch_size(
            voxels=voxels,
            num_points=num_points_per_voxel,
            voxel_coords=_runtime_coors_to_voxel_coords(coors),
            batch_size=1,
            image_bev=image_bev,
        )

    def _forward_with_batch_size(
        self,
        voxels: torch.Tensor | None = None,
        num_points: torch.Tensor | None = None,
        voxel_coords: torch.Tensor | None = None,
        img: Sequence[torch.Tensor] | None = None,
        points: Sequence[torch.Tensor] | None = None,
        lidar2img: Sequence[torch.Tensor] | None = None,
        camera_intrinsics: Sequence[torch.Tensor] | None = None,
        lidar2cam: Sequence[torch.Tensor] | None = None,
        batch_size: int | None = None,
        image_bev: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        """Run the configured BEV branches and dense head.

        Args:
            voxels: Optional lidar voxel tensor.
            num_points: Optional number of points per voxel.
            voxel_coords: Optional voxel coordinates with batch indices.
            img: Optional multiview image tensors.
            points: Optional per-sample lidar points for depth guidance.
            lidar2img: Optional lidar-to-image projection matrices.
            camera_intrinsics: Optional camera intrinsic matrices.
            lidar2cam: Optional lidar-to-camera extrinsics.
            batch_size: Optional explicit batch size.
            image_bev: Optional precomputed image BEV tensor.
            **kwargs: Unused extra keyword arguments.

        Returns:
            Detection head outputs for the configured branches.
        """
        del kwargs
        bev_features: list[torch.Tensor] = []

        if voxels is not None and num_points is not None and voxel_coords is not None:
            if batch_size is None:
                batch_size = infer_batch_size_from_voxel_coords(voxel_coords)
            bev_features.append(
                self._build_lidar_bev(voxels, num_points, voxel_coords, batch_size=batch_size)
            )

        if image_bev is not None:
            bev_features.append(image_bev)
        elif img is not None and camera_intrinsics is not None and lidar2cam is not None:
            if points is None or lidar2img is None:
                raise ValueError(
                    "BEVFusion image branch requires points and lidar2img for depth guidance."
                )
            bev_features.append(
                self._build_image_bev(img, points, lidar2img, camera_intrinsics, lidar2cam)
            )

        if not bev_features:
            raise ValueError("At least one BEV branch must be provided.")
        self._validate_runtime_bev_shapes(bev_features)

        if len(bev_features) == 1:
            fused = bev_features[0]
        else:
            if self.fusion_layer is None:
                raise ValueError(
                    "Fusion layer must be configured when multiple BEV branches are used."
                )
            fused = self.fusion_layer(bev_features)

        if self.pts_backbone is not None:
            fused = self.pts_backbone(fused)
        if self.pts_neck is not None:
            fused = self.pts_neck(fused)
        return self.bbox_head(fused)

    def forward(
        self,
        voxels: torch.Tensor | None = None,
        num_points: torch.Tensor | None = None,
        voxel_coords: torch.Tensor | None = None,
        img: Sequence[torch.Tensor] | None = None,
        points: Sequence[torch.Tensor] | None = None,
        lidar2img: Sequence[torch.Tensor] | None = None,
        camera_intrinsics: Sequence[torch.Tensor] | None = None,
        lidar2cam: Sequence[torch.Tensor] | None = None,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        """Run the detector on lidar, image, or fused BEV inputs.

        Args:
            voxels: Optional lidar voxel tensor.
            num_points: Optional number of points per voxel.
            voxel_coords: Optional voxel coordinates with batch indices.
            img: Optional multiview image tensors.
            points: Optional per-sample lidar points for depth guidance.
            lidar2img: Optional lidar-to-image projection matrices.
            camera_intrinsics: Optional camera intrinsic matrices.
            lidar2cam: Optional lidar-to-camera extrinsics.
            **kwargs: Additional arguments forwarded to the shared forward path.

        Returns:
            Detection head outputs.
        """
        return self._forward_with_batch_size(
            voxels=voxels,
            num_points=num_points,
            voxel_coords=voxel_coords,
            img=img,
            points=points,
            lidar2img=lidar2img,
            camera_intrinsics=camera_intrinsics,
            lidar2cam=lidar2cam,
            **kwargs,
        )

    def compute_metrics(
        self,
        batch_inputs_dict: dict[str, Any],
        outputs: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Compute BEVFusion training losses.

        Args:
            batch_inputs_dict: Full batch dictionary.
            outputs: Detection head outputs.

        Returns:
            Loss dictionary produced by the detection head.
        """
        return self.bbox_head.loss(
            outputs, batch_inputs_dict["gt_boxes"], batch_inputs_dict["gt_labels"]
        )

    def predict_outputs(
        self, batch_inputs_dict: dict[str, Any], outputs: dict[str, torch.Tensor]
    ) -> Any:
        """Decode predictions for inference.

        Args:
            batch_inputs_dict: Full batch dictionary.
            outputs: Detection head outputs.

        Returns:
            Decoded prediction results.
        """
        del batch_inputs_dict
        return self.bbox_head.predict(outputs)

    def build_eval_output(self, batch: Mapping[str, Any], outputs: Any) -> dict[str, Any]:
        """Decode detections and pair them with ground truth for metrics."""
        return detection_eval_output(self.bbox_head.predict(outputs), batch)

    def get_log_batch_size(self, batch_inputs_dict: dict[str, Any]) -> int | None:
        """Log the sample count for fusion detection batches."""
        if "gt_boxes" in batch_inputs_dict:
            return len(batch_inputs_dict["gt_boxes"])
        if "img" in batch_inputs_dict:
            return len(batch_inputs_dict["img"])
        if "points" in batch_inputs_dict:
            return len(batch_inputs_dict["points"])
        return super().get_log_batch_size(batch_inputs_dict)

    @staticmethod
    def _first_sample_voxel_inputs(
        batch_inputs_dict: dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extract single-sample voxel export inputs in the runtime layout.

        The exported main body is a single-sample graph, so only voxels of
        the first batch sample are kept. Coordinates are converted from the
        internal ``(batch, z, y, x)`` layout to the runtime ``(z, y, x)``
        layout without the batch column.

        Args:
            batch_inputs_dict: Batched model inputs used to derive export tensors.

        Returns:
            Tuple of voxels, runtime-ordered coordinates, and per-voxel point
            counts for the first sample.
        """
        voxel_coords = batch_inputs_dict["voxel_coords"]
        first_sample = voxel_coords[:, 0] == 0
        voxels = batch_inputs_dict["voxels"][first_sample]
        coors = voxel_coords[first_sample][:, 1:].int().contiguous()
        num_points_per_voxel = batch_inputs_dict["num_points"][first_sample].int()
        return voxels, coors, num_points_per_voxel

    def _prepare_export_model(self) -> "BEVFusionDetectionModel":
        """Return an export-ready model copy with exportable submodules.

        Returns:
            Deep copy of the model with the sparse middle encoder and the
            detection head replaced by their ONNX-exportable variants.
        """
        model = deepcopy(self).eval()
        if hasattr(model.pts_middle_encoder, "prepare_for_export"):
            middle_encoder = model.pts_middle_encoder.prepare_for_export()
            model.pts_middle_encoder = middle_encoder
            model.lidar_feature_extractor.pts_middle_encoder = middle_encoder
        if hasattr(model.bbox_head, "prepare_for_export"):
            model.bbox_head = model.bbox_head.prepare_for_export()
        return model

    def build_export_specs(self, batch_inputs_dict: dict[str, Any]) -> dict[str, ExportSpec]:
        """Build the ONNX export specifications for the runtime-compatible ABI.

        Lidar-only models export one ``bevfusion_lidar`` main body. Camera-lidar
        models export the ``bevfusion_image_backbone`` (raw ``uint8`` images to
        neck features) and the ``bevfusion_camera_lidar`` main body consuming
        those features together with precomputed BEV-pool metadata.

        Args:
            batch_inputs_dict: Batched model inputs used to derive export tensors.

        Returns:
            Ordered mapping of module name to export specification.
        """
        voxels, coors, num_points_per_voxel = self._first_sample_voxel_inputs(batch_inputs_dict)
        export_model = self._prepare_export_model()

        if self.view_transform is None:
            return {
                "bevfusion_lidar": ExportSpec(
                    module=_BEVFusionLidarExportWrapper(export_model),
                    args=(voxels, coors, num_points_per_voxel),
                    input_param_names=["voxels", "coors", "num_points_per_voxel"],
                )
            }

        img = torch.stack(batch_inputs_dict["img"], dim=0).float()[:1]
        camera_intrinsics = torch.stack(batch_inputs_dict["camera_intrinsics"], dim=0).float()[:1]
        lidar2cam = torch.stack(batch_inputs_dict["lidar2cam"], dim=0).float()[:1]
        lidar2img = torch.stack(batch_inputs_dict["lidar2img"], dim=0).float()[:1]
        img_aug_matrix = torch.stack(batch_inputs_dict["img_aug_matrix"], dim=0).float()[:1]
        points = batch_inputs_dict["points"][0].float()

        # The pipeline bakes the image augmentation into lidar2img; the
        # runtime provides the raw projection and augmentation separately, so
        # split them back apart for the export sample.
        lidar2image_raw = torch.inverse(img_aug_matrix).matmul(lidar2img)

        imgs_uint8 = (img[0] * 255.0).round().clamp(0.0, 255.0).to(torch.uint8)
        image_feats = self.get_image_backbone_features(img)[0]

        camera2lidar = torch.inverse(lidar2cam)
        identity_aug = (
            torch.eye(4, device=img.device).view(1, 1, 4, 4).repeat(1, img.shape[1], 1, 1)
        )
        geom = self.view_transform.camera_to_lidar_geometry(
            camera2lidar,
            camera_intrinsics,
            torch.eye(4, device=img.device).view(1, 4, 4),
            identity_aug,
        )
        geom_feats, kept, ranks, indices = self.view_transform.bev_pool_aux(geom)

        return {
            "bevfusion_image_backbone": ExportSpec(
                module=_BEVFusionImageBackboneExportWrapper(export_model),
                args=(imgs_uint8,),
                input_param_names=["imgs"],
            ),
            "bevfusion_camera_lidar": ExportSpec(
                module=_BEVFusionExportWrapper(export_model),
                args=(
                    voxels,
                    coors,
                    num_points_per_voxel,
                    points,
                    lidar2image_raw[0],
                    img_aug_matrix[0],
                    geom_feats.float(),
                    kept.bool(),
                    ranks.long(),
                    indices.long(),
                    image_feats,
                ),
                input_param_names=[
                    "voxels",
                    "coors",
                    "num_points_per_voxel",
                    "points",
                    "lidar2image",
                    "img_aug_matrix",
                    "geom_feats",
                    "kept",
                    "ranks",
                    "indices",
                    "image_feats",
                ],
            ),
        }
