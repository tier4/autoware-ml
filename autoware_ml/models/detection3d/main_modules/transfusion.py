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

"""Native TransFusion lidar detector.

This module contains the high-level TransFusion lidar detector wrapper and export ABI.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from types import MappingProxyType
from typing import Any

from jaxtyping import Float32, Int32
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from autoware_ml.dataclasses.models.detection3d.head_outputs import (
    Detection3DHeadOutputs,
    TransFusionHeadOutputs,
)
from autoware_ml.dataclasses.models.model_batch_inputs import ModelBatchInputs
from autoware_ml.dataclasses.models.model_outputs import ModelOutputs
from autoware_ml.dataclasses.models.model_predictions import ModelPredictions
from autoware_ml.metrics.base import MetricSuite
from autoware_ml.metrics.detection3d.eval_output import detection_eval_output
from autoware_ml.models.detection3d.heads.transfusions.transfusion_head import TransFusionHead
from autoware_ml.models.module_base_model import LogDictConfigs, ModuleBaseModel
from autoware_ml.preprocessing.data_preprocessor import DataPreprocessor
from autoware_ml.utils.deploy import ExportSpec


class _TransFusionExportWrapper(nn.Module):
    """Wrap TransFusion export with the deployment tensor contract.

    The wrapper keeps the exported forward signature tensor-only, fixes the batch size derived
    from the sample batch, and exposes only the deployment tensors expected by Autoware
    consumers.
    """

    def __init__(
        self,
        pts_voxel_encoder: nn.Module,
        pts_middle_encoder: nn.Module,
        pts_backbone: nn.Module,
        pts_neck: nn.Module,
        bbox_head: nn.Module,
        batch_size: int,
    ) -> None:
        """Initialize the export wrapper.

        Args:
            pts_voxel_encoder: Lidar voxel feature encoder.
            pts_middle_encoder: Export-ready sparse or dense middle encoder.
            pts_backbone: BEV backbone.
            pts_neck: BEV neck.
            bbox_head: Export-ready TransFusion head.
            batch_size: Explicit export batch size.
        """
        super().__init__()
        self.pts_voxel_encoder = pts_voxel_encoder
        self.pts_middle_encoder = pts_middle_encoder
        self.pts_backbone = pts_backbone
        self.pts_neck = pts_neck
        self.bbox_head = bbox_head
        self.batch_size = batch_size

    def forward(
        self,
        voxels: Float32[torch.Tensor, "num_voxels max_num_points num_point_features"],
        num_points: Int32[torch.Tensor, " num_voxels"],
        coors: Int32[torch.Tensor, "num_voxels 4"],
    ) -> tuple[
        Float32[torch.Tensor, "batch_size num_classes num_proposals"],
        Float32[torch.Tensor, "batch_size 8 num_proposals"],
        Float32[torch.Tensor, "batch_size 2 num_proposals"],
    ]:
        """Run export-time inference and return deployment tensors.

        Args:
            voxels: Voxel features.
            num_points: Number of points in each voxel.
            coors: Batched voxel coordinates in ``(batch, z, y, x)`` order.

        Returns:
            ``cls_score0``, ``bbox_pred0``, and ``dir_cls_pred0`` tensors.
        """
        voxel_features = self.pts_voxel_encoder(voxels=voxels, num_points=num_points, coords=coors)
        bev_features = self.pts_middle_encoder(
            voxel_features=voxel_features, coords=coors, batch_size=self.batch_size
        )
        bev_features = self.pts_backbone(bev_features)
        bev_features = self.pts_neck(bev_features)
        outputs = self.bbox_head(bev_features)
        return _format_transfusion_export_outputs(outputs)


def _format_transfusion_export_outputs(
    outputs: TransFusionHeadOutputs,
) -> tuple[
    Float32[torch.Tensor, "batch_size num_classes num_proposals"],
    Float32[torch.Tensor, "batch_size 8 num_proposals"],
    Float32[torch.Tensor, "batch_size 2 num_proposals"],
]:
    """Format TransFusion head outputs for deployment export.

    An auxiliary head concatenates every decoder layer along the proposal axis, so only the
    trailing ``num_proposals`` columns belonging to the last layer are exported.

    Args:
        outputs: Raw prediction tensors from the head forward pass.

    Returns:
        ``cls_score0`` as the sigmoid class heatmap weighted by the query heatmap scores,
        ``bbox_pred0`` as the concatenated ``center``, ``height``, ``dim`` and ``vel``
        channels, and ``dir_cls_pred0`` as the ``rot`` channels.

    Raises:
        ValueError: If the head has no velocity branch or the packed channel counts do not
            match the deployment contract.
    """
    separate_head_outputs = outputs.separate_head_outputs
    if separate_head_outputs.vel is None:
        raise ValueError("TransFusion export requires a velocity branch in the detection head.")

    num_proposals = outputs.query_heatmap_score.shape[-1]
    cls_score0 = (
        separate_head_outputs.heatmap[..., -num_proposals:].sigmoid()
        * outputs.query_heatmap_score
    )
    bbox_pred0 = torch.cat(
        (
            separate_head_outputs.center[..., -num_proposals:],
            separate_head_outputs.height[..., -num_proposals:],
            separate_head_outputs.dim[..., -num_proposals:],
            separate_head_outputs.vel[..., -num_proposals:],
        ),
        dim=1,
    )
    dir_cls_pred0 = separate_head_outputs.rot[..., -num_proposals:]

    if bbox_pred0.shape[1] != 8:
        raise ValueError(
            f"TransFusion export expects bbox_pred0 to have 8 channels, got {bbox_pred0.shape[1]}."
        )
    if dir_cls_pred0.shape[1] != 2:
        raise ValueError(
            "TransFusion export expects dir_cls_pred0 to have 2 channels, "
            f"got {dir_cls_pred0.shape[1]}."
        )
    return cls_score0, bbox_pred0, dir_cls_pred0


class TransFusionDetectionModel(ModuleBaseModel):
    """Compose a TransFusion lidar detector from reusable lidar detection modules.

    The wrapper wires together voxel encoding, BEV feature extraction, and the TransFusion
    query head, and exposes the shared Autoware-ML training, prediction, and export
    interfaces.
    """

    def __init__(
        self,
        data_preprocessor: DataPreprocessor,
        # TODO(KokSeang): Encoder and middle_encoder should be standardized to a common interface for all voxel encoders
        pts_voxel_encoder: nn.Module,
        pts_middle_encoder: nn.Module,
        pts_backbone: nn.Module,
        pts_neck: nn.Module,
        # TODO(KokSeang): Consider making the head a generic DetectionHead type instead of TransFusionHead.
        bbox_head: TransFusionHead,
        log_dict_configs: LogDictConfigs,
        optimizer: Callable[..., Optimizer] | None = None,
        scheduler: Callable[[Optimizer], LRScheduler] | None = None,
        metrics: Sequence[MetricSuite] | None = None,
    ) -> None:
        """Initialize TransFusion.

        Args:
            data_preprocessor: Preprocessor for the model inputs.
            pts_voxel_encoder: Lidar voxel feature encoder.
            pts_middle_encoder: Sparse 3D or pillar-scatter middle encoder.
            pts_backbone: BEV backbone.
            pts_neck: BEV neck.
            bbox_head: TransFusion detection head.
            log_dict_configs: Logging configuration for training and validation.
            optimizer: Optimizer factory.
            scheduler: Scheduler factory.
            metrics: Detection metrics accumulated during validation and test.
        """
        super().__init__(
            data_preprocessor=data_preprocessor,
            optimizer=optimizer,
            scheduler=scheduler,
            metrics=metrics,
            log_dict_configs=log_dict_configs,
        )
        self.pts_voxel_encoder = pts_voxel_encoder
        self.pts_middle_encoder = pts_middle_encoder
        self.pts_backbone = pts_backbone
        self.pts_neck = pts_neck
        self.bbox_head = bbox_head

    # TODO(KokSeang): This signature is temporary different from the base class,
    # and will be refactored to match the base class signature once the detection metric is refactored
    # to accept ModelPredictions and MultiTaskFeatures directly.
    def build_eval_output(  # type: ignore[override]
        self, batch: ModelBatchInputs, outputs: ModelOutputs
    ) -> dict[str, Any]:
        """Decode detections and pair them with ground truth for metrics."""
        if outputs.detection3d_head_outputs is None:
            raise ValueError(
                "ModelOutputs must contain detection3d_head_outputs for TransFusion build_eval_output pass."
            )

        return detection_eval_output(
            predictions=self.bbox_head.decode_outputs(outputs.detection3d_head_outputs),
            batch_inputs=batch,
        )

    def _forward_with_batch_size(
        self,
        multi_task_batch_inputs: ModelBatchInputs,
        batch_size: int | None = None,
    ) -> TransFusionHeadOutputs:
        """Run the lidar branch and the TransFusion head.

        Args:
            multi_task_batch_inputs: ModelBatchInputs containing the voxelized lidar inputs.
            batch_size: Optional explicit batch size. Inferred from the ground truth batch when
                omitted.

        Returns:
            TransFusion head outputs.

        Raises:
            ValueError: If the batch carries no voxel data.
        """
        voxels_data = multi_task_batch_inputs.voxels_data
        if voxels_data is None:
            raise ValueError(
                "ModelBatchInputs must contain voxels_data for TransFusion forward pass."
            )

        if batch_size is None:
            batch_size = multi_task_batch_inputs.multi_task_gt_batch.infer_batch_size()
        assert batch_size is not None, "Batch size must be provided for lidar forward pass."

        batch_coords = voxels_data.batch_zyx_coords()
        voxel_features = self.pts_voxel_encoder(
            voxels=voxels_data.voxels,
            num_points=voxels_data.num_points,
            coords=batch_coords,
        )
        bev_features = self.pts_middle_encoder(
            voxel_features=voxel_features, coords=batch_coords, batch_size=batch_size
        )
        bev_features = self.pts_backbone(bev_features)
        bev_features = self.pts_neck(bev_features)
        return self.bbox_head(bev_features)

    def forward(self, multi_task_batch_inputs: ModelBatchInputs) -> ModelOutputs:
        """Run the detector on voxelized lidar inputs.

        Args:
            multi_task_batch_inputs: ModelBatchInputs containing the voxelized lidar inputs.

        Returns:
            Detection head outputs.
        """
        detection_head_outputs = self._forward_with_batch_size(multi_task_batch_inputs)
        return ModelOutputs(
            detection3d_head_outputs=Detection3DHeadOutputs(
                center_head_outputs=None, transfusion_head_outputs=detection_head_outputs
            )
        )

    def compute_metrics(
        self, multi_task_batch_inputs: ModelBatchInputs, multi_task_outputs: ModelOutputs
    ) -> MappingProxyType[str, Float32[torch.Tensor, " num_losses"]]:
        """Compute TransFusionHead training losses."""
        if multi_task_batch_inputs.multi_task_gt_batch.detection3d_gt_batch is None:
            raise ValueError(
                "ModelBatchInputs must contain detection3d_gt_batch for TransFusion compute_metrics pass."
            )

        if multi_task_outputs.detection3d_head_outputs is None:
            raise ValueError(
                "ModelOutputs must contain detection3d_head_outputs for TransFusion compute_metrics pass."
            )

        detection3d_gt_batch = multi_task_batch_inputs.multi_task_gt_batch.detection3d_gt_batch
        return self.bbox_head.loss(
            outputs=multi_task_outputs.detection3d_head_outputs,
            gt_bboxes_3d=detection3d_gt_batch.gt_bboxes_3d,
            gt_labels_3d=detection3d_gt_batch.gt_labels_3d,
            gt_valid_bboxes=detection3d_gt_batch.gt_valid_bboxes,
            gt_traffic_cone_barrier_bbox_status=(
                detection3d_gt_batch.gt_traffic_cone_barrier_bbox_status
            ),
        )  # type: ignore[return-value]

    def decode_outputs(
        self, multi_task_batch_inputs: ModelBatchInputs, outputs: ModelOutputs
    ) -> ModelPredictions:
        """Decode predictions for inference."""
        if outputs.detection3d_head_outputs is None:
            raise ValueError(
                "ModelOutputs must contain detection3d_head_outputs for TransFusion decode_outputs pass."
            )

        return self.bbox_head.decode_outputs(outputs=outputs.detection3d_head_outputs)

    def build_export_spec(self, multi_task_batch_inputs: ModelBatchInputs) -> ExportSpec:
        """Build an export specification with explicit tensor inputs.

        The export modules are export-ready copies of the middle encoder and the head, so the
        training model is left untouched.

        Args:
            multi_task_batch_inputs: Preprocessed example batch used to derive export inputs.

        Returns:
            Export specification for ONNX and TensorRT deployment.

        Raises:
            ValueError: If the batch carries no voxel data.
        """
        voxels_data = multi_task_batch_inputs.voxels_data
        if voxels_data is None:
            raise ValueError(
                "ModelBatchInputs must contain voxels_data to build TransFusion export spec."
            )

        # The batch size is derived from the voxels rather than from the ground truth: a batch
        # prepared for deployment may carry no ground truth at all.
        batch_size = (
            int(voxels_data.batch_indices.max().item()) + 1
            if voxels_data.batch_indices.numel()
            else 1
        )
        pts_middle_encoder = self.pts_middle_encoder
        if hasattr(pts_middle_encoder, "prepare_for_export"):
            pts_middle_encoder = pts_middle_encoder.prepare_for_export()
        return ExportSpec(
            module=_TransFusionExportWrapper(
                pts_voxel_encoder=self.pts_voxel_encoder,
                pts_middle_encoder=pts_middle_encoder,
                pts_backbone=self.pts_backbone,
                pts_neck=self.pts_neck,
                bbox_head=self.bbox_head.prepare_for_export(),
                batch_size=batch_size,
            ),
            args=(
                voxels_data.voxels,
                voxels_data.num_points,
                voxels_data.batch_zyx_coords(),
            ),
            input_param_names=["voxels", "num_points", "coors"],
            output_names=["cls_score0", "bbox_pred0", "dir_cls_pred0"],
        )
