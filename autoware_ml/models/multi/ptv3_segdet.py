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

"""PTv3 joint segmentation and detection model.

One shared PTv3 backbone feeds a point-wise segmentation head and a BEV
detection branch.  The same class supports training (forward / compute_metrics)
and ONNX export (build_export_spec).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from typing import Any

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from autoware_ml.losses.segmentation3d.lovasz import LovaszLoss
from autoware_ml.metrics.base import MetricSuite
from autoware_ml.metrics.detection3d.eval_output import detection_eval_output
from autoware_ml.models.detection3d.ptv3 import (
    PTv3BEVEncoder,
    PTv3BEVProjection,
    _PTv3DetHeadExportModule,
)
from autoware_ml.models.segmentation3d.backbones.ptv3 import Block, PointTransformerV3Backbone
from autoware_ml.models.segmentation3d.ptv3_base import (
    PTv3BaseModel,
    _PTv3BackboneExportModule,
    _PTv3SegHeadExportModule,
    _run_ptv3_backbone_export,
    build_point_feature_dynamic_axes,
    build_ptv3_backbone_dynamic_axes,
    build_ptv3_input_dynamic_axes,
    build_serialized_pooling_export_inputs,
)
from autoware_ml.utils.deploy import ExportSpec
from autoware_ml.utils.point_cloud.structures import serialize_point_cloud_batch


class PTv3SegDetModel(PTv3BaseModel):
    """PTv3 joint segmentation and detection model.

    Training: call forward() then compute_metrics().
    Export: call build_export_spec() - requires export_output_names, grid_size,
    and point_cloud_range to be provided at construction time.
    """

    def __init__(
        self,
        backbone: PointTransformerV3Backbone,
        seg3d_head: nn.Module,
        bev_projector: PTv3BEVProjection,
        bev_encoder: PTv3BEVEncoder,
        bbox_head: nn.Module,
        segmentation_num_classes: int,
        segmentation_ignore_index: int = -1,
        segmentation_lovasz_weight: float = 1.0,
        segmentation_loss_weight: float = 1.0,
        detection_loss_weight: float = 1.0,
        export_output_names: Sequence[str] | None = None,
        grid_size: float | None = None,
        point_cloud_range: Sequence[float] | None = None,
        optimizer: Callable[..., Optimizer] | None = None,
        scheduler: Callable[[Optimizer], LRScheduler] | None = None,
        optimizer_group_overrides: Mapping[str, Mapping[str, Any]] | None = None,
        scheduler_config: Mapping[str, Any] | None = None,
        metrics: list[MetricSuite] | None = None,
    ) -> None:
        super().__init__(
            backbone=backbone,
            grid_size=grid_size,
            point_cloud_range=point_cloud_range,
            optimizer=optimizer,
            scheduler=scheduler,
            optimizer_group_overrides=optimizer_group_overrides,
            scheduler_config=scheduler_config,
            metrics=metrics,
        )
        self.seg3d_head = seg3d_head
        self.bev_projector = bev_projector
        self.bev_encoder = bev_encoder
        self.bbox_head = bbox_head
        self.segmentation_num_classes = int(segmentation_num_classes)
        self.segmentation_ignore_index = int(segmentation_ignore_index)
        self.segmentation_loss_weight = float(segmentation_loss_weight)
        self.detection_loss_weight = float(detection_loss_weight)
        self.segmentation_cross_entropy = nn.CrossEntropyLoss(
            ignore_index=self.segmentation_ignore_index
        )
        self.segmentation_lovasz = LovaszLoss(
            ignore_index=self.segmentation_ignore_index,
            loss_weight=segmentation_lovasz_weight,
        )
        self._export_output_names = (
            list(export_output_names) if export_output_names is not None else None
        )

    def build_optimizer_groups(self) -> Mapping[str, Sequence[torch.nn.Parameter]]:
        """Group pretrained and newly initialized joint-task parameters."""
        backbone_block_parameter_ids = {
            id(parameter)
            for module in self.backbone.modules()
            if isinstance(module, Block)
            for parameter in module.parameters()
            if parameter.requires_grad
        }
        backbone_block_params: list[torch.nn.Parameter] = []
        backbone_default_params: list[torch.nn.Parameter] = []
        for parameter in self.backbone.parameters():
            if not parameter.requires_grad:
                continue
            if id(parameter) in backbone_block_parameter_ids:
                backbone_block_params.append(parameter)
            else:
                backbone_default_params.append(parameter)

        seg3d_head_params = [
            parameter for parameter in self.seg3d_head.parameters() if parameter.requires_grad
        ]
        det3d_branch_params = [
            parameter
            for module in (self.bev_projector, self.bev_encoder, self.bbox_head)
            for parameter in module.parameters()
            if parameter.requires_grad
        ]
        return {
            "backbone_default": backbone_default_params,
            "backbone_block": backbone_block_params,
            "seg3d_head": seg3d_head_params,
            "det3d_branch": det3d_branch_params,
        }

    def forward(
        self,
        coord: torch.Tensor,
        feat: torch.Tensor,
        grid_coord: torch.Tensor,
        offset: torch.Tensor,
    ) -> dict[str, Any]:
        """Run one shared PTv3 backbone pass and branch into both heads."""
        point = self.backbone(
            {"coord": coord, "feat": feat, "grid_coord": grid_coord, "offset": offset}
        )
        seg_logits = self.seg3d_head(point.feat)
        bev_features = self.bev_projector(point.feat, point.grid_coord, point.offset)
        bev_features = self.bev_encoder(bev_features)
        det_outputs = self.bbox_head(bev_features)
        return {"seg_logits": seg_logits, "det_outputs": det_outputs}

    def compute_metrics(
        self,
        batch_inputs_dict: Mapping[str, Any],
        outputs: dict[str, Any],
    ) -> dict[str, torch.Tensor]:
        """Compute combined segmentation and detection losses."""
        seg_logits = outputs["seg_logits"]
        det_outputs = outputs["det_outputs"]
        segment = batch_inputs_dict["segment"]

        loss_ce = self.segmentation_cross_entropy(seg_logits, segment)
        loss_lovasz = self.segmentation_lovasz(seg_logits, segment)
        det_metrics = self.bbox_head.loss(
            det_outputs, batch_inputs_dict["gt_boxes"], batch_inputs_dict["gt_labels"]
        )
        seg_loss = loss_ce + loss_lovasz
        weighted_seg_loss = self.segmentation_loss_weight * seg_loss
        weighted_det_loss = self.detection_loss_weight * det_metrics["loss"]
        metrics: dict[str, torch.Tensor] = {
            "seg_loss_ce": loss_ce,
            "seg_loss_lovasz": loss_lovasz,
            "seg_loss": seg_loss,
            "weighted_seg_loss": weighted_seg_loss,
            "weighted_det_loss": weighted_det_loss,
            "loss": weighted_det_loss + weighted_seg_loss,
        }
        metrics.update({f"det_{name}": value for name, value in det_metrics.items()})
        return metrics

    def build_eval_output(
        self, batch: Mapping[str, Any], outputs: dict[str, Any]
    ) -> dict[str, Any]:
        """Produce detection and segmentation eval data for both task metrics."""
        eval_out = detection_eval_output(self.bbox_head.predict(outputs["det_outputs"]), batch)
        eval_out["seg_pred_labels"] = outputs["seg_logits"].argmax(dim=1)
        eval_out["seg_target_labels"] = batch["segment"].long()
        eval_out["seg_coord"] = batch["coord"]
        return eval_out

    def get_export_output_names(self) -> list[str]:
        """Return configured ONNX export output names.

        Returns:
            Output names passed to the export spec.

        Raises:
            ValueError: If export output names were not configured.
        """
        if self._export_output_names is None:
            raise ValueError(
                "export_output_names must be provided at construction time to use export."
            )
        return list(self._export_output_names)

    def build_export_spec(self, batch_inputs_dict: Mapping[str, torch.Tensor]) -> ExportSpec:
        """Build the ONNX export spec for joint PTv3 segmentation+detection."""
        if self.grid_size is None or self.point_cloud_range is None:
            raise ValueError(
                "grid_size and point_cloud_range must be provided at construction time to use "
                "export."
            )
        sparse_shape, serialization_depth = self._compute_export_geometry(batch_inputs_dict)
        point, input_args = serialize_point_cloud_batch(
            batch_inputs_dict, self.EXPORT_ORDER, serialization_depth
        )
        serialized_pooling_inputs, serialized_pooling_input_names = (
            build_serialized_pooling_export_inputs(
                point["grid_coord"],
                point["serialized_code"],
                point["serialized_order"],
                self.backbone.stride,
            )
        )
        export_module = _PTv3SegDetExportModule(
            backbone=self._prepare_backbone_export(),
            seg3d_head=deepcopy(self.seg3d_head).eval(),
            bev_projector=deepcopy(self.bev_projector).eval(),
            bev_encoder=deepcopy(self.bev_encoder).eval(),
            bbox_head=self.bbox_head.prepare_for_export(),
            sparse_shape=sparse_shape,
            serialized_depth=serialization_depth,
            output_names=self.get_export_output_names(),
        )
        export_module.eval()
        export_input_args = (
            input_args[0],
            input_args[1],
            input_args[3],
            *serialized_pooling_inputs,
        )
        input_param_names = [
            "grid_coord",
            "feat",
            "serialized_code",
            *serialized_pooling_input_names,
        ]
        output_names = self.get_export_output_names()
        dynamic_axes = build_ptv3_input_dynamic_axes(input_param_names)
        dynamic_axes.update(
            build_point_feature_dynamic_axes(
                tuple(name for name in output_names if name in {"pred_labels", "pred_probs"})
            )
        )
        return ExportSpec(
            module=export_module,
            args=export_input_args,
            input_param_names=input_param_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            supported_stages=self.EXPORT_SUPPORTED_STAGES,
        )

    def build_export_specs(
        self, batch_inputs_dict: Mapping[str, torch.Tensor]
    ) -> dict[str, ExportSpec]:
        """Build split PTv3 segdet ONNX export specs for backbone, seg head, and det head."""
        if self.grid_size is None or self.point_cloud_range is None:
            raise ValueError(
                "grid_size and point_cloud_range must be provided at construction time to use "
                "export."
            )
        sparse_shape, serialization_depth = self._compute_export_geometry(batch_inputs_dict)
        point, input_args = serialize_point_cloud_batch(
            batch_inputs_dict, self.EXPORT_ORDER, serialization_depth
        )
        serialized_pooling_inputs, serialized_pooling_input_names = (
            build_serialized_pooling_export_inputs(
                point["grid_coord"],
                point["serialized_code"],
                point["serialized_order"],
                self.backbone.stride,
            )
        )
        backbone_input_args = (
            input_args[0],
            input_args[1],
            input_args[3],
            *serialized_pooling_inputs,
        )
        backbone_input_names = [
            "grid_coord",
            "feat",
            "serialized_code",
            *serialized_pooling_input_names,
        ]

        backbone_module = _PTv3BackboneExportModule(
            backbone=self._prepare_backbone_export(),
            sparse_shape=sparse_shape,
            serialized_depth=serialization_depth,
        ).eval()
        with torch.no_grad():
            point_feat, point_grid_coord, point_offset = backbone_module(*backbone_input_args)

        det_output_names = [
            n for n in self.get_export_output_names() if n not in ("pred_labels", "pred_probs")
        ]
        det3d_head_module = _PTv3DetHeadExportModule(
            bev_projector=deepcopy(self.bev_projector).eval(),
            bev_encoder=deepcopy(self.bev_encoder).eval(),
            bbox_head=self.bbox_head.prepare_for_export(),
            output_names=det_output_names,
        ).eval()

        return {
            "backbone": ExportSpec(
                module=backbone_module,
                args=backbone_input_args,
                input_param_names=backbone_input_names,
                output_names=["point_feat", "point_grid_coord", "point_offset"],
                dynamic_axes=build_ptv3_backbone_dynamic_axes(backbone_input_names),
                supported_stages=self.EXPORT_SUPPORTED_STAGES,
            ),
            "seg3d_head": ExportSpec(
                module=_PTv3SegHeadExportModule(deepcopy(self.seg3d_head).eval()),
                args=(point_feat,),
                input_param_names=["point_feat"],
                output_names=["pred_labels", "pred_probs"],
                dynamic_axes=build_point_feature_dynamic_axes(
                    ("point_feat", "pred_labels", "pred_probs")
                ),
                supported_stages=self.EXPORT_SUPPORTED_STAGES,
            ),
            "det3d_head": ExportSpec(
                module=det3d_head_module,
                args=(point_feat, point_grid_coord, point_offset),
                input_param_names=["point_feat", "point_grid_coord", "point_offset"],
                output_names=det_output_names,
                dynamic_axes=build_point_feature_dynamic_axes(("point_feat", "point_grid_coord")),
                supported_stages=self.EXPORT_SUPPORTED_STAGES,
            ),
        }


class _PTv3SegDetExportModule(nn.Module):
    """ONNX-exportable PTv3 segmentation+detection graph with baked sparse shape."""

    def __init__(
        self,
        backbone: PointTransformerV3Backbone,
        seg3d_head: nn.Module,
        bev_projector: PTv3BEVProjection,
        bev_encoder: PTv3BEVEncoder,
        bbox_head: nn.Module,
        sparse_shape: torch.Tensor,
        serialized_depth: torch.Tensor,
        output_names: Sequence[str],
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.seg3d_head = seg3d_head
        self.bev_projector = bev_projector
        self.bev_encoder = bev_encoder
        self.bbox_head = bbox_head
        self.output_names = list(output_names)
        self.register_buffer("_sparse_shape", sparse_shape.to(dtype=torch.long), persistent=False)
        self.register_buffer("_serialized_depth", serialized_depth, persistent=False)

    def forward(
        self,
        grid_coord: torch.Tensor,
        feat: torch.Tensor,
        serialized_code: torch.Tensor,
        *serialized_pooling_inputs: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        """Run the export graph and return outputs in configured order.

        Args:
            grid_coord: Input voxel coordinates.
            feat: Input point or voxel features.
            serialized_code: Serialization codes for the base point set.
            serialized_pooling_inputs: Precomputed pooling metadata tensors.

        Returns:
            Tuple of export tensors ordered according to ``output_names``.
        """
        point_feat, point_grid_coord, point_offset = _run_ptv3_backbone_export(
            self.backbone,
            grid_coord,
            feat,
            self._serialized_depth,
            serialized_code,
            self._sparse_shape,
            *serialized_pooling_inputs,
        )
        seg_logits = self.seg3d_head(point_feat)
        pred_probs = torch.softmax(seg_logits, dim=1)
        pred_labels = pred_probs.argmax(dim=1)

        bev_features = self.bev_projector(point_feat, point_grid_coord, point_offset)
        bev_features = self.bev_encoder(bev_features)
        det_outputs = self.bbox_head(bev_features)

        outputs: dict[str, torch.Tensor] = {
            "pred_labels": pred_labels,
            "pred_probs": pred_probs,
            **det_outputs,
        }
        return tuple(outputs[name] for name in self.output_names)
