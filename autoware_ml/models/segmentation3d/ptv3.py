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

"""PTv3 segmentation model wrapper.

This module contains the high-level PTv3 Lightning wrapper and export logic.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from typing import Any

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LRScheduler

from autoware_ml.losses.segmentation3d.lovasz import LovaszLoss
from autoware_ml.metrics.segmentation3d import compute_segmentation_metrics
from autoware_ml.models.segmentation3d.backbones.ptv3 import Block, PointTransformerV3Backbone
from autoware_ml.models.segmentation3d.ptv3_base import (
    PTv3BaseModel,
    _PTv3BackboneExportModule,
    _PTv3SegHeadExportModule,
    build_point_feature_dynamic_axes,
    build_ptv3_backbone_dynamic_axes,
    build_ptv3_input_dynamic_axes,
    build_serialized_pooling_export_inputs,
    _run_ptv3_backbone_export,
)
from autoware_ml.utils.deploy import ExportSpec
from autoware_ml.utils.point_cloud.structures import serialize_point_cloud_batch


class _PTv3ExportModule(nn.Module):
    """Expose a deployment-oriented PTv3 export graph without mutating the model."""

    def __init__(
        self,
        backbone: PointTransformerV3Backbone,
        seg3d_head: nn.Module,
        sparse_shape: torch.Tensor,
        serialized_depth: torch.Tensor,
    ) -> None:
        """Initialize the isolated PTv3 export module.

        Args:
            backbone: Export-prepared PTv3 backbone copy.
            seg3d_head: Segmentation head copy.
            sparse_shape: Static sparse shape used by exported sparse ops.
            serialized_depth: Serialization depth baked at export time.
        """
        super().__init__()
        self.backbone = backbone
        self.seg3d_head = seg3d_head
        self.register_buffer("_sparse_shape", sparse_shape.to(dtype=torch.long), persistent=False)
        self.register_buffer("_serialized_depth", serialized_depth, persistent=False)

    def forward(
        self,
        grid_coord: torch.Tensor,
        feat: torch.Tensor,
        serialized_code: torch.Tensor,
        *serialized_pooling_inputs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run export-time inference on serialized point inputs.

        Args:
            grid_coord: Discretized grid coordinates.
            feat: Point features whose first three channels are xyz.
            serialized_code: Serialized coordinate codes.

        Returns:
            Predicted labels and point-wise semantic probabilities.
        """
        point_feat, _, _ = _run_ptv3_backbone_export(
            self.backbone,
            grid_coord,
            feat,
            self._serialized_depth,
            serialized_code,
            self._sparse_shape,
            *serialized_pooling_inputs,
        )
        point_logits = self.seg3d_head(point_feat)
        pred_probs = torch.softmax(point_logits, dim=1)
        pred_labels = pred_probs.argmax(dim=1)
        return pred_labels, pred_probs


class PTv3SegmentationModel(PTv3BaseModel):
    """Wrap PTv3 semantic segmentation in the shared training interface."""

    def __init__(
        self,
        backbone: PointTransformerV3Backbone,
        num_classes: int,
        backbone_out_channels: int,
        ignore_index: int,
        grid_size: float,
        point_cloud_range: Sequence[float],
        optimizer: Callable[..., torch.optim.Optimizer],
        scheduler: Callable[[torch.optim.Optimizer], LRScheduler] | None = None,
        optimizer_group_overrides: Mapping[str, Mapping[str, Any]] | None = None,
        scheduler_config: Mapping[str, Any] | None = None,
        lovasz_weight: float = 1.0,
    ) -> None:
        """Initialize the PTv3 segmentation model.

        Args:
            backbone: PTv3 backbone module.
            num_classes: Number of semantic classes.
            backbone_out_channels: Backbone output feature dimension.
            ignore_index: Label value ignored by the losses.
            grid_size: Voxel grid size used to derive sparse shape and
                serialization depth.
            point_cloud_range: Point-cloud range used to derive sparse shape
                and serialization depth.
            optimizer: Optimizer factory.
            scheduler: Scheduler factory.
            optimizer_group_overrides: Optional optimizer overrides keyed by
                model-defined optimizer group name.
            scheduler_config: Optional Lightning scheduler metadata such as
                ``interval`` or ``monitor``.
            lovasz_weight: Weight applied to the Lovasz loss term.
        """
        super().__init__(
            backbone=backbone,
            grid_size=grid_size,
            point_cloud_range=point_cloud_range,
            optimizer=optimizer,
            scheduler=scheduler,
            optimizer_group_overrides=optimizer_group_overrides,
            scheduler_config=scheduler_config,
        )
        self.seg3d_head = nn.Linear(backbone_out_channels, num_classes)
        self.cross_entropy = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.lovasz = LovaszLoss(ignore_index=ignore_index, loss_weight=lovasz_weight)
        self.ignore_index = ignore_index
        self.num_classes = num_classes

    def build_optimizer_groups(self) -> Mapping[str, Sequence[torch.nn.Parameter]]:
        """Group PTv3 parameters structurally for optimizer configuration."""
        block_parameter_ids = {
            id(parameter)
            for module in self.backbone.modules()
            if isinstance(module, Block)
            for parameter in module.parameters()
            if parameter.requires_grad
        }
        default_params: list[torch.nn.Parameter] = []
        block_params: list[torch.nn.Parameter] = []
        for parameter in self.parameters():
            if not parameter.requires_grad:
                continue
            if id(parameter) in block_parameter_ids:
                block_params.append(parameter)
            else:
                default_params.append(parameter)
        return {"default": default_params, "block": block_params}

    def forward(
        self,
        coord: torch.Tensor,
        feat: torch.Tensor,
        grid_coord: torch.Tensor,
        offset: torch.Tensor,
    ) -> torch.Tensor:
        """Run the backbone and segmentation head.

        Args:
            coord: Point coordinates.
            feat: Point features.
            grid_coord: Discretized grid coordinates.
            offset: Batch offsets.

        Returns:
            Voxel-level segmentation logits of shape
            ``(num_voxels, num_classes)``.
        """
        point = self.backbone(
            {
                "coord": coord,
                "feat": feat,
                "grid_coord": grid_coord,
                "offset": offset,
            }
        )
        return self.seg3d_head(point.feat)

    def compute_metrics(
        self,
        batch_inputs_dict: Mapping[str, Any],
        outputs: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute segmentation losses and point-wise accuracy.

        Losses are computed against voxel-level targets for efficiency.
        Accuracy metrics are computed at the original-point level by
        scattering voxel predictions back through ``inverse`` and comparing
        against ``origin_segment``.

        Args:
            batch_inputs_dict: Full batch dictionary. Must contain
                ``segment`` (voxel-level targets), ``inverse`` (voxel-to-point
                mapping), and ``origin_segment`` (original-point targets).
            outputs: Voxel-level segmentation logits returned by
                :meth:`forward`.

        Returns:
            Dictionary with losses and point-level metrics.
        """
        segment = batch_inputs_dict["segment"]
        inverse = batch_inputs_dict["inverse"]
        origin_segment = batch_inputs_dict["origin_segment"]

        loss_ce = self.cross_entropy(outputs, segment)
        loss_lovasz = self.lovasz(outputs, segment)
        metrics: dict[str, torch.Tensor] = {
            "loss_ce": loss_ce,
            "loss_lovasz": loss_lovasz,
            "loss": loss_ce + loss_lovasz,
        }

        with torch.no_grad():
            point_predictions = outputs.argmax(dim=1)[inverse.long()]
            metrics.update(
                compute_segmentation_metrics(
                    point_predictions,
                    origin_segment.long(),
                    self.num_classes,
                    self.ignore_index,
                )
            )

        return metrics

    def predict_outputs(
        self,
        batch_inputs_dict: Mapping[str, Any],
        outputs: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Format PTv3 segmentation predictions at the original-point level.

        Args:
            batch_inputs_dict: Full batch dictionary. Must contain ``inverse``
                (voxel-to-point mapping).
            outputs: Voxel-level segmentation logits returned by
                :meth:`forward`.

        Returns:
            Dictionary with ``"pred_labels"`` (predicted class indices) and
            ``"pred_probs"`` (per-class probabilities), both at the original-
            point level.
        """
        inverse = batch_inputs_dict["inverse"].long()
        point_probs = torch.softmax(outputs, dim=1)[inverse]
        return {"pred_labels": point_probs.argmax(dim=1), "pred_probs": point_probs}

    def get_export_output_names(self) -> list[str]:
        """Return ordered PTv3 segmentation export output names."""
        return ["pred_labels", "pred_probs"]

    def _build_export_module(
        self,
        sparse_shape: torch.Tensor,
        serialized_depth: torch.Tensor,
    ) -> _PTv3ExportModule:
        """Create an isolated PTv3 export module from model copies."""
        export_backbone = self._prepare_backbone_export()
        export_seg3d_head = deepcopy(self.seg3d_head).eval()
        return _PTv3ExportModule(
            export_backbone,
            export_seg3d_head,
            sparse_shape,
            serialized_depth,
        ).eval()

    def build_export_spec(self, batch: Mapping[str, torch.Tensor]) -> ExportSpec:
        """Build the ONNX export specification.

        Args:
            batch: Preprocessed prediction batch containing ``coord``,
                ``feat``, ``grid_coord``, and ``offset``.

        Returns:
            Deployment export specification for PTv3.
        """
        sparse_shape, serialization_depth = self._compute_export_geometry(batch)
        point, _ = serialize_point_cloud_batch(batch, self.EXPORT_ORDER, serialization_depth)
        serialized_pooling_inputs, serialized_pooling_input_names = (
            build_serialized_pooling_export_inputs(
                point["grid_coord"],
                point["serialized_code"],
                point["serialized_order"],
                self.backbone.stride,
            )
        )
        input_args = (
            batch["grid_coord"],
            batch["feat"],
            point["serialized_code"],
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
        dynamic_axes.update(build_point_feature_dynamic_axes(output_names))
        return ExportSpec(
            module=self._build_export_module(sparse_shape, serialization_depth),
            args=input_args,
            input_param_names=input_param_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            supported_stages=self.EXPORT_SUPPORTED_STAGES,
        )

    def build_export_specs(self, batch: Mapping[str, torch.Tensor]) -> dict[str, ExportSpec]:
        """Build split PTv3 segmentation ONNX export specs for backbone and segmentation head."""
        sparse_shape, serialization_depth = self._compute_export_geometry(batch)
        point, input_args = serialize_point_cloud_batch(
            batch, self.EXPORT_ORDER, serialization_depth
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

        seg3d_head_module = _PTv3SegHeadExportModule(deepcopy(self.seg3d_head).eval()).eval()

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
                module=seg3d_head_module,
                args=(point_feat,),
                input_param_names=["point_feat"],
                output_names=self.get_export_output_names(),
                dynamic_axes=build_point_feature_dynamic_axes(
                    ("point_feat", *self.get_export_output_names())
                ),
                supported_stages=self.EXPORT_SUPPORTED_STAGES,
            ),
        }
