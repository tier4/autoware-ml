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
from autoware_ml.models.base import BaseModel
from autoware_ml.models.segmentation3d.backbones.ptv3 import PointTransformerV3Backbone
from autoware_ml.utils.point_cloud.structures import Point, bit_length_tensor
from autoware_ml.models.segmentation3d.backbones.ptv3 import Block
from autoware_ml.ops.indexing.operators import argsort
from autoware_ml.utils.deploy import ExportSpec


class _PTv3ExportModule(nn.Module):
    """Expose a deployment-oriented PTv3 export graph without mutating the model."""

    def __init__(
        self,
        backbone: PointTransformerV3Backbone,
        seg_head: nn.Module,
        sparse_shape: torch.Tensor,
    ) -> None:
        """Initialize the isolated PTv3 export module.

        Args:
            backbone: Export-prepared PTv3 backbone copy.
            seg_head: Segmentation head copy.
            sparse_shape: Static sparse shape used by exported sparse ops.
        """
        super().__init__()
        self.backbone = backbone
        self.seg_head = seg_head
        self.register_buffer("_sparse_shape", sparse_shape.to(dtype=torch.long), persistent=False)

    @staticmethod
    def _build_serialized_inverse(serialized_order: torch.Tensor) -> torch.Tensor:
        """Build inverse permutation indices for serialized point orders."""
        return torch.zeros_like(serialized_order).scatter_(
            dim=1,
            index=serialized_order,
            src=torch.arange(serialized_order.shape[1], device=serialized_order.device).repeat(
                serialized_order.shape[0], 1
            ),
        )

    def forward(
        self,
        grid_coord: torch.Tensor,
        feat: torch.Tensor,
        serialized_depth: torch.Tensor,
        serialized_code: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run export-time inference on serialized point inputs.

        Args:
            grid_coord: Discretized grid coordinates.
            feat: Point features whose first three channels are xyz.
            serialized_depth: Serialization depth.
            serialized_code: Serialized coordinate codes.

        Returns:
            Predicted labels and point-wise semantic probabilities.
        """
        shape = torch._shape_as_tensor(grid_coord).to(grid_coord.device)
        serialized_order = torch.stack([argsort(code) for code in serialized_code], dim=0)
        serialized_inverse = self._build_serialized_inverse(serialized_order)
        point = self.backbone.export_forward(
            {
                "coord": feat[:, :3],
                "feat": feat,
                "grid_coord": grid_coord,
                "offset": shape[:1],
                "serialized_depth": serialized_depth,
                "serialized_code": serialized_code,
                "serialized_order": serialized_order,
                "serialized_inverse": serialized_inverse,
                "sparse_shape": self._sparse_shape,
            }
        )
        point_logits = self.seg_head(point.feat)
        pred_probs = torch.softmax(point_logits, dim=1)
        pred_labels = pred_probs.argmax(dim=1)
        return pred_labels, pred_probs


class PTv3SegmentationModel(BaseModel):
    """Wrap PTv3 semantic segmentation in the shared training interface."""

    EXPORT_ORDER = ("z", "z-trans")
    EXPORT_SUPPORTED_STAGES = frozenset({"onnx"})

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
            grid_size: Voxel grid size used to derive sparse shape and serialization depth.
            point_cloud_range: Point-cloud range used to derive sparse shape and serialization depth.
            optimizer: Optimizer factory.
            scheduler: Scheduler factory.
            optimizer_group_overrides: Optional optimizer overrides keyed by
                model-defined optimizer group name.
            scheduler_config: Optional Lightning scheduler metadata such as
                ``interval`` or ``monitor``.
            lovasz_weight: Weight applied to the Lovasz loss term.
        """
        super().__init__(
            optimizer=optimizer,
            scheduler=scheduler,
            optimizer_group_overrides=optimizer_group_overrides,
            scheduler_config=scheduler_config,
        )
        self.backbone = backbone
        self.seg_head = nn.Linear(backbone_out_channels, num_classes)
        self.cross_entropy = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.lovasz = LovaszLoss(ignore_index=ignore_index, loss_weight=lovasz_weight)
        self.ignore_index = ignore_index
        self.num_classes = num_classes
        self.grid_size = grid_size
        self.point_cloud_range = tuple(float(v) for v in point_cloud_range)

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
            {"coord": coord, "feat": feat, "grid_coord": grid_coord, "offset": offset}
        )
        return self.seg_head(point.feat)

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

    def get_log_batch_size(self, batch_inputs_dict: Mapping[str, torch.Tensor]) -> int:
        """Return the number of samples represented by the serialized point batch."""
        return int(batch_inputs_dict["offset"].numel())

    @classmethod
    def _prepare_export_backbone(
        cls, backbone: PointTransformerV3Backbone
    ) -> PointTransformerV3Backbone:
        """Delegate PTv3 export preparation to the backbone implementation."""
        return backbone.prepare_export_copy(cls.EXPORT_ORDER)

    def _build_export_module(self, sparse_shape: torch.Tensor) -> _PTv3ExportModule:
        """Create an isolated PTv3 export module from model copies."""
        export_backbone = self._prepare_export_backbone(self.backbone)
        export_seg_head = deepcopy(self.seg_head).eval()
        return _PTv3ExportModule(export_backbone, export_seg_head, sparse_shape)

    def build_export_spec(self, batch: Mapping[str, torch.Tensor]) -> ExportSpec:
        """Build the ONNX export specification.

        Args:
            batch: Preprocessed prediction batch containing ``coord``,
                ``feat``, ``grid_coord``, and ``offset``.

        Returns:
            Deployment export specification for PTv3.
        """
        point_cloud_range = torch.tensor(
            self.point_cloud_range, dtype=torch.float32, device=batch["coord"].device
        )
        axis_extents = (point_cloud_range[3:] - point_cloud_range[:3]) / self.grid_size
        serialization_depth = bit_length_tensor(torch.max(axis_extents))
        sparse_shape = torch.round(axis_extents).to(dtype=torch.long)
        point = Point(
            {
                "coord": batch["coord"],
                "feat": batch["feat"],
                "grid_coord": batch["grid_coord"],
                "offset": batch["offset"],
            }
        )
        point.serialization(self.EXPORT_ORDER, shuffle_orders=False, depth=serialization_depth)
        input_args = (
            batch["grid_coord"],
            batch["feat"],
            point["serialized_depth"],
            point["serialized_code"],
        )
        return ExportSpec(
            module=self._build_export_module(sparse_shape),
            args=input_args,
            input_param_names=["grid_coord", "feat", "serialized_depth", "serialized_code"],
            output_names=self.get_export_output_names(),
            supported_stages=self.EXPORT_SUPPORTED_STAGES,
        )
