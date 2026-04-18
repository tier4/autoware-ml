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
from autoware_ml.models.segmentation3d.backbones.ptv3 import PointTransformerV3Backbone
from autoware_ml.utils.point_cloud.structures import Point
from autoware_ml.models.segmentation3d.backbones.ptv3 import Block
from autoware_ml.models.segmentation3d.base import BaseSegmentationModel
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


class PTv3SegmentationModel(BaseSegmentationModel):
    """Wrap PTv3 semantic segmentation in the shared training interface."""

    EXPORT_ORDER = ("z", "z-trans")
    EXPORT_SUPPORTED_STAGES = frozenset({"onnx"})

    def __init__(
        self,
        backbone: PointTransformerV3Backbone,
        num_classes: int,
        backbone_out_channels: int,
        ignore_index: int,
        optimizer: Callable[..., torch.optim.Optimizer],
        scheduler: Callable[[torch.optim.Optimizer], LRScheduler] | None = None,
        optimizer_group_overrides: Mapping[str, Mapping[str, Any]] | None = None,
        scheduler_config: Mapping[str, Any] | None = None,
        lovasz_weight: float = 1.0,
        export_grid_size: float | None = None,
        export_point_cloud_range: Sequence[float] | None = None,
    ) -> None:
        """Initialize the PTv3 segmentation model.

        Args:
            backbone: PTv3 backbone module.
            num_classes: Number of semantic classes.
            backbone_out_channels: Backbone output feature dimension.
            ignore_index: Label value ignored by the losses.
            optimizer: Optimizer factory.
            scheduler: Scheduler factory.
            optimizer_group_overrides: Optional optimizer overrides keyed by
                model-defined optimizer group name.
            scheduler_config: Optional Lightning scheduler metadata such as
                ``interval`` or ``monitor``.
            lovasz_weight: Weight applied to the Lovasz loss term.
            export_grid_size: Grid size used to derive the export sparse shape.
            export_point_cloud_range: Point-cloud range used to derive the
                export sparse shape.
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
        self.export_grid_size = export_grid_size
        self.export_point_cloud_range = (
            tuple(float(value) for value in export_point_cloud_range)
            if export_point_cloud_range is not None
            else None
        )

    def build_optimizer_groups(self) -> Mapping[str, Sequence[torch.nn.Parameter]]:
        """Group PTv3 parameters structurally for optimizer configuration.

        AWML tunes transformer block parameters with a lower learning rate than
        the rest of the model. This hook keeps that grouping tied to actual
        :class:`Block` modules instead of relying on parameter-name matching.
        """
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
            Point-wise segmentation logits.
        """
        point = self.backbone(
            {"coord": coord, "feat": feat, "grid_coord": grid_coord, "offset": offset}
        )
        return self.seg_head(point.feat)

    def compute_metrics(
        self,
        outputs: torch.Tensor,
        segment: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute segmentation losses and point-wise accuracy.

        Args:
            outputs: Point-wise segmentation logits.
            segment: Ground-truth point labels.

        Returns:
            Dictionary with losses and point accuracy.
        """
        loss_ce = self.cross_entropy(outputs, segment)
        loss_lovasz = self.lovasz(outputs, segment)
        metrics: dict[str, torch.Tensor] = {
            "loss_ce": loss_ce,
            "loss_lovasz": loss_lovasz,
            "loss": loss_ce + loss_lovasz,
        }

        with torch.no_grad():
            metrics.update(self._compute_segmentation_metrics(outputs, segment))

        return metrics

    def _get_point_logits(self, outputs: torch.Tensor) -> torch.Tensor:
        """Extract point-wise logits (identity - forward returns logits)."""
        return outputs

    def get_log_batch_size(self, batch_inputs_dict: Mapping[str, torch.Tensor]) -> int | None:
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

    def _resolve_export_sparse_shape(
        self, device: torch.device, grid_coord: torch.Tensor
    ) -> torch.Tensor:
        """Resolve the sparse shape used by exported sparse-convolution ops."""
        if self.export_point_cloud_range is not None and self.export_grid_size is not None:
            point_cloud_range = torch.tensor(
                self.export_point_cloud_range,
                dtype=torch.float32,
                device=device,
            )
            return torch.round(
                (point_cloud_range[3:] - point_cloud_range[:3]) / self.export_grid_size
            ).to(dtype=torch.long)
        return torch.max(grid_coord, dim=0).values + 96

    def build_export_spec(self, batch: Mapping[str, torch.Tensor]) -> ExportSpec:
        """Build the ONNX export specification.

        Args:
            batch: Preprocessed prediction batch used to derive export tensors.

        Returns:
            Deployment export specification for PTv3.
        """
        point = Point(
            {
                "coord": batch["coord"],
                "feat": batch["feat"],
                "grid_coord": batch["grid_coord"],
                "offset": batch["offset"],
            }
        )
        point.serialization(self.EXPORT_ORDER, shuffle_orders=False)
        sparse_shape = self._resolve_export_sparse_shape(
            batch["grid_coord"].device, point["grid_coord"]
        )
        input_names = [
            "grid_coord",
            "feat",
            "serialized_depth",
            "serialized_code",
        ]
        input_args = (
            batch["grid_coord"],
            batch["feat"],
            point["serialized_depth"],
            point["serialized_code"],
        )
        return ExportSpec(
            module=self._build_export_module(sparse_shape),
            args=input_args,
            input_param_names=input_names,
            output_names=self.get_export_output_names(),
            supported_stages=self.EXPORT_SUPPORTED_STAGES,
        )
