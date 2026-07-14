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
The model composes the shared PTv3 encoder with the segmentation decoder
head; losses and prediction formatting are owned by the head module.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
import torch.nn as nn

from autoware_ml.models.segmentation3d.encoders.ptv3 import PointTransformerV3Encoder
from autoware_ml.models.segmentation3d.heads.ptv3 import (
    PTv3SegDecoderHead,
    segmentation_eval_output,
    segmentation_predict_outputs,
)
from autoware_ml.models.segmentation3d.ptv3_base import (
    PTv3BaseModel,
    _run_ptv3_encoder_export,
    build_encoder_export_spec,
    build_point_feature_dynamic_axes,
    build_ptv3_export_context,
    build_ptv3_input_dynamic_axes,
    build_seg_head_export_spec,
    build_serialized_pooling_metadata,
    flatten_serialized_pooling_inputs,
    split_block_parameters,
)
from autoware_ml.utils.deploy import ExportSpec
from autoware_ml.utils.point_cloud.structures import serialize_point_cloud_batch


class _PTv3SegmentationExportModule(nn.Module):
    """Expose a deployment-oriented PTv3 export graph without mutating the model."""

    def __init__(
        self,
        encoder: PointTransformerV3Encoder,
        seg3d_head: PTv3SegDecoderHead,
        sparse_shape: torch.Tensor,
        serialized_depth: torch.Tensor,
    ) -> None:
        """Initialize the isolated PTv3 export module.

        Args:
            encoder: Export-prepared PTv3 encoder copy.
            seg3d_head: Export-prepared decoder head copy.
            sparse_shape: Static sparse shape used by exported sparse ops.
            serialized_depth: Serialization depth baked at export time.
        """
        super().__init__()
        self.encoder = encoder
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
        point = _run_ptv3_encoder_export(
            self.encoder,
            grid_coord,
            feat,
            self._serialized_depth,
            serialized_code,
            self._sparse_shape,
            *serialized_pooling_inputs,
        )
        point_logits = self.seg3d_head(point)
        pred_probs = torch.softmax(point_logits, dim=1)
        pred_labels = pred_probs.argmax(dim=1)
        return pred_labels, pred_probs


class PTv3SegmentationModel(PTv3BaseModel):
    """Wrap PTv3 semantic segmentation in the shared training interface."""

    def __init__(
        self,
        encoder: PointTransformerV3Encoder,
        seg3d_head: PTv3SegDecoderHead,
        grid_size: float,
        point_cloud_range: Sequence[float],
        **kwargs: Any,
    ) -> None:
        """Initialize the PTv3 segmentation model.

        Args:
            encoder: PTv3 encoder module.
            seg3d_head: Segmentation decoder head owning losses and the
                classifier.
            grid_size: Voxel grid size used to derive sparse shape and
                serialization depth.
            point_cloud_range: Point-cloud range used to derive sparse shape
                and serialization depth.
            **kwargs: Keyword arguments forwarded to :class:`BaseModel`.
        """
        super().__init__(
            encoder=encoder,
            grid_size=grid_size,
            point_cloud_range=point_cloud_range,
            **kwargs,
        )
        self.seg3d_head = seg3d_head

    def build_optimizer_groups(self) -> Mapping[str, Sequence[torch.nn.Parameter]]:
        """Group PTv3 parameters structurally for optimizer configuration."""
        default_params, block_params = split_block_parameters(self)
        return {"default": default_params, "block": block_params}

    def forward(
        self,
        coord: torch.Tensor,
        feat: torch.Tensor,
        grid_coord: torch.Tensor,
        offset: torch.Tensor,
    ) -> torch.Tensor:
        """Run the encoder and segmentation decoder head.

        Args:
            coord: Point coordinates.
            feat: Point features.
            grid_coord: Discretized grid coordinates.
            offset: Batch offsets.

        Returns:
            Voxel-level segmentation logits of shape
            ``(num_voxels, num_classes)``.
        """
        point = self.encoder(
            {
                "coord": coord,
                "feat": feat,
                "grid_coord": grid_coord,
                "offset": offset,
            }
        )
        return self.seg3d_head(point)

    def compute_metrics(
        self,
        batch_inputs_dict: Mapping[str, Any],
        outputs: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute segmentation losses against voxel-level targets.

        Quality metrics (mIoU, accuracy) are produced at epoch end by the
        configured metrics through :meth:`build_eval_output`, not here.

        Args:
            batch_inputs_dict: Full batch dictionary. Must contain ``segment``
                (voxel-level targets).
            outputs: Voxel-level segmentation logits returned by :meth:`forward`.

        Returns:
            Dictionary with the segmentation losses.
        """
        return self.seg3d_head.loss(outputs, batch_inputs_dict["segment"])

    def build_eval_output(
        self, batch: Mapping[str, Any], outputs: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Scatter voxel predictions to points for the segmentation metric."""
        return segmentation_eval_output(outputs, batch)

    def predict_outputs(
        self,
        batch_inputs_dict: Mapping[str, Any],
        outputs: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Format PTv3 segmentation predictions at the original-point level."""
        return segmentation_predict_outputs(outputs, batch_inputs_dict)

    def get_export_output_names(self) -> list[str]:
        """Return ordered PTv3 segmentation export output names."""
        return ["pred_labels", "pred_probs"]

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
            flatten_serialized_pooling_inputs(
                build_serialized_pooling_metadata(
                    point["grid_coord"],
                    point["serialized_code"],
                    point["serialized_order"],
                    self.encoder.stride,
                )
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
        export_module = _PTv3SegmentationExportModule(
            self._prepare_encoder_export(),
            self.seg3d_head.prepare_for_export(self.EXPORT_ORDER),
            sparse_shape,
            serialization_depth,
        ).eval()
        output_names = self.get_export_output_names()
        dynamic_axes = build_ptv3_input_dynamic_axes(input_param_names)
        dynamic_axes.update(build_point_feature_dynamic_axes(output_names))
        return ExportSpec(
            module=export_module,
            args=input_args,
            input_param_names=input_param_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            supported_stages=self.EXPORT_SUPPORTED_STAGES,
        )

    def build_export_specs(self, batch: Mapping[str, torch.Tensor]) -> dict[str, ExportSpec]:
        """Build split PTv3 segmentation ONNX export specs for encoder and head."""
        context = build_ptv3_export_context(self, batch)
        return {
            "encoder": build_encoder_export_spec(context),
            "seg3d_head": build_seg_head_export_spec(
                context,
                self.seg3d_head.prepare_for_export(self.EXPORT_ORDER),
                self.get_export_output_names(),
            ),
        }
