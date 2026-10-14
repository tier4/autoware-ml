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

from autoware_ml.dataclasses.models.model_batch_inputs import ModelBatchInputs
from autoware_ml.dataclasses.models.model_outputs import ModelOutputs
from autoware_ml.dataclasses.models.model_predictions import ModelPredictions
from autoware_ml.dataclasses.models.segmentation3d.head_outputs import Segmentation3DHeadOutputs
from autoware_ml.models.segmentation3d.encoders.ptv3 import PointTransformerV3Encoder
from autoware_ml.models.segmentation3d.encoders.voxel import VoxelFeatureEncoder
from autoware_ml.models.segmentation3d.heads.ptv3 import (
    PTv3SegDecoderHead,
    segmentation_eval_output,
    segmentation_point_loss,
    segmentation_predict_outputs,
)
from autoware_ml.models.segmentation3d.ptv3_base import (
    SERIALIZED_POOLING_FIELDS,
    PTv3BaseModel,
    PTv3EncoderExportBase,
    build_encoder_export_spec,
    build_point_feature_dynamic_axes,
    build_ptv3_export_context,
    build_ptv3_input_dynamic_axes,
    build_seg_head_export_spec,
    prepare_ptv3_export_inputs,
    split_block_parameters,
)
from autoware_ml.utils.deploy import ExportSpec


class _PTv3SegmentationExportModule(PTv3EncoderExportBase):
    """Expose a deployment-oriented PTv3 export graph without mutating the model."""

    def __init__(
        self,
        encoder: PointTransformerV3Encoder,
        voxel_encoder: VoxelFeatureEncoder,
        seg3d_head: PTv3SegDecoderHead,
        sparse_shape: torch.Tensor,
        serialized_depth: torch.Tensor,
    ) -> None:
        """Initialize the isolated PTv3 export module.

        Args:
            encoder: Export-prepared PTv3 encoder copy.
            voxel_encoder: Voxel feature encoder feeding the embedding stem.
            seg3d_head: Export-prepared decoder head copy.
            sparse_shape: Static sparse shape used by exported sparse ops.
            serialized_depth: Serialization depth baked at export time.
        """
        super().__init__(encoder, voxel_encoder, sparse_shape, serialized_depth)
        self.seg3d_head = seg3d_head

    def forward(
        self,
        voxels: torch.Tensor,
        num_points_per_voxel: torch.Tensor,
        grid_coord: torch.Tensor,
        serialized_order: torch.Tensor,
        serialized_inverse: torch.Tensor,
        *serialized_pooling_inputs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run export-time inference on serialized voxel inputs.

        Args:
            voxels: Padded voxel points.
            num_points_per_voxel: Valid point count per voxel.
            grid_coord: Integer voxel coordinates.
            serialized_order: Level-0 serialization order, one row per curve.
            serialized_inverse: Inverse of ``serialized_order``.
            serialized_pooling_inputs: Precomputed pooling metadata tensors.

        Returns:
            Predicted labels and voxel-wise semantic probabilities.
        """
        point = self.run_encoder(
            voxels,
            num_points_per_voxel,
            grid_coord,
            serialized_order,
            serialized_inverse,
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
        voxel_encoder: VoxelFeatureEncoder,
        seg3d_head: PTv3SegDecoderHead,
        grid_size: float,
        point_cloud_range: Sequence[float],
        **kwargs: Any,
    ) -> None:
        """Initialize the PTv3 segmentation model.

        Args:
            encoder: PTv3 encoder module.
            voxel_encoder: Voxel feature encoder feeding the embedding stem.
            seg3d_head: Segmentation decoder head owning losses and the
                classifier.
            grid_size: Voxel grid size used to derive sparse shape and
                serialization depth.
            point_cloud_range: Point-cloud range used to derive sparse shape
                and serialization depth.
            **kwargs: Keyword arguments forwarded to :class:`ModuleBaseModel`.
        """
        super().__init__(
            encoder=encoder,
            voxel_encoder=voxel_encoder,
            grid_size=grid_size,
            point_cloud_range=point_cloud_range,
            **kwargs,
        )
        self.seg3d_head = seg3d_head

    def build_optimizer_groups(self) -> Mapping[str, Sequence[torch.nn.Parameter]]:
        """Group PTv3 parameters structurally for optimizer configuration."""
        default_params, block_params = split_block_parameters(self)
        return {"default": default_params, "block": block_params}

    def forward_network(
        self,
        voxels: torch.Tensor,
        num_points: torch.Tensor,
        voxel_coords: torch.Tensor,
        num_dropped_voxels: torch.Tensor,
        time_lag_column: int,
    ) -> ModelOutputs:
        """Run the voxel encoder, the PTv3 encoder and the segmentation decoder head.

        Args:
            voxels: Padded voxel points from the data preprocessing.
            num_points: Valid point count per voxel.
            voxel_coords: Voxel coordinates with a leading batch column.
            num_dropped_voxels: Occupied voxels the voxelizer discarded.
            time_lag_column: Point column holding the time lag.

        Returns:
            Segmentation head outputs with the voxel-level logits of shape
            ``(num_voxels, num_classes)``.
        """
        logits = self.seg3d_head(
            self.encode(voxels, num_points, voxel_coords, num_dropped_voxels, time_lag_column)
        )
        return ModelOutputs(segmentation3d_head_outputs=Segmentation3DHeadOutputs(logits=logits))

    def compute_metrics(
        self,
        batch_inputs: ModelBatchInputs,
        outputs: ModelOutputs,
    ) -> dict[str, torch.Tensor]:
        """Compute segmentation losses against point-level targets.

        Every voxel is supervised once by the label reduced from the points it holds.
        Quality metrics (mIoU, accuracy) are produced at epoch end by the
        configured metrics through :meth:`build_eval_output`, not here.

        Args:
            batch_inputs: Model inputs with the voxels and the point labels.
            outputs: Segmentation head outputs returned by :meth:`forward`.

        Returns:
            Dictionary with the segmentation losses.
        """
        return segmentation_point_loss(
            self.seg3d_head, outputs.segmentation3d().logits, batch_inputs
        )

    def build_eval_output(
        self, batch: ModelBatchInputs, outputs: ModelOutputs
    ) -> dict[str, torch.Tensor]:
        """Gather the voxel predictions of the current frame points for the metrics.

        Args:
            batch: Model inputs of the evaluation step.
            outputs: Segmentation head outputs returned by :meth:`forward`.

        Returns:
            The seg_frames eval output the segmentation suites consume.
        """
        return segmentation_eval_output(outputs.segmentation3d().logits, batch)

    def decode_outputs(
        self, multi_task_batch_inputs: ModelBatchInputs, outputs: ModelOutputs
    ) -> ModelPredictions:
        """Format PTv3 segmentation predictions for the current-frame points.

        Args:
            multi_task_batch_inputs: Model inputs of the prediction step.
            outputs: Segmentation head outputs returned by :meth:`forward`.

        Returns:
            The predicted labels and probabilities.
        """
        return ModelPredictions(
            segmentation3d_predictions=segmentation_predict_outputs(
                outputs.segmentation3d().logits, multi_task_batch_inputs
            )
        )

    def get_export_output_names(self) -> list[str]:
        """Return ordered PTv3 segmentation export output names."""
        return ["pred_labels", "pred_probs"]

    def build_export_spec(self, batch_inputs: ModelBatchInputs) -> ExportSpec:
        """Build the ONNX export specification.

        Args:
            batch_inputs: Example preprocessed batch with the voxels.

        Returns:
            Deployment export specification for PTv3.
        """
        inputs = prepare_ptv3_export_inputs(self, batch_inputs)
        input_args, input_param_names = inputs.encoder_args(SERIALIZED_POOLING_FIELDS)
        export_module = _PTv3SegmentationExportModule(
            self._prepare_encoder_export(),
            self.voxel_encoder,
            self.seg3d_head.prepare_for_export(self.EXPORT_ORDER),
            inputs.sparse_shape,
            inputs.serialization_depth,
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

    def build_export_specs(self, batch_inputs: ModelBatchInputs) -> dict[str, ExportSpec]:
        """Build split PTv3 segmentation ONNX export specs for encoder and head.

        Args:
            batch_inputs: Example preprocessed batch with the voxels.

        Returns:
            Export specs of the encoder and the segmentation head.
        """
        context = build_ptv3_export_context(self, batch_inputs)
        return {
            "ptv3_encoder": build_encoder_export_spec(context),
            "ptv3_seg3d_head": build_seg_head_export_spec(
                context,
                self.seg3d_head.prepare_for_export(self.EXPORT_ORDER),
                self.get_export_output_names(),
            ),
        }
