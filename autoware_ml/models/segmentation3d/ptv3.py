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
from autoware_ml.deployment.graphs import DeploymentGraph
from autoware_ml.models.segmentation3d.heads.ptv3 import (
    PTv3Seg3DDecoderHead,
    segmentation3d_eval_output,
    segmentation3d_point_loss,
    segmentation3d_predict_outputs,
)
from autoware_ml.models.module_base_model import LogDictConfigs
from autoware_ml.models.segmentation3d.ptv3_base import (
    PTv3BaseModel,
    PTv3EncoderExportModule,
    PTv3Seg3DHeadExportModule,
    split_block_parameters,
)
from autoware_ml.metrics.eval_output import EvalOutput


class PTv3Segmentation3DModel(PTv3BaseModel):
    """Wrap PTv3 semantic segmentation in the shared training interface."""

    def __init__(
        self,
        encoder: PointTransformerV3Encoder,
        voxel_encoder: VoxelFeatureEncoder,
        seg3d_head: PTv3Seg3DDecoderHead,
        grid_size: float,
        point_cloud_range: Sequence[float],
        log_dict_configs: LogDictConfigs,
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
            log_dict_configs: Logging configuration for training and validation.
            **kwargs: Keyword arguments forwarded to :class:`ModuleBaseModel`.
        """
        super().__init__(
            encoder=encoder,
            voxel_encoder=voxel_encoder,
            grid_size=grid_size,
            point_cloud_range=point_cloud_range,
            log_dict_configs=log_dict_configs,
            **kwargs,
        )
        self.seg3d_head = seg3d_head

    def build_optimizer_groups(self) -> Mapping[str, Sequence[torch.nn.Parameter]]:
        """Group PTv3 parameters structurally for optimizer configuration."""
        default_params, block_params = split_block_parameters(self)
        return {"default": default_params, "block": block_params}

    def forward(self, multi_task_batch_inputs: ModelBatchInputs) -> ModelOutputs:
        """Run the voxel encoder, the PTv3 encoder and the segmentation decoder head.

        Args:
            multi_task_batch_inputs: Model inputs holding the point cloud and its voxels.

        Returns:
            Segmentation head outputs with the voxel-level logits of shape
            ``(num_voxels, num_classes)``.
        """
        logits = self.seg3d_head(self.encode_batch(multi_task_batch_inputs))
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
        return segmentation3d_point_loss(
            self.seg3d_head, outputs.segmentation3d().logits, batch_inputs
        )

    def build_eval_output(self, batch: ModelBatchInputs, outputs: ModelOutputs) -> EvalOutput:
        """Gather the voxel predictions of the current frame points for the metrics.

        Args:
            batch: Model inputs of the evaluation step.
            outputs: Segmentation head outputs returned by :meth:`forward`.

        Returns:
            The eval output with the segmentation frames the suites consume.
        """
        return EvalOutput(
            segmentation3d=segmentation3d_eval_output(outputs.segmentation3d().logits, batch)
        )

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
            segmentation3d_predictions=segmentation3d_predict_outputs(
                outputs.segmentation3d().logits, multi_task_batch_inputs
            )
        )

    def deployment_graphs(self) -> Sequence[DeploymentGraph]:
        """Declare ``ptv3_encoder`` and ``ptv3_seg3d_head``, the runtime's split.

        Returns:
            The encoder graph, then the segmentation head graph.
        """
        sparse_shape, serialization_depth = self.compute_export_geometry(torch.device("cpu"))
        return (
            PTv3EncoderExportModule(
                self.encoder,
                self.voxel_encoder,
                self.EXPORT_ORDER,
                sparse_shape,
                serialization_depth,
            ).build_deployment_graph(),
            PTv3Seg3DHeadExportModule(
                self.seg3d_head, self.EXPORT_ORDER, sparse_shape, self.encoder.stride
            ).build_deployment_graph(),
        )
