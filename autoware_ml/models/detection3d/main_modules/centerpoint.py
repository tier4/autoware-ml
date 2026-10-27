# Copyright 2023 OpenMMLab.
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

"""Native CenterPoint lidar detector wrapper.

This module provides the task-level training, inference, and export wrapper
around the reusable PointPillars and CenterPoint detection components.
"""

from __future__ import annotations

from types import MappingProxyType
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from jaxtyping import Float32
import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from autoware_ml.dataclasses.models.model_batch_inputs import ModelBatchInputs
from autoware_ml.dataclasses.models.model_predictions import ModelPredictions
from autoware_ml.dataclasses.models.model_outputs import ModelOutputs
from autoware_ml.dataclasses.geometry.voxels import VoxelsData
from autoware_ml.dataclasses.models.detection3d.head_outputs import (
    CenterHeadOutputs,
    Detection3DHeadOutputs,
)
from autoware_ml.deployment.stages import GraphStage, RunState, Stage, TorchStage
from autoware_ml.metrics.base import MetricSuite
from autoware_ml.metrics.detection3d.eval_output import detection3d_eval_output
from autoware_ml.models.module_base_model import LogDictConfigs, ModuleBaseModel
from autoware_ml.models.detection3d.encoders.pillars.pillar_feature_net import PillarFeatureNet
from autoware_ml.models.detection3d.encoders.pillars.point_pillar_scatter import PointPillarsScatter
from autoware_ml.models.detection3d.heads.centerhead import CenterHead
from autoware_ml.preprocessing.data_preprocessor import DataPreprocessor
from autoware_ml.utils.deploy import ExportSpec


class _CenterPointVoxelEncoderExportWrapper(nn.Module):
    """Export the PointPillars PFN stack from decorated input features."""

    def __init__(self, voxel_encoder: PillarFeatureNet) -> None:
        """Initialize the voxel encoder export wrapper.

        Args:
            voxel_encoder: Pillar feature network whose PFN layers are exported.
        """
        super().__init__()
        self.voxel_encoder = voxel_encoder

    def forward(
        self,
        input_features: Float32[torch.Tensor, "num_pillars max_num_points feature_channels"],
    ) -> Float32[torch.Tensor, "num_pillars 1 num_output_channels"]:
        """Encode already-decorated pillar features."""
        return self.voxel_encoder.encode_decorated(input_features)


class _CenterPointBackboneNeckHeadExportWrapper(nn.Module):
    """Export CenterPoint backbone, neck, and dense head from BEV features."""

    def __init__(self, backbone: nn.Module, neck: nn.Module, bbox_head: CenterHead) -> None:
        """Initialize the backbone-neck-head export wrapper.

        Args:
            backbone: BEV backbone.
            neck: BEV neck.
            bbox_head: CenterPoint dense head. An export-ready copy is taken.
        """
        super().__init__()
        self.backbone = backbone
        self.neck = neck
        self.bbox_head = bbox_head.prepare_for_export()
        # The order the deployed runtime expects, the velocity branch last. Do not reorder.
        self.output_names = ["heatmap", "reg", "height", "dim", "rot"]
        if self.bbox_head.use_velocity:
            self.output_names.append("vel")

    def forward(
        self,
        spatial_features: Float32[torch.Tensor, "batch_size num_channels height width"],
    ) -> tuple[torch.Tensor, ...]:
        """Run BEV feature extraction and flatten the head outputs for export."""
        bev_features = self.backbone(spatial_features)
        bev_features = self.neck(bev_features)
        return self.bbox_head(bev_features).export_tensors(self.output_names)


class CenterPointDetection3DModel(ModuleBaseModel):
    """Compose a CenterPoint detector from reusable lidar detection modules.

    The wrapper wires together pillar encoding, BEV feature extraction, and the
    CenterPoint dense head inside the shared :class:`BaseModel` interface.
    """

    def __init__(
        self,
        data_preprocessor: DataPreprocessor,
        pts_voxel_encoder: PillarFeatureNet,
        pts_middle_encoder: PointPillarsScatter,
        pts_backbone: torch.nn.Module,
        pts_neck: torch.nn.Module,
        bbox_head: CenterHead,
        log_dict_configs: LogDictConfigs,
        optimizer: Callable[..., Optimizer] | None = None,
        scheduler: Callable[[Optimizer], LRScheduler] | None = None,
        metrics: Sequence[MetricSuite] | None = None,
    ) -> None:
        """
        Initialize CenterPoint.

        Args:
            data_preprocessor: Multi-task data preprocessor.
            pts_voxel_encoder: Lidar voxel feature encoder.
            pts_middle_encoder: Sparse 3D or pillar-scatter middle encoder.
            pts_backbone: BEV backbone.
            pts_neck: BEV neck.
            bbox_head: CenterPoint dense detection head.
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

    def build_eval_output(self, batch: ModelBatchInputs, outputs: ModelOutputs) -> dict[str, Any]:
        """Decode detections and pair them with ground truth for metrics."""
        if outputs.detection3d_head_outputs is None:
            raise ValueError(
                "ModelOutputs must contain detection3d_head_outputs for CenterPoint build_eval_output pass."
            )

        return detection3d_eval_output(
            predictions=self.bbox_head.decode_outputs(outputs.detection3d_head_outputs),
            batch_inputs=batch,
        )

    def forward(self, multi_task_batch_inputs: ModelBatchInputs) -> ModelOutputs:
        """Run the detector on voxelized lidar inputs.

        Args:
            multi_task_batch_inputs: ModelBatchInputs containing the voxelized lidar inputs.

        Returns:
            Detection head outputs.
        """
        if multi_task_batch_inputs.voxels_data is None:
            raise ValueError(
                "ModelBatchInputs must contain voxels_data for CenterPoint forward pass."
            )

        batch_size = multi_task_batch_inputs.multi_task_gt_batch.batch_size
        pillar_features = self.pts_voxel_encoder(multi_task_batch_inputs.voxels_data)
        bev_features = self.pts_middle_encoder(
            pillar_features=pillar_features,
            coords=multi_task_batch_inputs.voxels_data.coords,
            batch_indices=multi_task_batch_inputs.voxels_data.batch_indices,
            batch_size=batch_size,
        )
        bev_features = self.pts_backbone(bev_features)
        bev_features = self.pts_neck(bev_features)
        head_outputs = self.bbox_head(bev_features)
        return ModelOutputs(
            detection3d_head_outputs=Detection3DHeadOutputs(
                center_head_outputs=head_outputs, transfusion_head_outputs=None
            )
        )

    def compute_metrics(
        self, multi_task_batch_inputs: ModelBatchInputs, multi_task_outputs: ModelOutputs
    ) -> MappingProxyType[str, Float32[torch.Tensor, " 1"]]:
        """Compute CenterPoint training losses."""
        if multi_task_batch_inputs.multi_task_gt_batch.detection3d_gt_batch is None:
            raise ValueError(
                "ModelBatchInputs must contain detection3d_gt_batch for CenterPoint compute_metrics pass."
            )

        if multi_task_outputs.detection3d_head_outputs is None:
            raise ValueError(
                "ModelOutputs must contain detection3d_head_outputs for CenterPoint compute_metrics pass."
            )

        gt_bboxes_3d = multi_task_batch_inputs.multi_task_gt_batch.detection3d_gt_batch.gt_bboxes_3d
        gt_labels_3d = multi_task_batch_inputs.multi_task_gt_batch.detection3d_gt_batch.gt_labels_3d
        gt_valid_bboxes = (
            multi_task_batch_inputs.multi_task_gt_batch.detection3d_gt_batch.gt_valid_bboxes
        )

        return self.bbox_head.loss(
            outputs=multi_task_outputs.detection3d_head_outputs,
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            gt_valid_bboxes=gt_valid_bboxes,
        )

    def decode_outputs(
        self, multi_task_batch_inputs: ModelBatchInputs, outputs: ModelOutputs
    ) -> ModelPredictions:
        """Decode predictions for inference."""
        if outputs.detection3d_head_outputs is None:
            raise ValueError(
                "ModelOutputs must contain detection3d_head_outputs for CenterPoint decode_outputs pass."
            )

        multi_task_predictions = self.bbox_head.decode_outputs(
            outputs=outputs.detection3d_head_outputs
        )
        return multi_task_predictions

    # TODO(vividf): Remove together with ModuleBaseModel.build_export_spec().
    def build_export_spec(self, multi_task_batch_inputs: ModelBatchInputs) -> ExportSpec:
        """Reject single-module CenterPoint deployment export."""
        del multi_task_batch_inputs
        raise RuntimeError(
            "CenterPoint deployment uses split modules; build_export_specs() derives them "
            "from build_stages()."
        )

    def build_stages(self) -> Sequence[Stage]:
        """Declare the split the Autoware runtime loads; stage names match its artifact names::

            decorate (torch) -> pts_voxel_encoder_centerpoint (graph)
                -> scatter (torch) -> pts_backbone_neck_head_centerpoint (graph)

        Pillar decoration and the BEV scatter have no learned parameters and stay out of the
        graphs.
        """

        def voxels_data(state: RunState) -> VoxelsData:
            voxels_data = state.batch.voxels_data
            if voxels_data is None:
                raise ValueError(
                    "ModelBatchInputs must contain voxels_data to run the CenterPoint stages."
                )
            return voxels_data

        def decorate(state: RunState) -> dict[str, torch.Tensor]:
            return {"input_features": self.pts_voxel_encoder.encode(voxels_data(state))}

        def scatter(state: RunState) -> dict[str, torch.Tensor]:
            voxels = voxels_data(state)
            pillar_features = state["pillar_features"].to(state.device).squeeze(1)
            # Sized by the batch, as in forward(): the voxel batch indices miss a sample
            # without voxels.
            return {
                "spatial_features": self.pts_middle_encoder(
                    pillar_features=pillar_features,
                    coords=voxels.coords.to(state.device),
                    batch_indices=voxels.batch_indices.to(state.device),
                    batch_size=state.batch.batch_size(),
                )
            }

        head_wrapper = _CenterPointBackboneNeckHeadExportWrapper(
            self.pts_backbone, self.pts_neck, self.bbox_head
        )
        return (
            TorchStage("decorate", run=decorate, inputs=(), outputs=("input_features",)),
            GraphStage(
                "pts_voxel_encoder_centerpoint",
                module=_CenterPointVoxelEncoderExportWrapper(self.pts_voxel_encoder),
                inputs=("input_features",),
                outputs=("pillar_features",),
            ),
            TorchStage(
                "scatter", run=scatter, inputs=("pillar_features",), outputs=("spatial_features",)
            ),
            GraphStage(
                "pts_backbone_neck_head_centerpoint",
                module=head_wrapper,
                inputs=("spatial_features",),
                # Named after the CenterHeadOutputs fields; build_deployed_eval_output
                # rebuilds the typed outputs by name.
                outputs=tuple(head_wrapper.output_names),
            ),
        )

    def build_deployed_eval_output(
        self, multi_task_batch_inputs: ModelBatchInputs, outputs: Mapping[str, torch.Tensor]
    ) -> dict[str, Any]:
        """Score the deployed head maps exactly like forward()'s."""
        return self.build_eval_output(
            multi_task_batch_inputs,
            ModelOutputs(
                detection3d_head_outputs=Detection3DHeadOutputs(
                    center_head_outputs=CenterHeadOutputs.from_export_tensors(outputs),
                    transfusion_head_outputs=None,
                )
            ),
        )
