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

from collections.abc import Callable, Sequence
from typing import Any

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from autoware_ml.dataclasses.batch.detection3d import Detection3DGTBatch
from autoware_ml.dataclasses.geometry.voxels import VoxelsData
from autoware_ml.dataclasses.models.detection3d.head_outputs import Detection3DHeadOutputs
from autoware_ml.dataclasses.models.model_batch_inputs import ModelBatchInputs
from autoware_ml.dataclasses.models.model_outputs import ModelOutputs
from autoware_ml.dataclasses.models.model_predictions import ModelPredictions
from autoware_ml.metrics.base import MetricSuite
from autoware_ml.metrics.detection3d.eval_output import detection_eval_output
from autoware_ml.models.base import BaseModel
from autoware_ml.utils.deploy import ExportSpec
from autoware_ml.utils.point_cloud.batching import infer_batch_size_from_voxel_coords


def _voxels_data(batch_inputs: ModelBatchInputs) -> VoxelsData:
    """Read the voxels a pillar preprocessor added to the model inputs.

    Raises:
        ValueError: If the model inputs carry no voxels.
    """
    if batch_inputs.voxels_data is None:
        raise ValueError("CenterPoint needs the voxels of a PointPillarPreprocessor.")
    return batch_inputs.voxels_data


def _detection3d_gt_batch(batch_inputs: ModelBatchInputs) -> Detection3DGTBatch:
    """Read the ground truth boxes of the batch.

    Raises:
        ValueError: If the batch carries no detection ground truth.
    """
    gt_detections = batch_inputs.multi_task_gt_batch.detection3d_gt_batch
    if gt_detections is None:
        raise ValueError("CenterPoint losses need the 3D detection ground truth of the batch.")
    return gt_detections


class _CenterPointVoxelEncoderExportWrapper(nn.Module):
    """Export PointPillars PFN from decorated input features."""

    def __init__(self, voxel_encoder: nn.Module) -> None:
        """Initialize the voxel encoder export wrapper."""
        super().__init__()
        self.voxel_encoder = voxel_encoder

    def forward(self, input_features: torch.Tensor) -> torch.Tensor:
        """Encode decorated pillar features."""
        return self.voxel_encoder.encode_decorated(input_features)


class _CenterPointBackboneNeckHeadExportWrapper(nn.Module):
    """Export CenterPoint backbone, neck, and dense head from BEV features."""

    def __init__(self, backbone: nn.Module, neck: nn.Module, bbox_head: nn.Module) -> None:
        """Initialize the backbone-neck-head export wrapper."""
        super().__init__()
        self.backbone = backbone
        self.neck = neck
        self.bbox_head = bbox_head.prepare_for_export()
        self.output_names = ["heatmap", "reg", "height", "dim", "rot"]
        if self.bbox_head.use_velocity:
            self.output_names.append("vel")

    def forward(self, spatial_features: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Run CenterPoint BEV feature extraction and prediction heads."""
        bev_features = self.backbone(spatial_features)
        bev_features = self.neck(bev_features)
        outputs = self.bbox_head(bev_features)
        return outputs.export_tensors(self.output_names)


class CenterPointDetectionModel(BaseModel):
    """Compose a CenterPoint detector from reusable lidar detection modules.

    The wrapper wires together pillar encoding, BEV feature extraction, and the
    CenterPoint dense head inside the shared :class:`BaseModel` interface.
    """

    def __init__(
        self,
        pts_voxel_encoder: torch.nn.Module,
        pts_middle_encoder: torch.nn.Module,
        pts_backbone: torch.nn.Module,
        pts_neck: torch.nn.Module,
        bbox_head: torch.nn.Module,
        optimizer: Callable[..., Optimizer] | None = None,
        scheduler: Callable[[Optimizer], LRScheduler] | None = None,
        metrics: Sequence[MetricSuite] | None = None,
    ) -> None:
        """Initialize CenterPoint.

        Args:
            pts_voxel_encoder: Lidar voxel feature encoder.
            pts_middle_encoder: Sparse 3D or pillar-scatter middle encoder.
            pts_backbone: BEV backbone.
            pts_neck: BEV neck.
            bbox_head: CenterPoint dense detection head.
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

    def build_eval_output(self, batch: ModelBatchInputs, outputs: ModelOutputs) -> dict[str, Any]:
        """Decode detections and pair them with ground truth for metrics."""
        return detection_eval_output(self.predict_outputs(batch, outputs), batch)

    def forward(
        self,
        voxels: torch.Tensor,
        num_points: torch.Tensor,
        voxel_coords: torch.Tensor,
    ) -> ModelOutputs:
        """Run the detector on voxelized lidar inputs.

        Args:
            voxels: Voxel features.
            num_points: Number of points in each voxel.
            voxel_coords: Batched voxel coordinates.

        Returns:
            Detection head outputs.
        """
        batch_size = infer_batch_size_from_voxel_coords(voxel_coords)
        point_features = self.pts_voxel_encoder(voxels, num_points, voxel_coords)
        bev_features = self.pts_middle_encoder(point_features, voxel_coords, batch_size=batch_size)
        bev_features = self.pts_backbone(bev_features)
        bev_features = self.pts_neck(bev_features)
        return ModelOutputs(
            detection3d_head_outputs=Detection3DHeadOutputs(
                center_head_outputs=self.bbox_head(bev_features), transfusion_head_outputs=None
            )
        )

    def forward_inputs(self, batch_inputs: ModelBatchInputs) -> dict[str, Any]:
        """Pick the pillars of the batch.

        Args:
            batch_inputs: Model inputs holding the voxelized point cloud.

        Returns:
            The padded pillars, their point counts and their ``[batch, z, y, x]`` coordinates.
        """
        voxels_data = _voxels_data(batch_inputs)
        return {
            "voxels": voxels_data.voxels,
            "num_points": voxels_data.num_points,
            "voxel_coords": voxels_data.batch_zyx_coords(),
        }

    def compute_metrics(
        self,
        batch_inputs: ModelBatchInputs,
        outputs: ModelOutputs,
    ) -> dict[str, torch.Tensor]:
        """Compute CenterPoint training losses."""
        gt_detections = _detection3d_gt_batch(batch_inputs)
        return self.bbox_head.loss(
            outputs.detection3d().center_head(),
            gt_detections.valid_bboxes_3d(),
            gt_detections.valid_labels_3d(),
        )

    def predict_outputs(
        self, batch_inputs: ModelBatchInputs, outputs: ModelOutputs
    ) -> ModelPredictions:
        """Decode predictions for inference."""
        del batch_inputs
        return ModelPredictions(
            detection3d_predictions=self.bbox_head.predict(outputs.detection3d().center_head())
        )

    def build_export_spec(self, batch_inputs: ModelBatchInputs) -> ExportSpec:
        """Reject single-module CenterPoint deployment export."""
        del batch_inputs
        raise RuntimeError("CenterPoint deployment uses split modules; call build_export_specs().")

    def build_export_specs(self, batch_inputs: ModelBatchInputs) -> dict[str, ExportSpec]:
        """Build split CenterPoint deployment export specifications.

        The exported ABI follows the original CenterPoint deployment split:
        decorated pillar features feed the PFN ONNX module, and dense BEV
        spatial features feed the backbone/neck/head ONNX module. Scatter is a
        runtime preprocessing step between the two exported modules.
        """
        inputs = self.forward_inputs(batch_inputs)
        voxel_coords = inputs["voxel_coords"]
        batch_size = infer_batch_size_from_voxel_coords(voxel_coords)
        with torch.no_grad():
            input_features = self.pts_voxel_encoder.decorate(
                inputs["voxels"], inputs["num_points"], voxel_coords
            )
            pillar_features = self.pts_voxel_encoder.encode_decorated(input_features).squeeze(1)
            spatial_features = self.pts_middle_encoder(
                pillar_features, voxel_coords, batch_size=batch_size
            )

        head_wrapper = _CenterPointBackboneNeckHeadExportWrapper(
            self.pts_backbone,
            self.pts_neck,
            self.bbox_head,
        )
        return {
            "pts_voxel_encoder_centerpoint": ExportSpec(
                module=_CenterPointVoxelEncoderExportWrapper(self.pts_voxel_encoder),
                args=(input_features,),
                input_param_names=["input_features"],
                output_names=["pillar_features"],
            ),
            "pts_backbone_neck_head_centerpoint": ExportSpec(
                module=head_wrapper,
                args=(spatial_features,),
                input_param_names=["spatial_features"],
                output_names=head_wrapper.output_names,
            ),
        }
