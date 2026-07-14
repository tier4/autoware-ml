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

from collections.abc import Callable, Mapping
from typing import Any

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from autoware_ml.metrics.base import MetricSuite
from autoware_ml.metrics.detection3d.eval_output import detection_eval_output
from autoware_ml.models.base import BaseModel
from autoware_ml.utils.deploy import ExportSpec
from autoware_ml.utils.point_cloud.batching import infer_batch_size_from_voxel_coords


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
        return tuple(outputs[name] for name in self.output_names)


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
        metrics: list[MetricSuite] | None = None,
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

    def build_eval_output(self, batch: Mapping[str, Any], outputs: Any) -> dict[str, Any]:
        """Decode detections and pair them with ground truth for metrics."""
        return detection_eval_output(self.bbox_head.predict(outputs), batch)

    def _forward_with_batch_size(
        self,
        voxels: torch.Tensor,
        num_points: torch.Tensor,
        voxel_coords: torch.Tensor,
        batch_size: int | None = None,
    ) -> dict[str, torch.Tensor]:
        """Run the lidar backbone and CenterPoint head.

        Args:
            voxels: Voxel features.
            num_points: Number of points in each voxel.
            voxel_coords: Batched voxel coordinates.
            batch_size: Optional explicit batch size.

        Returns:
            Detection head outputs.
        """
        if batch_size is None:
            batch_size = infer_batch_size_from_voxel_coords(voxel_coords)
        point_features = self.pts_voxel_encoder(voxels, num_points, voxel_coords)
        bev_features = self.pts_middle_encoder(point_features, voxel_coords, batch_size=batch_size)
        bev_features = self.pts_backbone(bev_features)
        bev_features = self.pts_neck(bev_features)
        return self.bbox_head(bev_features)

    def forward(
        self,
        voxels: torch.Tensor,
        num_points: torch.Tensor,
        voxel_coords: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Run the detector on voxelized lidar inputs.

        Args:
            voxels: Voxel features.
            num_points: Number of points in each voxel.
            voxel_coords: Batched voxel coordinates.

        Returns:
            Detection head outputs.
        """
        return self._forward_with_batch_size(
            voxels=voxels, num_points=num_points, voxel_coords=voxel_coords
        )

    def compute_metrics(
        self,
        batch_inputs_dict: dict[str, Any],
        outputs: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Compute CenterPoint training losses."""
        return self.bbox_head.loss(
            outputs, batch_inputs_dict["gt_boxes"], batch_inputs_dict["gt_labels"]
        )

    def predict_outputs(
        self, batch_inputs_dict: dict[str, Any], outputs: dict[str, torch.Tensor]
    ) -> Any:
        """Decode predictions for inference."""
        del batch_inputs_dict
        return self.bbox_head.predict(outputs)

    def get_log_batch_size(self, batch_inputs_dict: dict[str, Any]) -> int | None:
        """Log the sample count instead of voxel count for lidar detection."""
        return len(batch_inputs_dict["gt_boxes"])

    def build_export_spec(self, batch_inputs_dict: Mapping[str, Any]) -> ExportSpec:
        """Reject single-module CenterPoint deployment export."""
        del batch_inputs_dict
        raise RuntimeError("CenterPoint deployment uses split modules; call build_export_specs().")

    def build_export_specs(self, batch_inputs_dict: Mapping[str, Any]) -> dict[str, ExportSpec]:
        """Build split CenterPoint deployment export specifications.

        The exported ABI follows the original CenterPoint deployment split:
        decorated pillar features feed the PFN ONNX module, and dense BEV
        spatial features feed the backbone/neck/head ONNX module. Scatter is a
        runtime preprocessing step between the two exported modules.
        """
        batch_size = infer_batch_size_from_voxel_coords(batch_inputs_dict["voxel_coords"])
        with torch.no_grad():
            input_features = self.pts_voxel_encoder.decorate(
                batch_inputs_dict["voxels"],
                batch_inputs_dict["num_points"],
                batch_inputs_dict["voxel_coords"],
            )
            pillar_features = self.pts_voxel_encoder.encode_decorated(input_features).squeeze(1)
            spatial_features = self.pts_middle_encoder(
                pillar_features,
                batch_inputs_dict["voxel_coords"],
                batch_size=batch_size,
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
