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

"""Native TransFusion lidar detector wrappers.

This module provides the task-level training and export wrapper around the
reusable TransFusion detection components.
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


class _TransFusionExportWrapper(nn.Module):
    """Wrap TransFusion export with the deployment tensor contract.

    The wrapper keeps the exported forward signature tensor-only, fixes the
    batch size derived from the sample batch, and exposes only the deployment
    tensors expected by Autoware consumers.
    """

    def __init__(
        self,
        pts_voxel_encoder: torch.nn.Module,
        pts_middle_encoder: torch.nn.Module,
        pts_backbone: torch.nn.Module,
        pts_neck: torch.nn.Module,
        bbox_head: torch.nn.Module,
        batch_size: int,
    ) -> None:
        """Initialize the export wrapper.

        Args:
            pts_voxel_encoder: Lidar voxel feature encoder.
            pts_middle_encoder: Export-ready sparse or dense middle encoder.
            pts_backbone: BEV backbone.
            pts_neck: BEV neck.
            bbox_head: Export-ready TransFusion head.
            batch_size: Explicit export batch size.
        """
        super().__init__()
        self.pts_voxel_encoder = pts_voxel_encoder
        self.pts_middle_encoder = pts_middle_encoder
        self.pts_backbone = pts_backbone
        self.pts_neck = pts_neck
        self.bbox_head = bbox_head
        self.batch_size = batch_size

    def forward(
        self,
        voxels: torch.Tensor,
        num_points: torch.Tensor,
        coors: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run export-time inference and return deployment tensors.

        Args:
            voxels: Voxel features.
            num_points: Number of points in each voxel.
            coors: Batched voxel coordinates.

        Returns:
            ``cls_score0``, ``bbox_pred0``, and ``dir_cls_pred0`` tensors.
        """
        point_features = self.pts_voxel_encoder(voxels, num_points, coors)
        bev_features = self.pts_middle_encoder(point_features, coors, batch_size=self.batch_size)
        bev_features = self.pts_backbone(bev_features)
        bev_features = self.pts_neck(bev_features)
        outputs = self.bbox_head(bev_features)
        return _format_transfusion_export_outputs(outputs)


def _format_transfusion_export_outputs(
    outputs: Mapping[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Format TransFusion head outputs for deployment export."""
    cls_score0 = outputs["heatmap"].sigmoid() * outputs["query_heatmap_score"]
    bbox_pred0 = torch.cat(
        (outputs["center"], outputs["height"], outputs["dim"], outputs["vel"]),
        dim=1,
    )
    dir_cls_pred0 = outputs["rot"]

    if bbox_pred0.shape[1] != 8:
        raise ValueError(
            f"TransFusion export expects bbox_pred0 to have 8 channels, got {bbox_pred0.shape[1]}."
        )
    if dir_cls_pred0.shape[1] != 2:
        raise ValueError(
            "TransFusion export expects dir_cls_pred0 to have 2 channels, "
            f"got {dir_cls_pred0.shape[1]}."
        )
    return cls_score0, bbox_pred0, dir_cls_pred0


class TransFusionDetectionModel(BaseModel):
    """Compose a TransFusion lidar detector from reusable task modules.

    The wrapper wires together voxel encoding, BEV backbone processing, and the
    TransFusion query head inside the shared :class:`BaseModel` interface.
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
        """Initialize TransFusion.

        Args:
            pts_voxel_encoder: Lidar voxel feature encoder.
            pts_middle_encoder: Sparse 3D or pillar-scatter middle encoder.
            pts_backbone: BEV backbone.
            pts_neck: BEV neck.
            bbox_head: Detection head.
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
        """Run the lidar backbone and dense head.

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
        self, voxels: torch.Tensor, num_points: torch.Tensor, voxel_coords: torch.Tensor
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
        """Compute training losses for one detection batch.

        Args:
            batch_inputs_dict: Full batch dictionary.
            outputs: Raw head outputs returned by :meth:`forward`.

        Returns:
            Dictionary of loss terms produced by the detection head.
        """
        return self.bbox_head.loss(
            outputs, batch_inputs_dict["gt_boxes"], batch_inputs_dict["gt_labels"]
        )

    def predict_outputs(
        self, batch_inputs_dict: dict[str, Any], outputs: dict[str, torch.Tensor]
    ) -> Any:
        """Decode predictions for inference.

        Args:
            batch_inputs_dict: Full batch dictionary.
            outputs: Raw head outputs returned by :meth:`forward`.

        Returns:
            Decoded detector predictions for the current batch.
        """
        del batch_inputs_dict
        return self.bbox_head.predict(outputs)

    def get_log_batch_size(self, batch_inputs_dict: dict[str, Any]) -> int | None:
        """Log the sample count instead of voxel count for lidar detection."""
        return len(batch_inputs_dict["gt_boxes"])

    def build_export_spec(self, batch_inputs_dict: dict[str, Any]) -> ExportSpec:
        """Build an export specification with explicit tensor inputs.

        Args:
            batch_inputs_dict: Preprocessed example batch used to derive export inputs.

        Returns:
            Export specification for ONNX and TensorRT deployment.
        """
        batch_size = infer_batch_size_from_voxel_coords(batch_inputs_dict["voxel_coords"])
        pts_middle_encoder = self.pts_middle_encoder
        if hasattr(pts_middle_encoder, "prepare_for_export"):
            pts_middle_encoder = pts_middle_encoder.prepare_for_export()
        return ExportSpec(
            module=_TransFusionExportWrapper(
                pts_voxel_encoder=self.pts_voxel_encoder,
                pts_middle_encoder=pts_middle_encoder,
                pts_backbone=self.pts_backbone,
                pts_neck=self.pts_neck,
                bbox_head=self.bbox_head.prepare_for_export(),
                batch_size=batch_size,
            ),
            args=(
                batch_inputs_dict["voxels"],
                batch_inputs_dict["num_points"],
                batch_inputs_dict["voxel_coords"],
            ),
            input_param_names=["voxels", "num_points", "coors"],
            output_names=["cls_score0", "bbox_pred0", "dir_cls_pred0"],
        )
