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
from types import MappingProxyType
from typing import Any

from jaxtyping import Float32
import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from autoware_ml.dataclasses.models.model_batch_inputs import ModelBatchInputs
from autoware_ml.dataclasses.models.model_predictions import ModelPredictions
from autoware_ml.dataclasses.models.model_outputs import ModelOutputs
from autoware_ml.dataclasses.models.detection3d.head_outputs import Detection3DHeadOutputs
from autoware_ml.metrics.base import MetricSuite
from autoware_ml.metrics.detection3d.eval_output import detection_eval_output
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

    # ONNX tensor name -> CenterHeadOutputs field, in the order the deployed runtime
    # expects. Deriving `output_names` and the returned tuple from one table keeps
    # them from drifting apart. Do not reorder.
    _OUTPUT_FIELDS: tuple[tuple[str, str], ...] = (
        ("heatmap", "heatmap"),
        ("reg", "reg"),
        ("height", "height"),
        ("dim", "dim"),
        ("rot", "rot"),
    )
    _VELOCITY_FIELD: tuple[str, str] = ("vel", "vel")

    def __init__(self, backbone: nn.Module, neck: nn.Module, bbox_head: CenterHead) -> None:
        """Initialize the backbone-neck-head export wrapper.

        Args:
            backbone: BEV backbone.
            neck: BEV neck.
            bbox_head: CenterPoint dense head; an export-ready copy is taken.
        """
        super().__init__()
        self.backbone = backbone
        self.neck = neck
        self.bbox_head = bbox_head.prepare_for_export()
        self._output_fields = self._OUTPUT_FIELDS
        if self.bbox_head.use_velocity:
            self._output_fields += (self._VELOCITY_FIELD,)
        self.output_names = [onnx_name for onnx_name, _ in self._output_fields]

    def forward(
        self,
        spatial_features: Float32[torch.Tensor, "batch_size num_channels height width"],
    ) -> tuple[torch.Tensor, ...]:
        """Run BEV feature extraction and flatten the head outputs for export."""
        bev_features = self.backbone(spatial_features)
        bev_features = self.neck(bev_features)
        center_head_outputs = self.bbox_head(bev_features)

        export_outputs: list[torch.Tensor] = []
        for onnx_name, field_name in self._output_fields:
            output = getattr(center_head_outputs, field_name)
            if output is None:
                raise ValueError(f"CenterHead produced no tensor for export output '{onnx_name}'.")
            export_outputs.append(output)
        return tuple(export_outputs)


class CenterPointDetectionModel(ModuleBaseModel):
    """Compose a CenterPoint detector from reusable lidar detection modules.

    The wrapper wires together pillar encoding, BEV feature extraction, and the
    CenterPoint dense head inside the shared :class:`BaseModel` interface.
    """

    def __init__(
        self,
        data_preprocessor: DataPreprocessor,
        # TODO(KokSeang): Encoder and middle_encoder should be standardized to a common interface for all voxel encoders
        pts_voxel_encoder: PillarFeatureNet,
        pts_middle_encoder: PointPillarsScatter,
        pts_backbone: torch.nn.Module,
        pts_neck: torch.nn.Module,
        # TODO(KokSeang): bbox_head should be standardized to a common interface for all detection heads
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

    # TODO(KokSeang): This signature is temporary different from the base class,
    # and will be refactored to match the base class signature once the detection metric is refactored
    # to accept ModelPredictions and MultiTaskFeatures directly.
    def build_eval_output(  # type: ignore[override]
        self, batch: ModelBatchInputs, outputs: ModelOutputs
    ) -> dict[str, Any]:
        """Decode detections and pair them with ground truth for metrics."""
        if outputs.detection3d_head_outputs is None:
            raise ValueError(
                "ModelOutputs must contain detection3d_head_outputs for CenterPoint build_eval_output pass."
            )

        return detection_eval_output(
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

        batch_size = multi_task_batch_inputs.multi_task_gt_batch.infer_batch_size()
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

    def build_export_spec(self, multi_task_batch_inputs: ModelBatchInputs) -> ExportSpec:
        """Reject single-module CenterPoint deployment export."""
        del multi_task_batch_inputs
        raise RuntimeError("CenterPoint deployment uses split modules; call build_export_specs().")

    def build_export_specs(
        self, multi_task_batch_inputs: ModelBatchInputs
    ) -> dict[str, ExportSpec]:
        """Build split CenterPoint deployment export specifications.

        The exported ABI follows the original CenterPoint deployment split:
        decorated pillar features feed the PFN ONNX module, and dense BEV
        spatial features feed the backbone/neck/head ONNX module. Scatter is a
        runtime preprocessing step between the two exported modules.

        Args:
            multi_task_batch_inputs: Example preprocessed batch used to trace both modules.

        Returns:
            Ordered mapping of module name to export specification.
        """
        voxels_data = multi_task_batch_inputs.voxels_data
        if voxels_data is None:
            raise ValueError(
                "ModelBatchInputs must contain voxels_data to build CenterPoint export specs."
            )

        # The scatter canvas is indexed with the pillar batch indices, so the batch size
        # is derived from the voxels rather than from the ground truth: a batch prepared
        # for deployment may carry no ground truth at all.
        batch_size = (
            int(voxels_data.batch_indices.max().item()) + 1
            if voxels_data.batch_indices.numel()
            else 1
        )

        # Tracing must not update the pillar encoder's BatchNorm running statistics.
        was_training = self.training
        self.eval()
        try:
            with torch.no_grad():
                input_features = self.pts_voxel_encoder.encode(voxels_data)
                pillar_features = self.pts_voxel_encoder.encode_decorated(input_features).squeeze(1)
                spatial_features = self.pts_middle_encoder(
                    pillar_features=pillar_features,
                    coords=voxels_data.coords,
                    batch_indices=voxels_data.batch_indices,
                    batch_size=batch_size,
                )
        finally:
            self.train(was_training)

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
