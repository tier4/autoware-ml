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

"""PTv3 joint segmentation and detection model.

One shared PTv3 encoder feeds the segmentation decoder head and the detection
BEV neck. Frames without any ground-truth box contribute no detection loss
and neutral detection metric entries, so annotation sources without detection
labels (e.g. segmentation ground truth mixed in for rehearsal, or a GT
segmentation validation split) train and evaluate only the segmentation
branch. The deliberate trade-off: genuinely empty scenes also provide no
pure-background detection supervision. The same class supports training
(forward / compute_metrics) and ONNX export (build_export_spec).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from typing import Any

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from autoware_ml.dataclasses.batch.detection3d import Detection3DGTBatch
from autoware_ml.dataclasses.models.detection3d.head_outputs import Detection3DHeadOutputs
from autoware_ml.dataclasses.models.detection3d.predictions import Detection3DSamplePredictions
from autoware_ml.dataclasses.models.model_batch_inputs import ModelBatchInputs
from autoware_ml.dataclasses.models.model_outputs import ModelOutputs
from autoware_ml.dataclasses.models.model_predictions import ModelPredictions
from autoware_ml.dataclasses.models.segmentation3d.head_outputs import Segmentation3DHeadOutputs
from autoware_ml.metrics.detection3d.eval_output import detection_eval_output
from autoware_ml.models.detection3d.ptv3 import PTv3DetBEVNeck, build_det_head_export_spec
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


class PTv3SegDetModel(PTv3BaseModel):
    """PTv3 joint segmentation and detection model.

    Training: call forward() then compute_metrics().
    Export: call build_export_spec() - requires export_output_names, grid_size,
    and point_cloud_range to be provided at construction time.
    """

    def __init__(
        self,
        encoder: PointTransformerV3Encoder,
        voxel_encoder: VoxelFeatureEncoder,
        seg3d_head: PTv3SegDecoderHead,
        bev_neck: PTv3DetBEVNeck,
        bbox_head: nn.Module,
        segmentation_loss_weight: float = 1.0,
        detection_loss_weight: float = 1.0,
        export_output_names: Sequence[str] | None = None,
        grid_size: float | None = None,
        point_cloud_range: Sequence[float] | None = None,
        optimizer: Callable[..., Optimizer] | None = None,
        scheduler: Callable[[Optimizer], LRScheduler] | None = None,
        optimizer_group_overrides: Mapping[str, Mapping[str, Any]] | None = None,
        scheduler_config: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the PTv3 joint segmentation and detection model.

        Args:
            encoder: PTv3 encoder module shared by both branches.
            voxel_encoder: Voxel feature encoder feeding the embedding stem.
            seg3d_head: Segmentation decoder head owning losses and the
                classifier.
            bev_neck: Detection BEV neck consuming the encoder pooling chain.
            bbox_head: Detection head producing the decoded predictions.
            segmentation_loss_weight: Weight of the segmentation loss term.
            detection_loss_weight: Weight of the detection loss term.
            export_output_names: Ordered output names used during export.
            grid_size: Voxel grid size used to derive sparse shape for export.
            point_cloud_range: Point-cloud range used to derive sparse shape
                for export.
            optimizer: Optimizer factory.
            scheduler: Scheduler factory.
            optimizer_group_overrides: Per-group optimizer overrides.
            scheduler_config: Lightning scheduler metadata.
            **kwargs: Keyword arguments forwarded up the MRO chain.
        """
        super().__init__(
            encoder=encoder,
            voxel_encoder=voxel_encoder,
            grid_size=grid_size,
            point_cloud_range=point_cloud_range,
            optimizer=optimizer,
            scheduler=scheduler,
            optimizer_group_overrides=optimizer_group_overrides,
            scheduler_config=scheduler_config,
            **kwargs,
        )
        self.seg3d_head = seg3d_head
        self.bev_neck = bev_neck
        self.bbox_head = bbox_head
        self.segmentation_loss_weight = float(segmentation_loss_weight)
        self.detection_loss_weight = float(detection_loss_weight)
        self._export_output_names = (
            list(export_output_names) if export_output_names is not None else None
        )

    def build_optimizer_groups(self) -> Mapping[str, Sequence[torch.nn.Parameter]]:
        """Group pretrained and newly initialized joint-task parameters."""
        encoder_default_params, encoder_block_params = split_block_parameters(self.encoder)
        seg3d_head_params = [
            parameter for parameter in self.seg3d_head.parameters() if parameter.requires_grad
        ]
        det3d_branch_params = [
            parameter
            for module in (self.bev_neck, self.bbox_head)
            for parameter in module.parameters()
            if parameter.requires_grad
        ]
        return {
            "encoder_default": encoder_default_params,
            "encoder_block": encoder_block_params,
            "seg3d_head": seg3d_head_params,
            "det3d_branch": det3d_branch_params,
        }

    def forward_network(
        self,
        voxels: torch.Tensor,
        num_points: torch.Tensor,
        voxel_coords: torch.Tensor,
        num_dropped_voxels: torch.Tensor,
        time_lag_column: int,
    ) -> ModelOutputs:
        """Run one shared PTv3 encoder pass and branch into both heads.

        Args:
            voxels: Padded voxel points from the data preprocessing.
            num_points: Valid point count per voxel.
            voxel_coords: Voxel coordinates with a leading batch column.
            num_dropped_voxels: Occupied voxels the voxelizer discarded.
            time_lag_column: Point column holding the time lag.
        """
        point = self.encode(voxels, num_points, voxel_coords, num_dropped_voxels, time_lag_column)
        # The BEV neck must read the encoder chain before the segmentation
        # decoder: SerializedUnpooling pops the chain and overwrites parent
        # features in place.
        bev_features = self.bev_neck(point)
        seg_logits = self.seg3d_head(point)
        det_outputs = self.bbox_head(bev_features)
        return ModelOutputs(
            detection3d_head_outputs=Detection3DHeadOutputs(
                center_head_outputs=None, transfusion_head_outputs=det_outputs
            ),
            segmentation3d_head_outputs=Segmentation3DHeadOutputs(logits=seg_logits),
        )

    @staticmethod
    def _detection_gt_batch(batch_inputs: ModelBatchInputs) -> Detection3DGTBatch:
        """Read the detection ground truth of the batch.

        Args:
            batch_inputs: Model inputs of the batch.

        Returns:
            The detection ground truth.

        Raises:
            ValueError: If the batch carries no detection ground truth.
        """
        gt_detections = batch_inputs.multi_task_gt_batch.detection3d_gt_batch
        if gt_detections is None:
            raise ValueError("PTv3 seg det needs the 3D detection ground truth of the batch.")
        return gt_detections

    @staticmethod
    def _detection_frame_mask(gt_detections: Detection3DGTBatch) -> torch.Tensor:
        """Return the per-frame detection supervision mask.

        Supervision is carried by the annotations themselves: a frame with no
        ground-truth box contributes no detection supervision. This includes
        genuinely empty scenes, which therefore provide no pure-background
        signal - the deliberate price of not carrying a separate flag.

        Args:
            gt_detections: Detection ground truth of the batch.

        Returns:
            Boolean tensor of shape ``(batch_size,)``.
        """
        return gt_detections.gt_valid_bboxes > 0

    @staticmethod
    def _mask_list(values: Sequence[Any], mask: torch.Tensor) -> list[Any]:
        """Select the flagged entries from a per-sample list."""
        return [value for value, flagged in zip(values, mask.tolist()) if flagged]

    def compute_metrics(
        self,
        batch_inputs: ModelBatchInputs,
        outputs: ModelOutputs,
    ) -> dict[str, torch.Tensor]:
        """Compute combined segmentation and detection losses.

        The detection loss runs only on frames that carry ground-truth boxes;
        on unlabeled frames, empty ground truth would turn every real object
        into a hard negative.
        """
        seg_logits = outputs.segmentation3d().logits
        det_outputs = outputs.detection3d().transfusion_head()
        seg_metrics = segmentation_point_loss(self.seg3d_head, seg_logits, batch_inputs)

        gt_detections = self._detection_gt_batch(batch_inputs)
        det_mask = self._detection_frame_mask(gt_detections)
        if bool(det_mask.any()):
            traffic_cone_status = gt_detections.gt_traffic_cone_barrier_bbox_status
            det_metrics = self.bbox_head.loss(
                outputs=Detection3DHeadOutputs(
                    center_head_outputs=None,
                    transfusion_head_outputs=det_outputs.select_samples(det_mask),
                ),
                gt_bboxes_3d=gt_detections.gt_bboxes_3d[det_mask],
                gt_labels_3d=gt_detections.gt_labels_3d[det_mask],
                gt_valid_bboxes=gt_detections.gt_valid_bboxes[det_mask],
                gt_traffic_cone_barrier_bbox_status=(
                    traffic_cone_status[det_mask] if traffic_cone_status is not None else None
                ),
            )
        else:
            # Keep the detection branch in the autograd graph with zero
            # gradients so DDP reducers see every parameter.
            branches = det_outputs.separate_head_outputs
            graph_tensors = [
                det_outputs.dense_heatmap,
                branches.heatmap,
                branches.center,
                branches.height,
                branches.dim,
                branches.rot,
            ]
            if branches.vel is not None:
                graph_tensors.append(branches.vel)
            zero_loss = sum(tensor.float().sum() for tensor in graph_tensors) * 0.0
            det_metrics = {"loss": zero_loss}

        seg_loss = seg_metrics["loss"]
        weighted_seg_loss = self.segmentation_loss_weight * seg_loss
        weighted_det_loss = self.detection_loss_weight * det_metrics["loss"]
        metrics: dict[str, torch.Tensor] = {
            "seg_loss_ce": seg_metrics["loss_ce"],
            "seg_loss_lovasz": seg_metrics["loss_lovasz"],
            "seg_loss": seg_loss,
            "weighted_seg_loss": weighted_seg_loss,
            "weighted_det_loss": weighted_det_loss,
            "loss": weighted_det_loss + weighted_seg_loss,
        }
        metrics.update({f"det_{name}": value for name, value in det_metrics.items()})
        return metrics

    def build_eval_output(self, batch: ModelBatchInputs, outputs: ModelOutputs) -> dict[str, Any]:
        """Produce the detection and the current frame segmentation eval data.

        Frames without detection supervision contribute empty predictions and
        their (already empty) ground truth instead of being dropped: the
        detection metric state must grow by exactly one entry per frame on
        every rank, or torchmetrics' per-element list-state ``all_gather``
        deadlocks under DDP when ranks see different seg/det frame mixes.
        Empty prediction + empty ground truth is metric-neutral.
        """
        det_mask = self._detection_frame_mask(self._detection_gt_batch(batch))
        predictions = list(
            self.bbox_head.decode_outputs(outputs=outputs.detection3d()).detection3d_predictions
        )
        predictions = [
            prediction
            if flagged
            else Detection3DSamplePredictions(
                bboxes_3d=prediction.bboxes_3d[:0],
                scores_3d=prediction.scores_3d[:0],
                labels_3d=prediction.labels_3d[:0],
            )
            for prediction, flagged in zip(predictions, det_mask.tolist())
        ]
        eval_out = detection_eval_output(
            ModelPredictions(detection3d_predictions=predictions), batch
        )
        eval_out.update(segmentation_eval_output(outputs.segmentation3d().logits, batch))
        return eval_out

    def decode_outputs(
        self, multi_task_batch_inputs: ModelBatchInputs, outputs: ModelOutputs
    ) -> ModelPredictions:
        """Decode the boxes and the point labels of the batch for inference.

        Args:
            multi_task_batch_inputs: Model inputs holding the grid samples the logits were
                predicted for.
            outputs: Raw outputs returned by :meth:`forward`.

        Returns:
            The decoded boxes of every sample and the label of every original point.
        """
        detection_predictions = self.bbox_head.decode_outputs(outputs=outputs.detection3d())
        return ModelPredictions(
            detection3d_predictions=detection_predictions.detection3d_predictions,
            segmentation3d_predictions=segmentation_predict_outputs(
                outputs.segmentation3d().logits, multi_task_batch_inputs
            ),
        )

    def get_export_output_names(self) -> list[str]:
        """Return configured ONNX export output names.

        Returns:
            Output names passed to the export spec.

        Raises:
            ValueError: If export output names were not configured.
        """
        if self._export_output_names is None:
            raise ValueError(
                "export_output_names must be provided at construction time to use export."
            )
        return list(self._export_output_names)

    def build_export_spec(self, batch_inputs: ModelBatchInputs) -> ExportSpec:
        """Build the ONNX export spec for joint PTv3 segmentation+detection.

        Args:
            batch_inputs: Example preprocessed batch with the voxels.

        Returns:
            Deployment export specification for the joint model.
        """
        if self.grid_size is None or self.point_cloud_range is None:
            raise ValueError(
                "grid_size and point_cloud_range must be provided at construction time to use "
                "export."
            )
        inputs = prepare_ptv3_export_inputs(self, batch_inputs)
        export_input_args, input_param_names = inputs.encoder_args(SERIALIZED_POOLING_FIELDS)
        export_module = _PTv3SegDetExportModule(
            encoder=self._prepare_encoder_export(),
            voxel_encoder=self.voxel_encoder,
            seg3d_head=self.seg3d_head.prepare_for_export(self.EXPORT_ORDER),
            bev_neck=deepcopy(self.bev_neck).eval(),
            bbox_head=self.bbox_head.prepare_for_export(),
            sparse_shape=inputs.sparse_shape,
            serialized_depth=inputs.serialization_depth,
            output_names=self.get_export_output_names(),
        )
        export_module.eval()
        output_names = self.get_export_output_names()
        dynamic_axes = build_ptv3_input_dynamic_axes(input_param_names)
        dynamic_axes.update(
            build_point_feature_dynamic_axes(
                tuple(name for name in output_names if name in {"pred_labels", "pred_probs"})
            )
        )
        return ExportSpec(
            module=export_module,
            args=export_input_args,
            input_param_names=input_param_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            supported_stages=self.EXPORT_SUPPORTED_STAGES,
        )

    def build_export_specs(self, batch_inputs: ModelBatchInputs) -> dict[str, ExportSpec]:
        """Build split PTv3 segdet ONNX export specs for encoder, seg head, and det head.

        Args:
            batch_inputs: Example preprocessed batch with the voxels.

        Returns:
            Export specs of the encoder, the segmentation head, and the detection head.
        """
        if self.grid_size is None or self.point_cloud_range is None:
            raise ValueError(
                "grid_size and point_cloud_range must be provided at construction time to use "
                "export."
            )
        context = build_ptv3_export_context(self, batch_inputs)
        det_output_names = [
            n for n in self.get_export_output_names() if n not in ("pred_labels", "pred_probs")
        ]
        return {
            "ptv3_encoder": build_encoder_export_spec(context),
            "ptv3_seg3d_head": build_seg_head_export_spec(
                context,
                self.seg3d_head.prepare_for_export(self.EXPORT_ORDER),
                ["pred_labels", "pred_probs"],
            ),
            "ptv3_det3d_head": build_det_head_export_spec(
                context,
                self.bev_neck,
                self.bbox_head.prepare_for_export(),
                det_output_names,
            ),
        }


class _PTv3SegDetExportModule(PTv3EncoderExportBase):
    """ONNX-exportable PTv3 segmentation+detection graph with baked sparse shape."""

    def __init__(
        self,
        encoder: PointTransformerV3Encoder,
        voxel_encoder: VoxelFeatureEncoder,
        seg3d_head: PTv3SegDecoderHead,
        bev_neck: PTv3DetBEVNeck,
        bbox_head: nn.Module,
        sparse_shape: torch.Tensor,
        serialized_depth: torch.Tensor,
        output_names: Sequence[str],
    ) -> None:
        super().__init__(encoder, voxel_encoder, sparse_shape, serialized_depth)
        self.seg3d_head = seg3d_head
        self.bev_neck = bev_neck
        self.bbox_head = bbox_head
        self.output_names = list(output_names)

    def forward(
        self,
        voxels: torch.Tensor,
        num_points_per_voxel: torch.Tensor,
        grid_coord: torch.Tensor,
        serialized_order: torch.Tensor,
        serialized_inverse: torch.Tensor,
        *serialized_pooling_inputs: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        """Run the export graph and return outputs in configured order.

        Args:
            voxels: Padded voxel points.
            num_points_per_voxel: Valid point count per voxel.
            grid_coord: Input voxel coordinates.
            serialized_order: Level-0 serialization order, one row per curve.
            serialized_inverse: Inverse of ``serialized_order``.
            serialized_pooling_inputs: Precomputed pooling metadata tensors.

        Returns:
            Tuple of export tensors ordered according to ``output_names``.
        """
        point = self.run_encoder(
            voxels,
            num_points_per_voxel,
            grid_coord,
            serialized_order,
            serialized_inverse,
            *serialized_pooling_inputs,
        )
        # BEV branch first: the segmentation decoder consumes the pooling
        # chain destructively.
        bev_features = self.bev_neck(point)
        det_outputs = self.bbox_head(bev_features)

        seg_logits = self.seg3d_head(point)
        pred_probs = torch.softmax(seg_logits, dim=1)
        pred_labels = pred_probs.argmax(dim=1)

        segmentation = {"pred_labels": pred_labels, "pred_probs": pred_probs}
        detection_names = [name for name in self.output_names if name not in segmentation]
        detection = dict(zip(detection_names, det_outputs.export_tensors(detection_names)))
        outputs = segmentation | detection
        return tuple(outputs[name] for name in self.output_names)
