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
(forward / compute_metrics) and deployment (deployment_graphs).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from autoware_ml.dataclasses.models.detection3d.head_outputs import Detection3DHeadOutputs
from autoware_ml.dataclasses.models.model_batch_inputs import ModelBatchInputs
from autoware_ml.dataclasses.models.model_outputs import ModelOutputs
from autoware_ml.dataclasses.models.model_predictions import ModelPredictions
from autoware_ml.dataclasses.models.segmentation3d.head_outputs import Segmentation3DHeadOutputs
from autoware_ml.deployment.graphs import DeploymentGraph
from autoware_ml.metrics.detection3d.eval_output import detection3d_eval_output
from autoware_ml.metrics.eval_output import EvalOutput
from autoware_ml.models.detection3d.task_modules.partial_annotations import PartialAnnotations
from autoware_ml.models.detection3d.ptv3 import PTv3Det3DBEVNeck, PTv3Det3DHeadExportModule
from autoware_ml.models.multi.detection3d_frames import (
    detection3d_frame_mask,
    detection3d_gt_batch,
    empty_unsupervised_predictions,
)
from autoware_ml.models.segmentation3d.encoders.ptv3 import PointTransformerV3Encoder
from autoware_ml.models.segmentation3d.encoders.voxel import VoxelFeatureEncoder
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


class PTv3Det3DSeg3DModel(PTv3BaseModel):
    """PTv3 joint segmentation and detection model.

    Training: call forward() then compute_metrics().
    Deployment: deployment_graphs() - requires grid_size and point_cloud_range to be
    provided at construction time.
    """

    def __init__(
        self,
        encoder: PointTransformerV3Encoder,
        voxel_encoder: VoxelFeatureEncoder,
        seg3d_head: PTv3Seg3DDecoderHead,
        bev_neck: PTv3Det3DBEVNeck,
        bbox_head: nn.Module,
        log_dict_configs: LogDictConfigs,
        segmentation3d_loss_weight: float = 1.0,
        detection3d_loss_weight: float = 1.0,
        grid_size: float | None = None,
        point_cloud_range: Sequence[float] | None = None,
        optimizer: Callable[..., Optimizer] | None = None,
        scheduler: Callable[[Optimizer], LRScheduler] | None = None,
        optimizer_group_overrides: Mapping[str, Mapping[str, Any]] | None = None,
        scheduler_config: Mapping[str, Any] | None = None,
        partial_annotations: PartialAnnotations | None = None,
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
            log_dict_configs: Logging configuration for training and validation.
            segmentation3d_loss_weight: Weight of the segmentation loss term.
            detection3d_loss_weight: Weight of the detection loss term.
            grid_size: Voxel grid size used to derive sparse shape for export.
            point_cloud_range: Point-cloud range used to derive sparse shape
                for export.
            optimizer: Optimizer factory.
            scheduler: Scheduler factory.
            optimizer_group_overrides: Per-group optimizer overrides.
            scheduler_config: Lightning scheduler metadata.
            partial_annotations: Maps the dataset's per-sample annotation flag to the classes
                the head must not supervise in that sample. None supervises every class.
            **kwargs: Keyword arguments forwarded up the MRO chain.
        """
        super().__init__(
            encoder=encoder,
            voxel_encoder=voxel_encoder,
            grid_size=grid_size,
            point_cloud_range=point_cloud_range,
            log_dict_configs=log_dict_configs,
            optimizer=optimizer,
            scheduler=scheduler,
            optimizer_group_overrides=optimizer_group_overrides,
            scheduler_config=scheduler_config,
            **kwargs,
        )
        self.seg3d_head = seg3d_head
        self.bev_neck = bev_neck
        self.bbox_head = bbox_head
        self.partial_annotations = partial_annotations
        self.segmentation3d_loss_weight = float(segmentation3d_loss_weight)
        self.detection3d_loss_weight = float(detection3d_loss_weight)

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

    def forward(self, multi_task_batch_inputs: ModelBatchInputs) -> ModelOutputs:
        """Run one shared PTv3 encoder pass and branch into both heads.

        Args:
            multi_task_batch_inputs: Model inputs holding the point cloud and its voxels.

        Returns:
            The outputs of the task heads.
        """
        point = self.encode_batch(multi_task_batch_inputs)
        # The BEV neck must read the encoder chain before the segmentation
        # decoder: SerializedUnpooling pops the chain and overwrites parent
        # features in place.
        bev_features = self.bev_neck(point)
        seg3d_logits = self.seg3d_head(point)
        det3d_outputs = self.bbox_head(bev_features)
        return ModelOutputs(
            detection3d_head_outputs=Detection3DHeadOutputs(
                center_head_outputs=None, transfusion_head_outputs=det3d_outputs
            ),
            segmentation3d_head_outputs=Segmentation3DHeadOutputs(logits=seg3d_logits),
        )

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
        seg3d_logits = outputs.segmentation3d().logits
        det3d_outputs = outputs.detection3d().transfusion_head()
        seg3d_metrics = segmentation3d_point_loss(self.seg3d_head, seg3d_logits, batch_inputs)

        gt_detections = detection3d_gt_batch(batch_inputs)
        det3d_mask = detection3d_frame_mask(gt_detections)
        if bool(det3d_mask.any()):
            traffic_cone_status = gt_detections.gt_traffic_cone_barrier_bbox_status
            det3d_metrics = self.bbox_head.loss(
                outputs=Detection3DHeadOutputs(
                    center_head_outputs=None,
                    transfusion_head_outputs=det3d_outputs.select_samples(det3d_mask),
                ),
                gt_bboxes_3d=gt_detections.gt_bboxes_3d[det3d_mask],
                gt_labels_3d=gt_detections.gt_labels_3d[det3d_mask],
                gt_valid_bboxes=gt_detections.gt_valid_bboxes[det3d_mask],
                gt_annotated_class_masks=(
                    self.partial_annotations.build_annotated_class_masks(
                        traffic_cone_status[det3d_mask] if traffic_cone_status is not None else None
                    )
                    if self.partial_annotations is not None
                    else None
                ),
            )
        else:
            # Keep the detection branch in the autograd graph with zero
            # gradients so DDP reducers see every parameter.
            branches = det3d_outputs.separate_head_outputs
            graph_tensors = [
                det3d_outputs.dense_heatmap,
                branches.heatmap,
                branches.center,
                branches.height,
                branches.dim,
                branches.rot,
            ]
            if branches.vel is not None:
                graph_tensors.append(branches.vel)
            zero_loss = sum(tensor.float().sum() for tensor in graph_tensors) * 0.0
            det3d_metrics = {"loss": zero_loss}

        seg3d_loss = seg3d_metrics["loss"]
        weighted_seg3d_loss = self.segmentation3d_loss_weight * seg3d_loss
        weighted_det3d_loss = self.detection3d_loss_weight * det3d_metrics["loss"]
        metrics: dict[str, torch.Tensor] = {
            "seg3d_loss_ce": seg3d_metrics["loss_ce"],
            "seg3d_loss_lovasz": seg3d_metrics["loss_lovasz"],
            "seg3d_loss": seg3d_loss,
            "weighted_seg3d_loss": weighted_seg3d_loss,
            "weighted_det3d_loss": weighted_det3d_loss,
            "loss": weighted_det3d_loss + weighted_seg3d_loss,
        }
        metrics.update({f"det3d_{name}": value for name, value in det3d_metrics.items()})
        return metrics

    def build_eval_output(self, batch: ModelBatchInputs, outputs: ModelOutputs) -> EvalOutput:
        """Produce the detection and the current frame segmentation eval data.

        Frames without detection supervision contribute empty predictions and
        their (already empty) ground truth instead of being dropped: the
        detection metric state must grow by exactly one entry per frame on
        every rank, or torchmetrics' per-element list-state ``all_gather``
        deadlocks under DDP when ranks see different seg/det frame mixes.
        Empty prediction + empty ground truth is metric-neutral.
        """
        det3d_mask = detection3d_frame_mask(detection3d_gt_batch(batch))
        predictions = empty_unsupervised_predictions(
            self.bbox_head.decode_outputs(outputs=outputs.detection3d()).detection3d_predictions,
            det3d_mask,
        )
        return EvalOutput(
            detection3d=detection3d_eval_output(
                ModelPredictions(detection3d_predictions=predictions), batch
            ),
            segmentation3d=segmentation3d_eval_output(outputs.segmentation3d().logits, batch),
        )

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
        detection3d_predictions = self.bbox_head.decode_outputs(outputs=outputs.detection3d())
        return ModelPredictions(
            detection3d_predictions=detection3d_predictions.detection3d_predictions,
            segmentation3d_predictions=segmentation3d_predict_outputs(
                outputs.segmentation3d().logits, multi_task_batch_inputs
            ),
        )

    def deployment_graphs(self) -> Sequence[DeploymentGraph]:
        """Declare ``lidar_encoder``, ``seg3d_head`` and ``det3d_head``.

        Returns:
            The three graphs; the model outputs are the seg head's, then the det head's.

        Raises:
            ValueError: If the geometry was not configured.
        """
        if self.grid_size is None or self.point_cloud_range is None:
            raise ValueError(
                "grid_size and point_cloud_range must be provided at construction time to deploy."
            )
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
            PTv3Det3DHeadExportModule(
                self.bev_neck, self.bbox_head, len(self.encoder.stride) + 1
            ).build_deployment_graph(),
        )
