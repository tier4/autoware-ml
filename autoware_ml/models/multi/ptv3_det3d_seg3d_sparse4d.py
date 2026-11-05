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

"""PTv3 joint segmentation and Sparse4D detection model.

A temporal head needs scenario streams, see ``DataModule(streaming=True)``. Its export
carries the instance bank state through the inputs and the outputs of the detection graph.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch

from autoware_ml.dataclasses.models.detection3d.head_outputs import Sparse4DHeadOutputs
from autoware_ml.dataclasses.models.model_batch_inputs import ModelBatchInputs
from autoware_ml.dataclasses.models.model_outputs import ModelOutputs
from autoware_ml.dataclasses.models.model_predictions import ModelPredictions
from autoware_ml.dataclasses.models.segmentation3d.head_outputs import Segmentation3DHeadOutputs
from autoware_ml.models.detection3d.heads.sparse4d import (
    Sparse4DHead,
    Sparse4DTargets,
    SparseStage,
)
from autoware_ml.models.detection3d.ptv3 import PTv3Det3DFeaturePyramid
from autoware_ml.models.detection3d.ptv3_sparse4d import (
    PTv3Sparse4DHeadExportModule,
    build_sparse4d_head_export_spec,
    log_sparse4d_step_metrics,
    run_sparse4d_head,
    sparse4d_detection_loss,
    sparse4d_eval_output,
    sparse4d_frame_meta,
    sparse4d_head_outputs,
    validate_sampled_levels,
)
from autoware_ml.models.module_base_model import LogDictConfigs
from autoware_ml.models.segmentation3d.encoders.ptv3 import PointTransformerV3Encoder
from autoware_ml.models.segmentation3d.encoders.voxel import VoxelFeatureEncoder
from autoware_ml.models.segmentation3d.heads.ptv3 import (
    PTv3Seg3DDecoderHead,
    segmentation3d_eval_output,
    segmentation3d_point_loss,
    segmentation3d_predict_outputs,
)
from autoware_ml.models.segmentation3d.ptv3_base import (
    PTv3BaseModel,
    PTv3ExportContext,
    build_encoder_export_spec,
    build_ptv3_export_context,
    build_seg3d_head_export_spec,
    split_block_parameters,
)
from autoware_ml.utils.deploy import ExportSpec


class PTv3Det3DSeg3DSparse4DModel(PTv3BaseModel):
    """PTv3 joint segmentation and Sparse4D detection model."""

    def __init__(
        self,
        encoder: PointTransformerV3Encoder,
        voxel_encoder: VoxelFeatureEncoder,
        seg3d_head: PTv3Seg3DDecoderHead,
        det_fusion: PTv3Det3DFeaturePyramid,
        bbox_head: Sparse4DHead,
        log_dict_configs: LogDictConfigs,
        segmentation3d_loss_weight: float = 1.0,
        detection3d_loss_weight: float = 1.0,
        grid_size: float | None = None,
        point_cloud_range: Sequence[float] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the PTv3 joint segmentation and Sparse4D detection model.

        Args:
            encoder: PTv3 encoder module shared by both branches.
            voxel_encoder: Voxel feature encoder feeding the embedding stem.
            seg3d_head: Segmentation decoder head owning losses and the classifier.
            det_fusion: Top-down fusion of the deepest encoder stage into the finer stages,
                whose sparse voxels the detection head samples.
            bbox_head: Sparse4D query head.
            log_dict_configs: Logging settings of the step metrics.
            segmentation3d_loss_weight: Weight of the segmentation loss term.
            detection3d_loss_weight: Weight of the detection loss term.
            grid_size: Voxel grid size of the encoder input.
            point_cloud_range: Point-cloud range of the encoder input.
            **kwargs: Data preprocessor, optimizer, scheduler and metrics of the base model.

        Raises:
            ValueError: If the pyramid and the head disagree on the sampled levels.
        """
        super().__init__(
            encoder=encoder,
            voxel_encoder=voxel_encoder,
            grid_size=grid_size,
            point_cloud_range=point_cloud_range,
            log_dict_configs=log_dict_configs,
            **kwargs,
        )
        validate_sampled_levels(det_fusion, bbox_head)
        self.seg3d_head = seg3d_head
        self.det_fusion = det_fusion
        self.bbox_head = bbox_head
        self.segmentation3d_loss_weight = float(segmentation3d_loss_weight)
        self.detection3d_loss_weight = float(detection3d_loss_weight)

    def build_optimizer_groups(self) -> Mapping[str, Sequence[torch.nn.Parameter]]:
        """Group the pretrained encoder parameters apart from the newly initialized heads."""
        encoder_default_params, encoder_block_params = split_block_parameters(self.encoder)
        return {
            "encoder_default": encoder_default_params,
            "encoder_block": encoder_block_params,
            "seg3d_head": [p for p in self.seg3d_head.parameters() if p.requires_grad],
            "det3d_branch": [
                parameter
                for module in (self.det_fusion, self.bbox_head)
                for parameter in module.parameters()
                if parameter.requires_grad
            ],
        }

    def forward(self, multi_task_batch_inputs: ModelBatchInputs) -> ModelOutputs:
        """Run one shared PTv3 encoder pass and branch into both heads.

        Args:
            multi_task_batch_inputs: Model inputs with the voxels, the frame metadata and the
                detection ground truth.

        Returns:
            The Sparse4D head outputs and the segmentation logits.
        """
        _, det3d_outputs, seg3d_logits = self.forward_branches(multi_task_batch_inputs)
        return ModelOutputs(
            detection3d_head_outputs=sparse4d_head_outputs(det3d_outputs),
            segmentation3d_head_outputs=Segmentation3DHeadOutputs(logits=seg3d_logits),
        )

    def forward_branches(
        self, multi_task_batch_inputs: ModelBatchInputs
    ) -> tuple[list[SparseStage], Sparse4DHeadOutputs, torch.Tensor]:
        """Encode the batch once and run the detection and the segmentation heads.

        The frame metadata places every frame of a stream for the temporal head, the boxes feed
        the denoising queries and their instance ids and point counts the tracks.

        Args:
            multi_task_batch_inputs: Model inputs with the voxels, the frame metadata and the
                detection ground truth.

        Returns:
            The fused stages the detection head sampled, its outputs and the segmentation logits.

        Raises:
            ValueError: If the batch carries no frame metadata.
        """
        frame_meta = sparse4d_frame_meta(multi_task_batch_inputs)
        point = self.encode_batch(multi_task_batch_inputs)
        # Fuse before the decoder, SerializedUnpooling pops the encoder chain
        det3d_stages = self.det_fusion(point)
        seg3d_logits = self.seg3d_head(point)
        det3d_outputs = run_sparse4d_head(
            self.bbox_head,
            det3d_stages,
            multi_task_batch_inputs,
            frame_meta,
            self.head_context(multi_task_batch_inputs),
        )
        return det3d_stages, det3d_outputs, seg3d_logits

    def head_context(self, multi_task_batch_inputs: ModelBatchInputs) -> Any:
        """Frame context of the detection head, none for the lidar model."""
        return None

    def compute_metrics(
        self,
        multi_task_batch_inputs: ModelBatchInputs,
        multi_task_outputs: ModelOutputs,
    ) -> dict[str, torch.Tensor]:
        """Compute combined segmentation and detection losses.

        Frames without ground-truth boxes and samples padding a streaming lane skip the
        detection loss.
        """
        metrics, _, _ = self.detection_segmentation_metrics(
            multi_task_batch_inputs, multi_task_outputs
        )
        log_sparse4d_step_metrics(self, metrics, multi_task_batch_inputs)
        return metrics

    def detection_segmentation_metrics(
        self,
        multi_task_batch_inputs: ModelBatchInputs,
        multi_task_outputs: ModelOutputs,
    ) -> tuple[dict[str, torch.Tensor], list[Sparse4DTargets] | None, torch.Tensor]:
        """Compute the segmentation and the detection losses without logging them.

        Args:
            multi_task_batch_inputs: Model inputs with the ground truth.
            multi_task_outputs: Raw outputs from :meth:`forward`.

        Returns:
            The weighted losses, the detection targets of every decoder layer over the frames
            with boxes (None when no frame has boxes) and the mask of those frames.
        """
        seg3d_logits = multi_task_outputs.segmentation3d().logits
        det3d_outputs = multi_task_outputs.detection3d().sparse4d_head()
        seg3d_metrics = segmentation3d_point_loss(
            self.seg3d_head, seg3d_logits, multi_task_batch_inputs
        )

        det3d_metrics, det3d_targets, det3d_mask = sparse4d_detection_loss(
            self.bbox_head, det3d_outputs, multi_task_batch_inputs
        )

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
        return metrics, det3d_targets, det3d_mask

    def build_eval_output(self, batch: ModelBatchInputs, outputs: ModelOutputs) -> dict[str, Any]:
        """Produce the detection and the current frame segmentation eval data.

        Frames without detection supervision contribute empty predictions. A temporal head adds
        which boxes were observed in the stream so far and the stream of every frame.
        """
        eval_out = sparse4d_eval_output(self.bbox_head, batch, outputs)
        eval_out.update(segmentation3d_eval_output(outputs.segmentation3d().logits, batch))
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
        return ModelPredictions(
            detection3d_predictions=self.bbox_head.decode_outputs(
                outputs.detection3d()
            ).detection3d_predictions,
            segmentation3d_predictions=segmentation3d_predict_outputs(
                outputs.segmentation3d().logits, multi_task_batch_inputs
            ),
        )

    def build_export_specs(
        self, multi_task_batch_inputs: ModelBatchInputs
    ) -> dict[str, ExportSpec]:
        """Build the split ONNX export specs of the encoder, the seg head and the det head.

        Args:
            multi_task_batch_inputs: Example preprocessed batch with the voxels.

        Returns:
            Export specs of the encoder, the segmentation head and the temporal detection head.

        Raises:
            ValueError: If the grid size or the point cloud range is missing.
        """
        if self.grid_size is None or self.point_cloud_range is None:
            raise ValueError(
                "grid_size and point_cloud_range must be provided at construction time to use "
                "export."
            )
        context = build_ptv3_export_context(self, multi_task_batch_inputs)
        return {
            "ptv3_lidar_encoder": build_encoder_export_spec(context),
            "ptv3_seg3d_head": build_seg3d_head_export_spec(
                context,
                self.seg3d_head.prepare_for_export(self.EXPORT_ORDER),
                ["pred_labels", "pred_probs"],
            ),
            "ptv3_det3d_head": self.build_det3d_head_export_spec(context, multi_task_batch_inputs),
        }

    def build_det3d_head_export_spec(
        self, context: PTv3ExportContext, multi_task_batch_inputs: ModelBatchInputs
    ) -> ExportSpec:
        """Build the temporal detection head export spec.

        Args:
            context: Shared export context.
            multi_task_batch_inputs: Example preprocessed batch, read for a head context.

        Returns:
            Export spec of the detection graph over the fused stages and the bank state.
        """
        return build_sparse4d_head_export_spec(
            context,
            PTv3Sparse4DHeadExportModule(self.det_fusion, self.bbox_head.prepare_for_export()),
        )
