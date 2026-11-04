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
from copy import deepcopy
from typing import Any

import torch
import torch.nn as nn

from autoware_ml.dataclasses.models.detection3d.head_outputs import (
    Detection3DHeadOutputs,
    Sparse4DHeadOutputs,
)
from autoware_ml.dataclasses.models.model_batch_inputs import ModelBatchInputs
from autoware_ml.dataclasses.models.model_outputs import ModelOutputs
from autoware_ml.dataclasses.models.model_predictions import ModelPredictions
from autoware_ml.dataclasses.models.segmentation3d.head_outputs import Segmentation3DHeadOutputs
from autoware_ml.metrics.detection3d.eval_output import detection3d_eval_output
from autoware_ml.models.detection3d.heads.sparse4d import (
    TEMPORAL_QUERY_INDEX_OUTPUT_NAME,
    TEMPORAL_STATE_INPUT_NAMES,
    TEMPORAL_STATE_OUTPUT_NAMES,
    Sparse4DHead,
    Sparse4DTargets,
    SparseQueryHead,
    SparseStage,
)
from autoware_ml.models.detection3d.ptv3 import (
    PTv3Det3DFeaturePyramid,
    det3d_head_export_args,
    det3d_head_export_dynamic_axes,
    det3d_head_export_input_names,
    rebuild_point_chain,
)
from autoware_ml.models.module_base_model import LogDictConfigs
from autoware_ml.models.multi.ptv3_det3d_seg3d import (
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
from autoware_ml.models.segmentation3d.ptv3_base import (
    PTv3BaseModel,
    PTv3ExportContext,
    build_encoder_export_spec,
    build_ptv3_export_context,
    build_seg3d_head_export_spec,
    split_block_parameters,
)
from autoware_ml.utils.deploy import ExportSpec

SPARSE4D_EXPORT_OUTPUT_NAMES = (
    "classification",
    "prediction",
    "quality",
    *TEMPORAL_STATE_OUTPUT_NAMES,
    TEMPORAL_QUERY_INDEX_OUTPUT_NAME,
)


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
        if det_fusion.num_levels != len(bbox_head.level_grids):
            raise ValueError(
                f"The pyramid returns {det_fusion.num_levels} levels but the head samples "
                f"{len(bbox_head.level_grids)}."
            )
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
            detection3d_head_outputs=Detection3DHeadOutputs(
                center_head_outputs=None,
                transfusion_head_outputs=None,
                sparse4d_head_outputs=det3d_outputs,
            ),
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
        frame_meta = multi_task_batch_inputs.multi_task_gt_batch.frame_meta_batch
        if frame_meta is None:
            raise ValueError("The Sparse4D head needs the frame metadata of every sample.")
        gt_detections = detection3d_gt_batch(multi_task_batch_inputs)
        point = self.encode_batch(multi_task_batch_inputs)
        # Fuse before the decoder, SerializedUnpooling pops the encoder chain
        det3d_stages = self.det_fusion(point)
        seg3d_logits = self.seg3d_head(point)
        det3d_outputs = self.bbox_head(
            det3d_stages,
            timestamps_seconds=frame_meta.timestamps,
            lidar2global=frame_meta.ego2globals,
            scene_tokens=frame_meta.scene_tokens,
            gt_boxes=gt_detections.valid_bboxes_3d(),
            gt_labels=gt_detections.valid_labels_3d(),
            gt_instance_ids=gt_detections.valid_bboxes_instance_ids(),
            gt_num_points=gt_detections.valid_bboxes_num_points(),
            context=self.head_context(multi_task_batch_inputs),
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
        self.log_step_metrics(metrics, multi_task_batch_inputs)
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

        gt_detections = detection3d_gt_batch(multi_task_batch_inputs)
        # Samples that only pad a streaming lane take no part in the detection loss
        padding_mask = multi_task_batch_inputs.multi_task_gt_batch.padding_mask
        det3d_mask = detection3d_frame_mask(gt_detections) & ~padding_mask.to(
            gt_detections.gt_bboxes_3d.device
        )
        det3d_targets = None
        if bool(det3d_mask.any()):
            supervised = det3d_mask.tolist()
            lanes = det3d_mask.nonzero().flatten().tolist()
            supervised_outputs = det3d_outputs.select_samples(det3d_mask)
            det3d_targets = self.bbox_head.assign(
                supervised_outputs,
                _flagged(gt_detections.valid_bboxes_3d(), supervised),
                _flagged(gt_detections.valid_labels_3d(), supervised),
                _flagged(gt_detections.valid_bboxes_instance_ids(), supervised),
                lanes,
            )
            det3d_metrics = self.bbox_head.loss_from_targets(
                supervised_outputs, det3d_targets, lanes
            )
        else:
            # Zero gradients keep the detection branch visible to DDP
            graph_tensors = [
                det3d_outputs.classification,
                det3d_outputs.prediction,
                det3d_outputs.quality,
            ]
            det3d_metrics = {"loss": sum(tensor.float().sum() for tensor in graph_tensors) * 0.0}

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

    def log_step_metrics(
        self, metrics: dict[str, torch.Tensor], multi_task_batch_inputs: ModelBatchInputs
    ) -> None:
        """Log the per-step values of rank 0 in training, to watch the convergence in an epoch.

        Args:
            metrics: Losses of the step.
            multi_task_batch_inputs: Model inputs of the step.
        """
        if not self.training:
            return
        self.log_dict(
            {
                f"train_step/{name}": value.detach()
                for name, value in metrics.items()
                if "_layer_" not in name or "_layer_-1_" in name
            },
            on_step=True,
            on_epoch=False,
            batch_size=self.get_log_batch_size(multi_task_batch_inputs),
        )

    def build_eval_output(self, batch: ModelBatchInputs, outputs: ModelOutputs) -> dict[str, Any]:
        """Produce the detection and the current frame segmentation eval data.

        Frames without detection supervision contribute empty predictions. A temporal head adds
        which boxes were observed in the stream so far and the stream of every frame.
        """
        det3d_mask = detection3d_frame_mask(detection3d_gt_batch(batch))
        predictions = empty_unsupervised_predictions(
            self.bbox_head.decode_outputs(outputs.detection3d()).detection3d_predictions,
            det3d_mask,
        )
        eval_out = detection3d_eval_output(
            ModelPredictions(detection3d_predictions=predictions), batch
        )
        tracker = self.bbox_head.tracker
        if tracker is not None:
            eval_out["gt_observed"] = list(tracker.frame_observed)
            eval_out["stream_ids"] = list(tracker.stream_ids)
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


def _flagged(values: Sequence[torch.Tensor], flags: Sequence[bool]) -> list[torch.Tensor]:
    """Select the entries of the flagged samples from a per-sample list."""
    return [value for value, flagged in zip(values, flags, strict=True) if flagged]


def build_sparse4d_head_export_spec(
    context: PTv3ExportContext,
    module: PTv3Sparse4DHeadExportModule,
    head_context_args: Sequence[torch.Tensor] = (),
    head_context_names: Sequence[str] = (),
) -> ExportSpec:
    """Build the temporal Sparse4D head export spec from the stages the pyramid fuses.

    Args:
        context: Shared export context.
        module: Export module of a temporal query head.
        head_context_args: Example graph inputs after the bank state that the module turns into
            the head context.
        head_context_names: Graph names of ``head_context_args``.

    Returns:
        Export spec whose graph takes the bank state of the last frame and returns the next one
        with the query every state slot was selected from.
    """
    det_fusion = module.det_fusion
    finest_stage = context.stage_count - det_fusion.num_stages
    deepest_grid_coord = det_fusion.keep_deepest
    stage_args = det3d_head_export_args(context, finest_stage, deepest_grid_coord)
    return ExportSpec(
        module=module.eval(),
        args=(
            *stage_args,
            *module.bbox_head.temporal_state_export_inputs(stage_args[0].device),
            *head_context_args,
        ),
        input_names=[
            *det3d_head_export_input_names(context.stage_count, finest_stage, deepest_grid_coord),
            *TEMPORAL_STATE_INPUT_NAMES,
            *head_context_names,
        ],
        output_names=list(module.output_names),
        dynamic_axes=det3d_head_export_dynamic_axes(
            context.stage_count, finest_stage, deepest_grid_coord
        ),
        supported_stages=PTv3BaseModel.EXPORT_SUPPORTED_STAGES,
    )


def fuse_export_stages(
    det_fusion: PTv3Det3DFeaturePyramid, inputs: Sequence[torch.Tensor]
) -> tuple[list[SparseStage], tuple[torch.Tensor, ...]]:
    """Fuse the encoder stages a head graph takes ahead of its other inputs.

    Args:
        det_fusion: Detection feature pyramid.
        inputs: Stage features from the finest to the deepest stage, the pooling clusters of every
            stage but the deepest, the voxel coordinates of every stage the head reads (the
            deepest one too when the pyramid keeps it), then the other graph inputs.

    Returns:
        The fused stages and the other graph inputs.
    """
    num_stages = det_fusion.num_stages
    grid_end = 3 * num_stages - 2 + int(det_fusion.keep_deepest)
    feats = inputs[:num_stages]
    clusters = inputs[num_stages : 2 * num_stages - 1]
    grid_coords = inputs[2 * num_stages - 1 : grid_end]
    return det_fusion(rebuild_point_chain(feats, clusters, grid_coords)), tuple(inputs[grid_end:])


class PTv3Sparse4DHeadExportModule(nn.Module):
    """Export-only temporal Sparse4D head consuming the encoder stages the pyramid fuses.

    The graph inputs are the stage features from the finest to the deepest stage, the pooling
    clusters of every stage but the deepest, the voxel coordinates of every stage the head reads
    (the deepest one too when the pyramid keeps it), the bank state, then the inputs a subclass
    turns into the head context.
    """

    def __init__(
        self,
        det_fusion: PTv3Det3DFeaturePyramid,
        bbox_head: SparseQueryHead,
        output_names: Sequence[str] = SPARSE4D_EXPORT_OUTPUT_NAMES,
    ) -> None:
        """Initialize the export-only head module.

        Args:
            det_fusion: Detection feature pyramid (copied internally).
            bbox_head: Export-ready temporal query head.
            output_names: Frame outputs the graph returns, in order.
        """
        super().__init__()
        self.det_fusion = deepcopy(det_fusion).eval()
        self.bbox_head = bbox_head
        self.output_names = tuple(output_names)

    def forward(self, *inputs: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Decode one frame and return the configured outputs."""
        outputs = self.frame_outputs(*inputs)
        return tuple(outputs[name] for name in self.output_names)

    def frame_outputs(self, *inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        """Fuse the encoder stages and decode one frame with the bank state of the last one.

        Args:
            *inputs: Stage features, pooling clusters, voxel coordinates, the bank state and the
                head context inputs.

        Returns:
            The outputs of :meth:`Sparse4DHead.forward_temporal_export`.
        """
        stages, rest = fuse_export_stages(self.det_fusion, inputs)
        state_end = len(TEMPORAL_STATE_INPUT_NAMES)
        return self.bbox_head.forward_temporal_export(
            stages, *rest[:state_end], context=self.head_context(rest[state_end:])
        )

    def head_context(self, inputs: Sequence[torch.Tensor]) -> Any:
        """Head context from the graph inputs after the bank state, none for the lidar head.

        Raises:
            ValueError: If the lidar head graph gets inputs after the bank state.
        """
        if inputs:
            raise ValueError("The lidar head graph takes no inputs after the bank state.")
        return None
