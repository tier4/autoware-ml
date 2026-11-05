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

"""PTv3 Sparse4D detection model and the pieces the joint model shares with it.

A temporal head needs scenario streams, see ``DataModule(streaming=True)``. Its export
carries the instance bank state through the inputs and the outputs of the detection graph.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any

import torch
import torch.nn as nn

from autoware_ml.dataclasses.batch.frame_meta import FrameMetaBatch
from autoware_ml.dataclasses.models.detection3d.head_outputs import (
    Detection3DHeadOutputs,
    Sparse4DHeadOutputs,
)
from autoware_ml.dataclasses.models.model_batch_inputs import ModelBatchInputs
from autoware_ml.dataclasses.models.model_outputs import ModelOutputs
from autoware_ml.dataclasses.models.model_predictions import ModelPredictions
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
from autoware_ml.models.segmentation3d.ptv3_base import (
    PTv3BaseModel,
    PTv3ExportContext,
    build_encoder_export_spec,
    build_ptv3_export_context,
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


class PTv3Detection3DSparse4DModel(PTv3BaseModel):
    """PTv3 Sparse4D detection model."""

    def __init__(
        self,
        encoder: PointTransformerV3Encoder,
        voxel_encoder: VoxelFeatureEncoder,
        det_fusion: PTv3Det3DFeaturePyramid,
        bbox_head: Sparse4DHead,
        log_dict_configs: LogDictConfigs,
        grid_size: float | None = None,
        point_cloud_range: Sequence[float] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the PTv3 Sparse4D detection model.

        Args:
            encoder: PTv3 encoder module.
            voxel_encoder: Voxel feature encoder feeding the embedding stem.
            det_fusion: Top-down fusion of the deepest encoder stage into the finer stages,
                whose sparse voxels the detection head samples.
            bbox_head: Sparse4D query head.
            log_dict_configs: Logging settings of the step metrics.
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
        self.det_fusion = det_fusion
        self.bbox_head = bbox_head

    def build_optimizer_groups(self) -> Mapping[str, Sequence[torch.nn.Parameter]]:
        """Group the pretrained encoder parameters apart from the newly initialized head."""
        encoder_default_params, encoder_block_params = split_block_parameters(self.encoder)
        return {
            "encoder_default": encoder_default_params,
            "encoder_block": encoder_block_params,
            "det3d_branch": [
                parameter
                for module in (self.det_fusion, self.bbox_head)
                for parameter in module.parameters()
                if parameter.requires_grad
            ],
        }

    def forward(self, batch_inputs: ModelBatchInputs) -> ModelOutputs:
        """Run one PTv3 encoder pass and the Sparse4D head.

        Args:
            batch_inputs: Model inputs with the voxels, the frame metadata and the detection
                ground truth.

        Returns:
            The Sparse4D head outputs.
        """
        _, det3d_outputs = self.forward_stages(batch_inputs)
        return ModelOutputs(detection3d_head_outputs=sparse4d_head_outputs(det3d_outputs))

    def forward_stages(
        self, batch_inputs: ModelBatchInputs
    ) -> tuple[list[SparseStage], Sparse4DHeadOutputs]:
        """Encode the batch and run the detection head on the fused stages.

        Args:
            batch_inputs: Model inputs with the voxels, the frame metadata and the detection
                ground truth.

        Returns:
            The fused stages the detection head sampled and its outputs.

        Raises:
            ValueError: If the batch carries no frame metadata.
        """
        frame_meta = sparse4d_frame_meta(batch_inputs)
        point = self.encode_batch(batch_inputs)
        det3d_stages = self.det_fusion(point)
        det3d_outputs = run_sparse4d_head(
            self.bbox_head, det3d_stages, batch_inputs, frame_meta, self.head_context(batch_inputs)
        )
        return det3d_stages, det3d_outputs

    def head_context(self, batch_inputs: ModelBatchInputs) -> Any:
        """Frame context of the detection head, none for the lidar model."""
        return None

    def compute_metrics(
        self, batch_inputs: ModelBatchInputs, outputs: ModelOutputs
    ) -> dict[str, torch.Tensor]:
        """Compute the detection losses of one step.

        Frames without ground-truth boxes and samples padding a streaming lane skip the loss.

        Args:
            batch_inputs: Model inputs with the ground truth.
            outputs: Raw outputs from :meth:`forward`.

        Returns:
            The losses of the head, ``loss`` the total.
        """
        metrics, _, _ = sparse4d_detection_loss(
            self.bbox_head, outputs.detection3d().sparse4d_head(), batch_inputs
        )
        log_sparse4d_step_metrics(self, metrics, batch_inputs)
        return metrics

    def build_eval_output(self, batch: ModelBatchInputs, outputs: ModelOutputs) -> dict[str, Any]:
        """Produce the detection eval data, with the tracks of a temporal head."""
        return sparse4d_eval_output(self.bbox_head, batch, outputs)

    def decode_outputs(
        self, batch_inputs: ModelBatchInputs, outputs: ModelOutputs
    ) -> ModelPredictions:
        """Decode the boxes of the batch for inference.

        Args:
            batch_inputs: Model inputs of the prediction step, unused.
            outputs: Raw outputs returned by :meth:`forward`.

        Returns:
            The decoded boxes of every sample.
        """
        del batch_inputs
        return self.bbox_head.decode_outputs(outputs.detection3d())

    def build_export_specs(self, batch_inputs: ModelBatchInputs) -> dict[str, ExportSpec]:
        """Build the split ONNX export specs of the encoder and the det head.

        Args:
            batch_inputs: Example preprocessed batch with the voxels.

        Returns:
            Export specs of the encoder and the temporal detection head.

        Raises:
            ValueError: If the grid size or the point cloud range is missing.
        """
        if self.grid_size is None or self.point_cloud_range is None:
            raise ValueError(
                "grid_size and point_cloud_range must be provided at construction time to use "
                "export."
            )
        context = build_ptv3_export_context(self, batch_inputs)
        return {
            "ptv3_lidar_encoder": build_encoder_export_spec(context),
            "ptv3_det3d_head": self.build_det3d_head_export_spec(context, batch_inputs),
        }

    def build_det3d_head_export_spec(
        self, context: PTv3ExportContext, batch_inputs: ModelBatchInputs
    ) -> ExportSpec:
        """Build the temporal detection head export spec.

        Args:
            context: Shared export context.
            batch_inputs: Example preprocessed batch, read for a head context.

        Returns:
            Export spec of the detection graph over the fused stages and the bank state.
        """
        return build_sparse4d_head_export_spec(
            context,
            PTv3Sparse4DHeadExportModule(self.det_fusion, self.bbox_head.prepare_for_export()),
        )


def validate_sampled_levels(det_fusion: PTv3Det3DFeaturePyramid, bbox_head: Sparse4DHead) -> None:
    """Check that the pyramid returns the levels the head samples.

    Raises:
        ValueError: If the pyramid and the head disagree on the sampled levels.
    """
    if det_fusion.num_levels != len(bbox_head.level_grids):
        raise ValueError(
            f"The pyramid returns {det_fusion.num_levels} levels but the head samples "
            f"{len(bbox_head.level_grids)}."
        )


def sparse4d_frame_meta(batch_inputs: ModelBatchInputs) -> FrameMetaBatch:
    """Read the frame metadata a temporal head places every frame of a stream with.

    Args:
        batch_inputs: Model inputs of the batch.

    Returns:
        The frame metadata of every sample.

    Raises:
        ValueError: If the batch carries no frame metadata.
    """
    frame_meta = batch_inputs.multi_task_gt_batch.frame_meta_batch
    if frame_meta is None:
        raise ValueError("The Sparse4D head needs the frame metadata of every sample.")
    return frame_meta


def run_sparse4d_head(
    bbox_head: Sparse4DHead,
    det3d_stages: Sequence[SparseStage],
    batch_inputs: ModelBatchInputs,
    frame_meta: FrameMetaBatch,
    context: Any = None,
) -> Sparse4DHeadOutputs:
    """Run the Sparse4D head over the fused stages of a batch.

    The frame metadata places every frame of a stream for the temporal head, the boxes feed the
    denoising queries and their instance ids and point counts the tracks.

    Args:
        bbox_head: Sparse4D query head.
        det3d_stages: Fused encoder stages the head samples.
        batch_inputs: Model inputs with the detection ground truth.
        frame_meta: Frame metadata of every sample, see :func:`sparse4d_frame_meta`.
        context: Frame context of the head.

    Returns:
        The head outputs.
    """
    gt_detections = detection3d_gt_batch(batch_inputs)
    return bbox_head(
        det3d_stages,
        timestamps_seconds=frame_meta.timestamps,
        lidar2global=frame_meta.ego2globals,
        scene_tokens=frame_meta.scene_tokens,
        gt_boxes=gt_detections.valid_bboxes_3d(),
        gt_labels=gt_detections.valid_labels_3d(),
        gt_instance_ids=gt_detections.valid_bboxes_instance_ids(),
        gt_num_points=gt_detections.valid_bboxes_num_points(),
        context=context,
    )


def sparse4d_head_outputs(det3d_outputs: Sparse4DHeadOutputs) -> Detection3DHeadOutputs:
    """Wrap the Sparse4D head outputs as the detection outputs of the model."""
    return Detection3DHeadOutputs(
        center_head_outputs=None,
        transfusion_head_outputs=None,
        sparse4d_head_outputs=det3d_outputs,
    )


def sparse4d_detection_loss(
    bbox_head: Sparse4DHead, det3d_outputs: Sparse4DHeadOutputs, batch_inputs: ModelBatchInputs
) -> tuple[dict[str, torch.Tensor], list[Sparse4DTargets] | None, torch.Tensor]:
    """Compute the detection losses over the frames with boxes.

    Frames without ground-truth boxes and samples that only pad a streaming lane take no part in
    the loss. Without any such frame the loss is zero with gradients, which keeps the detection
    branch visible to DDP.

    Args:
        bbox_head: Sparse4D query head.
        det3d_outputs: Head outputs of the batch.
        batch_inputs: Model inputs with the ground truth.

    Returns:
        The losses of the head, the targets of every decoder layer over the supervised frames
        (None when no frame has boxes) and the mask of those frames.
    """
    gt_detections = detection3d_gt_batch(batch_inputs)
    padding_mask = batch_inputs.multi_task_gt_batch.padding_mask
    det3d_mask = detection3d_frame_mask(gt_detections) & ~padding_mask.to(
        gt_detections.gt_bboxes_3d.device
    )
    if not bool(det3d_mask.any()):
        graph_tensors = [
            det3d_outputs.classification,
            det3d_outputs.prediction,
            det3d_outputs.quality,
        ]
        zero = sum(tensor.float().sum() for tensor in graph_tensors) * 0.0
        return {"loss": zero}, None, det3d_mask

    supervised = det3d_mask.tolist()
    lanes = det3d_mask.nonzero().flatten().tolist()
    supervised_outputs = det3d_outputs.select_samples(det3d_mask)
    det3d_targets = bbox_head.assign(
        supervised_outputs,
        _flagged(gt_detections.valid_bboxes_3d(), supervised),
        _flagged(gt_detections.valid_labels_3d(), supervised),
        _flagged(gt_detections.valid_bboxes_instance_ids(), supervised),
        lanes,
    )
    det3d_metrics = bbox_head.loss_from_targets(supervised_outputs, det3d_targets, lanes)
    return det3d_metrics, det3d_targets, det3d_mask


def log_sparse4d_step_metrics(
    model: PTv3BaseModel, metrics: Mapping[str, torch.Tensor], batch_inputs: ModelBatchInputs
) -> None:
    """Log the per-step values of rank 0 in training, to watch the convergence in an epoch.

    The per layer losses are logged for the last decoder layer only.

    Args:
        model: Model logging the step.
        metrics: Losses of the step.
        batch_inputs: Model inputs of the step.
    """
    if not model.training:
        return
    model.log_dict(
        {
            f"train_step/{name}": value.detach()
            for name, value in metrics.items()
            if "_layer_" not in name or "_layer_-1_" in name
        },
        on_step=True,
        on_epoch=False,
        batch_size=model.get_log_batch_size(batch_inputs),
    )


def sparse4d_eval_output(
    bbox_head: Sparse4DHead, batch: ModelBatchInputs, outputs: ModelOutputs
) -> dict[str, Any]:
    """Produce the detection eval data of a batch.

    Frames without detection supervision contribute empty predictions. A temporal head adds
    which boxes were observed in the stream so far and the stream of every frame.

    Args:
        bbox_head: Sparse4D query head.
        batch: Model inputs of the evaluation step.
        outputs: Raw outputs returned by the model forward.

    Returns:
        Flat eval output dict consumed by the detection and the tracking metrics.
    """
    det3d_mask = detection3d_frame_mask(detection3d_gt_batch(batch))
    predictions = empty_unsupervised_predictions(
        bbox_head.decode_outputs(outputs.detection3d()).detection3d_predictions, det3d_mask
    )
    eval_out = detection3d_eval_output(ModelPredictions(detection3d_predictions=predictions), batch)
    tracker = bbox_head.tracker
    if tracker is not None:
        eval_out["gt_observed"] = list(tracker.frame_observed)
        eval_out["stream_ids"] = list(tracker.stream_ids)
    return eval_out


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
