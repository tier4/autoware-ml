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

"""SparseDrive model: the PTv3 lidar encoder, the Sparse4D detection head, the PTv3
segmentation head and, attached on demand, the SparseDrive agent motion head.

The Sparse4D head needs scenario streams, see ``StreamingDataModule``, composed with
``/datamodule/streaming@datamodule: default_streaming_datamodule``. Its export carries the
instance bank state through the inputs and the outputs of the detection graph, and the motion
graph carries the history of the motion head the same way.
"""

from __future__ import annotations

import itertools
import operator
from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any, ClassVar, NamedTuple

import torch
import torch.nn as nn
from jaxtyping import Bool, Float, Shaped

from autoware_ml.deployment.graphs import DeploymentGraph, GraphTensor
from autoware_ml.dataclasses.models.detection3d.head_outputs import (
    Detection3DHeadOutputs,
    Sparse4DHeadOutputs,
)
from autoware_ml.dataclasses.models.model_batch_inputs import ModelBatchInputs
from autoware_ml.dataclasses.models.model_outputs import ModelOutputs
from autoware_ml.dataclasses.models.model_predictions import ModelPredictions
from autoware_ml.dataclasses.models.segmentation3d.head_outputs import Segmentation3DHeadOutputs
from autoware_ml.metrics.detection3d.eval_output import detection3d_eval_output
from autoware_ml.metrics.eval_output import EvalOutput
from autoware_ml.metrics.motion.eval_output import MotionEvalOutput
from autoware_ml.models.sparsedrive.lidar_stages import PTv3Det3DFeaturePyramid, SparseStage
from autoware_ml.models.sparsedrive.motion_head import (
    MOTION_DETECTION_OUTPUT_NAMES,
    MotionHead,
    MotionHeadExportModule,
    check_motion_head,
    motion_deployment_inputs,
    motion_eval_frames,
    motion_forward,
    motion_loss,
)
from autoware_ml.models.sparsedrive.sparse4d_head import (
    TEMPORAL_QUERY_INDEX_OUTPUT_NAME,
    TEMPORAL_STATE_INPUT_NAMES,
    TEMPORAL_STATE_OUTPUT_NAMES,
    Sparse4DFrameOutputs,
    Sparse4DHead,
    Sparse4DTargets,
    TemporalStateInputs,
)
from autoware_ml.models.detection3d.ptv3 import (
    det3d_head_export_dynamic_axes,
    det3d_head_export_input_names,
    rebuild_point_chain,
)
from autoware_ml.models.detection3d.task_modules.partial_annotations import PartialAnnotations
from autoware_ml.models.module_base_model import LogDictConfigs
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
from autoware_ml.models.segmentation3d.ptv3_base import (
    DET3D_HEAD_GRAPH,
    PTv3BaseModel,
    PTv3EncoderExportModule,
    PTv3Seg3DHeadExportModule,
    split_block_parameters,
)

SPARSE4D_EXPORT_OUTPUT_NAMES = (
    "classification",
    "prediction",
    "quality",
    *TEMPORAL_STATE_OUTPUT_NAMES,
    TEMPORAL_QUERY_INDEX_OUTPUT_NAME,
)


class ForwardBranches(NamedTuple):
    """Outputs of the shared encoder pass and the two heads.

    Attributes:
        det3d_outputs: Outputs of the Sparse4D head.
        seg3d_logits: Segmentation logits of every voxel.
    """

    det3d_outputs: Sparse4DHeadOutputs
    seg3d_logits: Float[torch.Tensor, "num_voxels num_classes"]


class DetectionSegmentationLosses(NamedTuple):
    """Losses of one step with the detection supervision they came from.

    Attributes:
        metrics: Weighted losses and their terms by logged name.
        det3d_targets: Detection targets of every decoder layer over the frames with boxes.
        det3d_mask: Whether every frame takes part in the detection loss.
    """

    metrics: dict[str, Float[torch.Tensor, ""]]
    det3d_targets: list[Sparse4DTargets]
    det3d_mask: Bool[torch.Tensor, " batch_size"]


class FusedExportInputs(NamedTuple):
    """Fused stages of a head graph and the graph inputs that follow the stage inputs.

    Attributes:
        stages: Fused stages from the finest to the coarsest.
        rest: Graph inputs after the stage inputs.
    """

    stages: list[SparseStage]
    rest: tuple[Shaped[torch.Tensor, "..."], ...]


class SparseDriveModel(PTv3BaseModel):
    """SparseDrive model with the temporal Sparse4D detection head and an optional motion head.

    The motion head reads the last layer queries of the detection head and adds its loss, its
    eval frames and its export graph. Without it the model detects and segments only.
    """

    # The instance bank and the tracker need lane-contiguous batches.
    requires_streaming: ClassVar[bool] = True
    skips_unsupervised_frames: ClassVar[bool] = True

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
        partial_annotations: PartialAnnotations | None = None,
        motion_head: MotionHead | None = None,
        motion_loss_weight: float = 1.0,
        motion_eval_queries: int = 300,
        **kwargs: Any,
    ) -> None:
        """Initialize the SparseDrive model.

        Args:
            encoder: PTv3 encoder module shared by both branches.
            voxel_encoder: Voxel feature encoder feeding the embedding stem.
            seg3d_head: Segmentation decoder head owning losses and the classifier.
            det_fusion: Top-down fusion of the deepest encoder stage into the finer stages,
                whose sparse voxels the detection head samples.
            bbox_head: Temporal Sparse4D query head.
            log_dict_configs: Logging settings of the step metrics.
            segmentation3d_loss_weight: Weight of the segmentation loss term.
            detection3d_loss_weight: Weight of the detection loss term.
            grid_size: Voxel grid size of the encoder input.
            point_cloud_range: Point-cloud range of the encoder input.
            partial_annotations: Maps the dataset's per-sample annotation flag to the classes
                the head must not supervise in that sample. None supervises every class.
            motion_head: Motion head reading the detection queries, None detects and segments
                only.
            motion_loss_weight: Weight of the motion loss term.
            motion_eval_queries: Most confident detection queries whose trajectories the motion
                metrics read per frame.
            **kwargs: Data preprocessor, optimizer, scheduler and metrics of the base model.

        Raises:
            ValueError: If the pyramid and the head disagree on the strides of the sampled
                levels, or the motion head has anchors of other classes than the detection head.
        """
        super().__init__(
            encoder=encoder,
            voxel_encoder=voxel_encoder,
            grid_size=grid_size,
            point_cloud_range=point_cloud_range,
            log_dict_configs=log_dict_configs,
            **kwargs,
        )
        # Stage k pools by the product of the first k encoder strides
        stage_strides = list(itertools.accumulate(encoder.stride, operator.mul, initial=1))
        first = len(stage_strides) - det_fusion.num_stages
        pyramid_strides = stage_strides[first : first + det_fusion.num_levels]
        head_strides = [grid.feature_stride for grid in bbox_head.level_grids]
        if head_strides != pyramid_strides:
            raise ValueError(
                f"The pyramid returns the strides {pyramid_strides} but the head samples "
                f"{head_strides}."
            )
        if bbox_head.bev is not None and bbox_head.bev.stage_strides != pyramid_strides:
            raise ValueError(
                f"The pyramid returns the strides {pyramid_strides} but the BEV pyramid projects "
                f"stages of {bbox_head.bev.stage_strides}."
            )
        self.seg3d_head = seg3d_head
        self.det_fusion = det_fusion
        self.bbox_head = bbox_head
        self.partial_annotations = partial_annotations
        self.segmentation3d_loss_weight = float(segmentation3d_loss_weight)
        self.detection3d_loss_weight = float(detection3d_loss_weight)
        if motion_head is not None:
            check_motion_head(motion_head, bbox_head)
        self.motion_head = motion_head
        self.motion_loss_weight = float(motion_loss_weight)
        self.motion_eval_queries = motion_eval_queries

    def build_optimizer_groups(self) -> Mapping[str, Sequence[torch.nn.Parameter]]:
        """Group the pretrained encoder parameters apart from the newly initialized heads."""
        encoder_default_params, encoder_block_params = split_block_parameters(self.encoder)
        groups = {
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
        if self.motion_head is not None:
            groups["motion_head"] = [p for p in self.motion_head.parameters() if p.requires_grad]
        return groups

    def reset_streams(self) -> None:
        """Forget the temporal state of every lane, so the next frame of a lane starts a stream."""
        self.bbox_head.reset()
        if self.motion_head is not None:
            self.motion_head.reset()

    def on_train_epoch_start(self) -> None:
        """Start new streams for the shuffled scenes of the epoch."""
        super().on_train_epoch_start()
        self.reset_streams()

    def on_validation_epoch_start(self) -> None:
        """Start new streams for the validation scenes, after the metrics are reset."""
        super().on_validation_epoch_start()
        self.reset_streams()

    def on_test_epoch_start(self) -> None:
        """Start new streams for the test scenes, after the metrics are reset."""
        super().on_test_epoch_start()
        self.reset_streams()

    def on_predict_epoch_start(self) -> None:
        """Start new streams for the predict scenes."""
        super().on_predict_epoch_start()
        self.reset_streams()

    def forward(self, multi_task_batch_inputs: ModelBatchInputs) -> ModelOutputs:
        """Run one shared PTv3 encoder pass, branch into both heads and run the motion head.

        Args:
            multi_task_batch_inputs: Model inputs with the voxels, the frame metadata and the
                detection ground truth.

        Returns:
            The Sparse4D head outputs, the segmentation logits and the motion outputs when a
            motion head is attached.
        """
        branches = self.forward_branches(multi_task_batch_inputs)
        motion_outputs = None
        if self.motion_head is not None:
            motion_outputs = motion_forward(
                self.motion_head, self.bbox_head, branches.det3d_outputs, multi_task_batch_inputs
            )
        return ModelOutputs(
            detection3d_head_outputs=Detection3DHeadOutputs(
                center_head_outputs=None,
                transfusion_head_outputs=None,
                sparse4d_head_outputs=branches.det3d_outputs,
            ),
            segmentation3d_head_outputs=Segmentation3DHeadOutputs(logits=branches.seg3d_logits),
            motion_head_outputs=motion_outputs,
        )

    def forward_branches(self, multi_task_batch_inputs: ModelBatchInputs) -> ForwardBranches:
        """Encode the batch once and run the detection and the segmentation heads.

        The frame metadata places every frame of a stream for the temporal head, the boxes feed
        the denoising queries and their instance ids and point counts the tracks.

        Args:
            multi_task_batch_inputs: Model inputs with the voxels, the frame metadata and, to
                train or evaluate, the detection ground truth.

        Returns:
            The detection outputs and the segmentation logits.

        Raises:
            ValueError: If the batch carries no frame metadata.
        """
        frame_meta = multi_task_batch_inputs.multi_task_gt_batch.frame_meta_batch
        if frame_meta is None:
            raise ValueError("The Sparse4D head needs the frame metadata of every sample.")
        point = self.encode_batch(multi_task_batch_inputs)
        # Fuse before the decoder, SerializedUnpooling pops the encoder chain
        det3d_stages = self.det_fusion(point)
        seg3d_logits = self.seg3d_head(point)
        det3d_outputs = self.bbox_head(
            det3d_stages,
            timestamps_seconds=frame_meta.timestamps,
            lidar2global=frame_meta.ego2globals,
            scene_tokens=frame_meta.scene_tokens,
            gt_detections=multi_task_batch_inputs.multi_task_gt_batch.detection3d_gt_batch,
        )
        return ForwardBranches(det3d_outputs=det3d_outputs, seg3d_logits=seg3d_logits)

    def compute_metrics(
        self,
        multi_task_batch_inputs: ModelBatchInputs,
        multi_task_outputs: ModelOutputs,
    ) -> dict[str, Float[torch.Tensor, ""]]:
        """Compute the segmentation, detection and, with a motion head, motion losses.

        Frames without ground-truth boxes and samples padding a streaming lane skip the
        detection loss, and the motion loss with it, since its targets follow the detection
        matches.

        Raises:
            ValueError: If a motion head is attached and the batch carries no futures.
        """
        losses = self.detection_segmentation_metrics(multi_task_batch_inputs, multi_task_outputs)
        metrics = losses.metrics
        if self.motion_head is not None:
            motion_metrics = motion_loss(
                self.motion_head,
                multi_task_batch_inputs,
                multi_task_outputs.motion(),
                losses.det3d_targets,
                losses.det3d_mask,
            )
            weighted_motion_loss = self.motion_loss_weight * motion_metrics["loss"]
            metrics["weighted_motion_loss"] = weighted_motion_loss
            metrics["loss"] = metrics["loss"] + weighted_motion_loss
            metrics.update({f"motion_{name}": value for name, value in motion_metrics.items()})
        self.log_step_metrics(metrics, multi_task_batch_inputs)
        return metrics

    def detection_segmentation_metrics(
        self,
        multi_task_batch_inputs: ModelBatchInputs,
        multi_task_outputs: ModelOutputs,
    ) -> DetectionSegmentationLosses:
        """Compute the segmentation and the detection losses without logging them.

        A batch without a supervised frame runs the same losses over no frame, so every step
        reports the same terms, as the epoch sync of the logged values across ranks needs.

        Args:
            multi_task_batch_inputs: Model inputs with the ground truth.
            multi_task_outputs: Raw outputs from :meth:`forward`.

        Returns:
            The weighted losses, the detection targets of every decoder layer over the frames
            with boxes and the mask of those frames.
        """
        seg3d_logits = multi_task_outputs.segmentation3d().logits
        det3d_outputs = multi_task_outputs.detection3d().sparse4d_head()
        seg3d_metrics = segmentation3d_point_loss(
            self.seg3d_head, seg3d_logits, multi_task_batch_inputs
        )

        gt_detections = detection3d_gt_batch(multi_task_batch_inputs)
        # Samples that pad a streaming lane or lie between key frames take no part in the loss
        supervised = multi_task_batch_inputs.multi_task_gt_batch.supervised_mask()
        det3d_mask = detection3d_frame_mask(gt_detections) & supervised.to(
            gt_detections.gt_bboxes_3d.device
        )
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
        self.bbox_head.follow_assignment(supervised_outputs, det3d_targets, lanes)
        annotated_class_masks = None
        if self.partial_annotations is not None:
            status = gt_detections.gt_traffic_cone_barrier_bbox_status
            annotated_class_masks = self.partial_annotations.build_annotated_class_masks(
                None if status is None else status[det3d_mask]
            )
        det3d_metrics = self.bbox_head.loss_from_targets(
            supervised_outputs, det3d_targets, lanes, annotated_class_masks
        )

        seg3d_loss = seg3d_metrics["loss"]
        weighted_seg3d_loss = self.segmentation3d_loss_weight * seg3d_loss
        weighted_det3d_loss = self.detection3d_loss_weight * det3d_metrics["loss"]
        metrics: dict[str, Float[torch.Tensor, ""]] = {
            "seg3d_loss_ce": seg3d_metrics["loss_ce"],
            "seg3d_loss_lovasz": seg3d_metrics["loss_lovasz"],
            "seg3d_loss": seg3d_loss,
            "weighted_seg3d_loss": weighted_seg3d_loss,
            "weighted_det3d_loss": weighted_det3d_loss,
            "loss": weighted_det3d_loss + weighted_seg3d_loss,
        }
        metrics.update({f"det3d_{name}": value for name, value in det3d_metrics.items()})
        return DetectionSegmentationLosses(
            metrics=metrics, det3d_targets=det3d_targets, det3d_mask=det3d_mask
        )

    def log_step_metrics(
        self, metrics: dict[str, Float[torch.Tensor, ""]], multi_task_batch_inputs: ModelBatchInputs
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

    def build_eval_output(self, batch: ModelBatchInputs, outputs: ModelOutputs) -> EvalOutput:
        """Produce the detection, the current frame segmentation and the motion eval data.

        Frames without detection supervision contribute empty predictions. The frames carry
        which boxes were observed in the stream so far and the stream of every frame.

        Args:
            batch: Model inputs of the evaluation step.
            outputs: Raw head outputs returned by :meth:`forward`.

        Returns:
            The eval output of the detection and the segmentation suites, and of the motion
            suites when a motion head is attached.
        """
        det3d_mask = detection3d_frame_mask(detection3d_gt_batch(batch))
        predictions = empty_unsupervised_predictions(
            self.bbox_head.decode_outputs(outputs.detection3d()).detection3d_predictions,
            det3d_mask,
        )
        detection = detection3d_eval_output(
            ModelPredictions(detection3d_predictions=predictions), batch
        )
        motion = None
        if self.motion_head is not None:
            motion = MotionEvalOutput(
                frames=motion_eval_frames(
                    self.motion_head,
                    self.bbox_head,
                    batch,
                    outputs,
                    detection,
                    self.motion_eval_queries,
                )
            )
        return EvalOutput(
            detection3d=detection,
            segmentation3d=segmentation3d_eval_output(outputs.segmentation3d().logits, batch),
            motion=motion,
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
        return ModelPredictions(
            detection3d_predictions=self.bbox_head.decode_outputs(
                outputs.detection3d()
            ).detection3d_predictions,
            segmentation3d_predictions=segmentation3d_predict_outputs(
                outputs.segmentation3d().logits, multi_task_batch_inputs
            ),
        )

    def deployment_graphs(self) -> Sequence[DeploymentGraph]:
        """Declare ``lidar_encoder``, ``seg3d_head``, the temporal ``det3d_head`` and, with a
        motion head, ``motion_head``.

        The detection graph then also returns the query features the motion graph reads, and the
        motion graph shares the anchor encoder of the detection graph's export head.

        Returns:
            The graphs in that order, the model outputs are the seg head's, then the det head's,
            then the motion head's.

        Raises:
            ValueError: If the grid size or the point cloud range is missing, or a module holds
                no anchors.
        """
        if self.grid_size is None or self.point_cloud_range is None:
            raise ValueError(
                "grid_size and point_cloud_range must be provided at construction time to deploy."
            )
        pending = self.pending_anchored_modules()
        if pending:
            raise ValueError(
                f"{[type(module).__name__ for module in pending]} hold no anchors, export the "
                "weights of a trained model."
            )
        sparse_shape, serialization_depth = self.compute_export_geometry(torch.device("cpu"))
        det3d_output_names = SPARSE4D_EXPORT_OUTPUT_NAMES
        if self.motion_head is not None:
            det3d_output_names = (*det3d_output_names, *MOTION_DETECTION_OUTPUT_NAMES)
        det3d_head = Sparse4DHeadExportModule(
            self.det_fusion,
            self.bbox_head.prepare_for_export(),
            det3d_output_names,
            len(self.encoder.stride) + 1,
        )
        graphs = (
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
            det3d_head.build_deployment_graph(),
        )
        if self.motion_head is None:
            return graphs
        motion_head = MotionHeadExportModule(
            self.motion_head.prepare_for_export(), det3d_head.bbox_head
        )
        return (*graphs, motion_head.build_deployment_graph())

    def deployment_inputs(
        self, multi_task_batch_inputs: ModelBatchInputs
    ) -> dict[str, GraphTensor]:
        """Serialize the points and add the empty state the first frame of a stream reads.

        Args:
            multi_task_batch_inputs: The preprocessed single-sample batch, holding the voxels.

        Returns:
            The inputs of every graph by name.
        """
        tensors = super().deployment_inputs(multi_task_batch_inputs)
        device = tensors["voxels"].device
        state = self.bbox_head.temporal_state_export_inputs(device)
        tensors.update(zip(TEMPORAL_STATE_INPUT_NAMES, state, strict=True))
        if self.motion_head is not None:
            tensors.update(motion_deployment_inputs(self.motion_head, self.bbox_head, device))
        return tensors


def _flagged(
    values: Sequence[Shaped[torch.Tensor, "..."]], flags: Sequence[bool]
) -> list[Shaped[torch.Tensor, "..."]]:
    """Select the entries of the flagged samples from a per-sample list."""
    return [value for value, flagged in zip(values, flags, strict=True) if flagged]


def fuse_export_stages(
    det_fusion: PTv3Det3DFeaturePyramid, inputs: Sequence[Shaped[torch.Tensor, "..."]]
) -> FusedExportInputs:
    """Fuse the encoder stages a head graph takes ahead of its other inputs.

    Args:
        det_fusion: Detection feature pyramid.
        inputs: Stage features from the finest to the deepest stage, the pooling clusters and the
            voxel coordinates of every stage but the deepest, then the other graph inputs.

    Returns:
        The fused stages and the other graph inputs.
    """
    num_stages = det_fusion.num_stages
    grid_end = 3 * num_stages - 2
    feats = inputs[:num_stages]
    clusters = inputs[num_stages : 2 * num_stages - 1]
    grid_coords = inputs[2 * num_stages - 1 : grid_end]
    return FusedExportInputs(
        stages=det_fusion(rebuild_point_chain(feats, clusters, grid_coords)),
        rest=tuple(inputs[grid_end:]),
    )


class Sparse4DHeadExportModule(nn.Module):
    """Export-only temporal Sparse4D head consuming the encoder stages the pyramid fuses.

    The graph inputs are the stage features from the finest to the deepest stage, the pooling
    clusters and the voxel coordinates of every stage but the deepest, then the bank state.
    """

    def __init__(
        self,
        det_fusion: PTv3Det3DFeaturePyramid,
        bbox_head: Sparse4DHead,
        output_names: Sequence[str],
        stage_count: int,
    ) -> None:
        """Initialize the export-only head module.

        Args:
            det_fusion: Detection feature pyramid (copied internally).
            bbox_head: Export-ready temporal head.
            output_names: Frame outputs the graph returns, in order.
            stage_count: Number of encoder stages, the input level included.

        Raises:
            ValueError: If an output name is no frame output of the head.
        """
        super().__init__()
        unknown = sorted(set(output_names) - set(Sparse4DFrameOutputs._fields))
        if unknown:
            raise ValueError(f"The head graph has no outputs {unknown}.")
        self.det_fusion = deepcopy(det_fusion).eval()
        self.bbox_head = bbox_head
        self.output_names = tuple(output_names)
        self.stage_count = stage_count
        self.finest_stage = stage_count - det_fusion.num_stages

    def build_deployment_graph(self) -> DeploymentGraph:
        """Declare this module as the ``det3d_head`` graph.

        Returns:
            The graph from the fused stages and the bank state of the last frame to the frame
            outputs and the next bank state.
        """
        return DeploymentGraph(
            DET3D_HEAD_GRAPH,
            module=self,
            input_names=(
                *det3d_head_export_input_names(self.stage_count, self.finest_stage),
                *TEMPORAL_STATE_INPUT_NAMES,
            ),
            output_names=self.output_names,
            onnx_dynamic_axes=det3d_head_export_dynamic_axes(self.stage_count, self.finest_stage),
        )

    def forward(
        self, *inputs: Shaped[torch.Tensor, "..."]
    ) -> tuple[Shaped[torch.Tensor, "..."], ...]:
        """Decode one frame and return the configured outputs."""
        outputs = self.frame_outputs(*inputs)._asdict()
        return tuple(outputs[name] for name in self.output_names)

    def frame_outputs(self, *inputs: Shaped[torch.Tensor, "..."]) -> Sparse4DFrameOutputs:
        """Fuse the encoder stages and decode one frame with the bank state of the last one.

        Args:
            *inputs: Stage features, pooling clusters, voxel coordinates and the bank state.

        Returns:
            The outputs of :meth:`Sparse4DHead.forward_temporal_export`.
        """
        fused = fuse_export_stages(self.det_fusion, inputs)
        return self.bbox_head.forward_temporal_export(
            fused.stages, TemporalStateInputs(*fused.rest)
        )
