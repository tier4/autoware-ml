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

"""PTv3 segmentation and Sparse4D detection with the SparseDrive agent motion head.

The motion head predicts the trajectory of every detection query from the queries of the temporal
detection head and their history, so the model trains on scenario streams.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
import torch.nn as nn
from jaxtyping import Bool, Float, Int64, Shaped

from autoware_ml.dataclasses.models.detection3d.head_outputs import (
    Detection3DHeadOutputs,
    Sparse4DHeadOutputs,
)
from autoware_ml.dataclasses.models.model_batch_inputs import ModelBatchInputs
from autoware_ml.dataclasses.models.model_outputs import ModelOutputs
from autoware_ml.dataclasses.models.motion.head_outputs import MotionHeadOutputs
from autoware_ml.dataclasses.models.motion.targets import AgentFutureTargets
from autoware_ml.dataclasses.models.segmentation3d.head_outputs import Segmentation3DHeadOutputs
from autoware_ml.metrics.eval_output import Detection3DEvalOutput, EvalOutput
from autoware_ml.metrics.motion.eval_output import MotionEvalFrame, MotionEvalOutput
from autoware_ml.models.detection3d.heads.sparse4d import Sparse4DHead, Sparse4DTargets
from autoware_ml.models.motion.instance_queue import QueueState
from autoware_ml.models.motion.motion_head import (
    MOTION_STATE_INPUT_NAMES,
    MOTION_STATE_OUTPUT_NAMES,
    MotionHead,
)
from autoware_ml.models.multi.detection3d_frames import detection3d_frame_mask, detection3d_gt_batch
from autoware_ml.models.multi.ptv3_det3d_seg3d_sparse4d import PTv3Det3DSeg3DSparse4DModel
from autoware_ml.models.segmentation3d.ptv3_base import PTv3BaseModel, PTv3ExportContext
from autoware_ml.types.geometry import AnchorFieldIndex
from autoware_ml.utils.deploy import ExportSpec

# Anchor channels of the velocity a detection estimates
_VELOCITY = [AnchorFieldIndex.VELOCITY_X, AnchorFieldIndex.VELOCITY_Y]

# The motion graph takes the detection outputs of the frame and the history of the last one
MOTION_EXPORT_INPUT_NAMES = (
    "instance_feature",
    "prediction",
    "classification",
    *MOTION_STATE_INPUT_NAMES,
    # Slot index the detection graph returned for the last frame
    "carried_index",
    "temp_valid",
    "prev2current",
)
MOTION_EXPORT_OUTPUT_NAMES = (
    "agent_mode_logits",
    "agent_trajectories",
    *MOTION_STATE_OUTPUT_NAMES,
)


class PTv3SparseDriveMotionModel(PTv3Det3DSeg3DSparse4DModel):
    """PTv3 joint segmentation and Sparse4D detection with the SparseDrive agent motion head."""

    def __init__(
        self,
        *,
        motion_head: MotionHead,
        motion_loss_weight: float = 1.0,
        motion_eval_queries: int = 300,
        **kwargs: Any,
    ) -> None:
        """Initialize the model.

        Args:
            motion_head: Motion head reading the detection queries.
            motion_loss_weight: Weight of the motion loss term.
            motion_eval_queries: Most confident detection queries whose trajectories the motion
                metrics read per frame.
            **kwargs: Arguments of :class:`PTv3Det3DSeg3DSparse4DModel`.

        Raises:
            ValueError: If the motion head has anchors of other classes.
        """
        super().__init__(**kwargs)
        if motion_head.num_classes != self.bbox_head.num_classes:
            raise ValueError(
                f"The motion head has anchors for {motion_head.num_classes} classes, the detection "
                f"head predicts {self.bbox_head.num_classes}."
            )
        self.motion_head = motion_head
        self.motion_loss_weight = float(motion_loss_weight)
        self.motion_eval_queries = motion_eval_queries

    def reset_streams(self) -> None:
        """Forget the temporal state of every lane, the motion history too."""
        super().reset_streams()
        self.motion_head.reset()

    def build_optimizer_groups(self) -> Mapping[str, Sequence[torch.nn.Parameter]]:
        """Group the parameters of the base model and the motion head."""
        groups = dict(super().build_optimizer_groups())
        groups["motion_head"] = [p for p in self.motion_head.parameters() if p.requires_grad]
        return groups

    def forward(self, multi_task_batch_inputs: ModelBatchInputs) -> ModelOutputs:
        """Run the encoder, the detection and segmentation heads and the motion head.

        Args:
            multi_task_batch_inputs: Model inputs with the voxels, the frame metadata and the
                detection ground truth.

        Returns:
            The detection, segmentation and motion outputs.
        """
        branches = self.forward_branches(multi_task_batch_inputs)
        carried_index = self.bbox_head.instance_bank.carried_query_index
        motion_outputs = self.forward_motion(
            multi_task_batch_inputs, branches.det3d_outputs, carried_index
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

    def forward_motion(
        self,
        multi_task_batch_inputs: ModelBatchInputs,
        det3d_outputs: Sparse4DHeadOutputs,
        carried_index: Int64[torch.Tensor, "batch_size num_temp_instances"] | None,
    ) -> MotionHeadOutputs:
        """Run the motion head on the last layer detection queries.

        Args:
            multi_task_batch_inputs: Model inputs with the frame metadata.
            det3d_outputs: Outputs of the detection head.
            carried_index: Slot index of the previous frame, None at the first frame.

        Returns:
            The motion outputs.
        """
        frame_meta = multi_task_batch_inputs.multi_task_gt_batch.frame_meta_batch
        return self.motion_head(
            det3d_outputs.instance_feature,
            det3d_outputs.prediction[:, -1],
            det3d_outputs.classification[:, -1],
            self.bbox_head.anchor_encoder,
            self.bbox_head.instance_bank.lane_continues,
            carried_index,
            frame_meta.ego2globals,
        )

    def compute_metrics(
        self,
        multi_task_batch_inputs: ModelBatchInputs,
        multi_task_outputs: ModelOutputs,
    ) -> dict[str, Float[torch.Tensor, ""]]:
        """Compute the segmentation, detection and motion losses.

        The motion targets follow the detection matches, so frames without boxes skip the motion
        loss as they skip the detection loss.

        Raises:
            ValueError: If the batch carries no futures.
        """
        losses = self.detection_segmentation_metrics(multi_task_batch_inputs, multi_task_outputs)
        metrics = losses.metrics
        motion_metrics = self.motion_loss(
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

    def agent_future_targets(self, batch: ModelBatchInputs) -> AgentFutureTargets:
        """Align the agent futures of the batch with its box slots, at the future steps of the head.

        Args:
            batch: Model inputs with the ground truth.

        Returns:
            The future of the agent of every box slot.

        Raises:
            ValueError: If the batch carries no agent tracks, or tracks of other future steps
                than the head.
        """
        motion_gt = batch.multi_task_gt_batch.motion_gt_batch
        if motion_gt is None:
            raise ValueError("The motion head needs the agent tracks of every sample.")
        targets = AgentFutureTargets.from_agent_tracks(
            motion_gt.agents, detection3d_gt_batch(batch).gt_bboxes_instance_ids
        )
        steps = targets.agent_future_valid.shape[-1]
        if steps != self.motion_head.agent_steps:
            raise ValueError(
                f"The batch carries futures of {steps} steps, the motion head predicts "
                f"{self.motion_head.agent_steps}."
            )
        return targets

    def motion_loss(
        self,
        multi_task_batch_inputs: ModelBatchInputs,
        motion_outputs: MotionHeadOutputs,
        det3d_targets: list[Sparse4DTargets],
        det3d_mask: Bool[torch.Tensor, " batch_size"],
    ) -> dict[str, Float[torch.Tensor, ""]]:
        """Supervise the motion head on the frames with boxes.

        Args:
            multi_task_batch_inputs: Model inputs with the ground truth.
            motion_outputs: Raw outputs of the motion head.
            det3d_targets: Detection targets of every decoder layer over the frames with boxes.
            det3d_mask: Mask of the frames with boxes.

        Returns:
            The motion loss terms and their sum under ``loss``.

        Raises:
            ValueError: If the batch carries no futures of the steps of the head.
        """
        targets = self.agent_future_targets(multi_task_batch_inputs)
        gt_detections = detection3d_gt_batch(multi_task_batch_inputs)
        return self.motion_head.loss(
            motion_outputs.select_samples(det3d_mask),
            det3d_targets[-1].assigned_instance,
            gt_detections.gt_bboxes_3d[det3d_mask],
            gt_detections.gt_bboxes_instance_ids[det3d_mask],
            targets.select_samples(det3d_mask),
        )

    def build_eval_output(self, batch: ModelBatchInputs, outputs: ModelOutputs) -> EvalOutput:
        """Produce the detection, segmentation and motion eval data.

        Args:
            batch: Model inputs of the evaluation step.
            outputs: Raw outputs from :meth:`forward`.

        Returns:
            The eval output of the detection, segmentation and motion suites.
        """
        eval_output = super().build_eval_output(batch, outputs)
        return eval_output.model_copy(
            update={
                "motion": MotionEvalOutput(
                    frames=self.motion_eval_frames(batch, outputs, eval_output.detection3d)
                )
            }
        )

    def motion_eval_frames(
        self,
        batch: ModelBatchInputs,
        outputs: ModelOutputs,
        detection: Detection3DEvalOutput | None,
    ) -> list[MotionEvalFrame]:
        """Pair the trajectories of the most confident queries with the futures of every frame.

        Args:
            batch: Model inputs with the ground truth boxes and futures.
            outputs: Raw outputs from :meth:`forward`.
            detection: Detection frames of the batch, which carry the observation flags of the
                boxes and whether the metrics score the frame.

        Returns:
            One motion frame per sample of the batch.

        Raises:
            ValueError: If the batch carries no futures of the steps of the head, or the
                detection frames carry no observation flags.
        """
        if detection is None or not detection.check_all_gt_observed:
            raise ValueError(
                "The motion metrics need the observation flags of the boxes, the temporal head "
                "attaches them to the detection frames."
            )
        targets = self.agent_future_targets(batch)
        det3d_outputs = outputs.detection3d().sparse4d_head()
        motion_outputs = outputs.motion()
        class_scores = self.bbox_head.query_scores(det3d_outputs).detach()
        scores, labels = class_scores.max(dim=-1)
        anchor = det3d_outputs.prediction[:, -1].detach()
        top = scores.topk(min(self.motion_eval_queries, scores.shape[1]), dim=1).indices
        gt_detections = detection3d_gt_batch(batch)
        # A frame without boxes is empty for the motion metrics, as for the detection metrics
        with_boxes = detection3d_frame_mask(gt_detections).tolist()
        frames = []
        for index, count in enumerate(gt_detections.gt_valid_bboxes.tolist()):
            queries = top[index] if with_boxes[index] else top[index, :0]
            boxes = gt_detections.gt_bboxes_3d[index, :count]
            detection_frame = detection.frames[index]
            frames.append(
                MotionEvalFrame(
                    pred_xy=anchor[index, queries, :2],
                    pred_labels=labels[index, queries],
                    pred_scores=scores[index, queries],
                    pred_trajectories=anchor[index, queries, None, None, :2]
                    + motion_outputs.agent_trajectories[index, queries].detach(),
                    pred_mode_scores=motion_outputs.agent_mode_logits[index, queries]
                    .detach()
                    .float()
                    .softmax(dim=-1),
                    pred_velocities=anchor[index, queries][:, _VELOCITY],
                    gt_xy=boxes[:, :2],
                    gt_labels=gt_detections.gt_labels_3d[index, :count],
                    gt_futures=targets.agent_future_positions[index, :count],
                    gt_valid=targets.agent_future_valid[index, :count],
                    gt_observed=detection_frame.gt_observed,
                    scored=detection_frame.scored,
                )
            )
        return frames

    def det3d_export_output_names(self) -> tuple[str, ...]:
        """Frame outputs of the detection graph, the instance features for the motion graph too."""
        return (*super().det3d_export_output_names(), "instance_feature")

    def build_context_export_specs(self, context: PTv3ExportContext) -> dict[str, ExportSpec]:
        """Build the split export specs, the detection graph handing the motion graph its queries.

        Args:
            context: Shared export context.

        Returns:
            The specs of the base model, the detection graph with the instance features as an
            extra output and the motion graph.
        """
        specs = super().build_context_export_specs(context)
        det_module = specs["ptv3_det3d_head"].module
        specs["ptv3_motion_head"] = self.build_motion_export_spec(context, det_module.bbox_head)
        return specs

    def build_motion_export_spec(
        self, context: PTv3ExportContext, bbox_head: Sparse4DHead
    ) -> ExportSpec:
        """Build the motion export spec, one frame with the history at the graph boundary.

        Args:
            context: Shared export context.
            bbox_head: Export-prepared detection head of the detection graph.

        Returns:
            Export spec whose graph takes the detection outputs of the frame, the history of the
            last frame, the slot index the detection graph returned for the last frame, the
            stream continuity and the frame to frame transform.
        """
        motion_head = self.motion_head.prepare_for_export()
        device = context.stage_feats[0].device
        bank = bbox_head.instance_bank
        num_queries = bank.num_anchor
        args = (
            torch.zeros(1, num_queries, bbox_head.embed_dims, device=device),
            bank.anchor.detach()[None].to(device),
            torch.zeros(1, num_queries, bbox_head.num_classes, device=device),
            *motion_head.export_state_inputs(num_queries, device),
            torch.zeros(1, bank.num_temp_instances, dtype=torch.long, device=device),
            torch.zeros(1, device=device),
            torch.eye(4, device=device)[None],
        )
        return ExportSpec(
            module=_MotionExportModule(motion_head, bbox_head).eval(),
            args=args,
            input_names=list(MOTION_EXPORT_INPUT_NAMES),
            output_names=list(MOTION_EXPORT_OUTPUT_NAMES),
            dynamic_axes=None,
            supported_stages=PTv3BaseModel.EXPORT_SUPPORTED_STAGES,
        )


class _MotionExportModule(nn.Module):
    """Export-only motion head, one frame with its history at the graph boundary."""

    def __init__(self, motion_head: MotionHead, bbox_head: Sparse4DHead) -> None:
        """Initialize the export module.

        Args:
            motion_head: Export-prepared motion head.
            bbox_head: Export-prepared detection head lending its anchor encoder.
        """
        super().__init__()
        self.motion_head = motion_head
        self.bbox_head = bbox_head

    def forward(
        self,
        instance_feature: Float[torch.Tensor, "batch_size num_queries embed_dims"],
        prediction: Float[torch.Tensor, "batch_size num_queries anchor_dim"],
        classification: Float[torch.Tensor, "batch_size num_queries num_classes"],
        queue_feature: Float[torch.Tensor, "batch_size history_length num_queries embed_dims"],
        queue_anchor: Float[torch.Tensor, "batch_size history_length num_queries anchor_dim"],
        queue_map: Int64[torch.Tensor, "batch_size history_length num_queries"],
        queue_count: Int64[torch.Tensor, " batch_size"],
        queue_age: Int64[torch.Tensor, " batch_size"],
        carried_index: Int64[torch.Tensor, "batch_size num_temp_instances"],
        temp_valid: Float[torch.Tensor, " batch_size"],
        prev2current: Float[torch.Tensor, "batch_size 4 4"],
    ) -> tuple[Shaped[torch.Tensor, "..."], ...]:
        """Predict the trajectories of one frame, see the input names."""
        outputs, state = self.motion_head.step(
            instance_feature,
            prediction,
            classification,
            self.bbox_head.anchor_encoder,
            QueueState(queue_feature, queue_anchor, queue_map, queue_count, queue_age),
            temp_valid > 0.5,
            carried_index,
            prev2current,
        )
        return (outputs.agent_mode_logits, outputs.agent_trajectories, *state)
