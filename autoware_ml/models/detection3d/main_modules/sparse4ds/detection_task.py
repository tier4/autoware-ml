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

"""The detection task of a Sparse4D head: running it over a stream, its loss and its outputs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch

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
    Sparse4DHead,
    Sparse4DTargets,
    SparseStage,
)
from autoware_ml.models.detection3d.ptv3 import PTv3Det3DFeaturePyramid
from autoware_ml.models.module_base_model import ModuleBaseModel
from autoware_ml.models.multi.ptv3_det3d_seg3d import (
    detection3d_frame_mask,
    detection3d_gt_batch,
    empty_unsupervised_predictions,
)


class Sparse4DDetectionTask:
    """Run a Sparse4D head over the frames of a stream and score it.

    The task holds the head of a model and gathers what every model with a Sparse4D head does
    alike: feeding the frame metadata and the boxes to the head, the loss over the supervised
    frames, the eval output with the tracks and the decoding. The model keeps the head as its
    own submodule, the task only refers to it.
    """

    def __init__(self, bbox_head: Sparse4DHead) -> None:
        """Initialize the task.

        Args:
            bbox_head: Sparse4D query head of the model.
        """
        self.bbox_head = bbox_head

    def validate_levels(self, det_fusion: PTv3Det3DFeaturePyramid) -> None:
        """Check that a pyramid returns the levels the head samples.

        Args:
            det_fusion: Feature pyramid feeding the head.

        Raises:
            ValueError: If the pyramid and the head disagree on the sampled levels.
        """
        if det_fusion.num_levels != len(self.bbox_head.level_grids):
            raise ValueError(
                f"The pyramid returns {det_fusion.num_levels} levels but the head samples "
                f"{len(self.bbox_head.level_grids)}."
            )

    @staticmethod
    def frame_meta(batch_inputs: ModelBatchInputs) -> FrameMetaBatch:
        """Read the frame metadata the head places every frame of a stream with.

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

    def run(
        self,
        stages: Sequence[SparseStage],
        batch_inputs: ModelBatchInputs,
        frame_meta: FrameMetaBatch,
        context: Any = None,
    ) -> Sparse4DHeadOutputs:
        """Run the head over the fused stages of a batch.

        The frame metadata places every frame of a stream for the temporal head, the boxes feed
        the denoising queries and their instance ids and point counts the tracks.

        Args:
            stages: Fused encoder stages the head samples.
            batch_inputs: Model inputs with the detection ground truth.
            frame_meta: Frame metadata of every sample, see :meth:`frame_meta`.
            context: Frame context of the head.

        Returns:
            The head outputs.
        """
        gt_detections = detection3d_gt_batch(batch_inputs)
        return self.bbox_head(
            stages,
            timestamps_seconds=frame_meta.timestamps,
            lidar2global=frame_meta.ego2globals,
            scene_tokens=frame_meta.scene_tokens,
            gt_boxes=gt_detections.valid_bboxes_3d(),
            gt_labels=gt_detections.valid_labels_3d(),
            gt_instance_ids=gt_detections.valid_bboxes_instance_ids(),
            gt_num_points=gt_detections.valid_bboxes_num_points(),
            context=context,
        )

    @staticmethod
    def head_outputs(det3d_outputs: Sparse4DHeadOutputs) -> Detection3DHeadOutputs:
        """Wrap the head outputs as the detection outputs of a model."""
        return Detection3DHeadOutputs(
            center_head_outputs=None,
            transfusion_head_outputs=None,
            sparse4d_head_outputs=det3d_outputs,
        )

    def loss(
        self, det3d_outputs: Sparse4DHeadOutputs, batch_inputs: ModelBatchInputs
    ) -> tuple[dict[str, torch.Tensor], list[Sparse4DTargets] | None, torch.Tensor]:
        """Compute the detection losses over the frames with boxes.

        Frames without ground-truth boxes and samples that only pad a streaming lane take no
        part in the loss. Without any such frame the loss is zero with gradients, which keeps
        the detection branch visible to DDP.

        Args:
            det3d_outputs: Head outputs of the batch.
            batch_inputs: Model inputs with the ground truth.

        Returns:
            The losses of the head, the targets of every decoder layer over the supervised
            frames (None when no frame has boxes) and the mask of those frames.
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
        det3d_targets = self.bbox_head.assign(
            supervised_outputs,
            _flagged(gt_detections.valid_bboxes_3d(), supervised),
            _flagged(gt_detections.valid_labels_3d(), supervised),
            _flagged(gt_detections.valid_bboxes_instance_ids(), supervised),
            lanes,
        )
        det3d_metrics = self.bbox_head.loss_from_targets(supervised_outputs, det3d_targets, lanes)
        return det3d_metrics, det3d_targets, det3d_mask

    def eval_output(self, batch_inputs: ModelBatchInputs, outputs: ModelOutputs) -> dict[str, Any]:
        """Produce the detection eval data of a batch.

        Frames without detection supervision contribute empty predictions. A temporal head adds
        which boxes were observed in the stream so far and the stream of every frame.

        Args:
            batch_inputs: Model inputs of the evaluation step.
            outputs: Raw outputs returned by the model forward.

        Returns:
            Flat eval output dict consumed by the detection and the tracking metrics.
        """
        det3d_mask = detection3d_frame_mask(detection3d_gt_batch(batch_inputs))
        predictions = empty_unsupervised_predictions(
            self.decode(outputs).detection3d_predictions, det3d_mask
        )
        eval_out = detection3d_eval_output(
            ModelPredictions(detection3d_predictions=predictions), batch_inputs
        )
        tracker = self.bbox_head.tracker
        if tracker is not None:
            eval_out["gt_observed"] = list(tracker.frame_observed)
            eval_out["stream_ids"] = list(tracker.stream_ids)
        return eval_out

    def decode(self, outputs: ModelOutputs) -> ModelPredictions:
        """Decode the boxes of every sample for inference."""
        return self.bbox_head.decode_outputs(outputs.detection3d())

    @staticmethod
    def log_step_metrics(
        model: ModuleBaseModel,
        metrics: Mapping[str, torch.Tensor],
        batch_inputs: ModelBatchInputs,
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


def _flagged(values: Sequence[torch.Tensor], flags: Sequence[bool]) -> list[torch.Tensor]:
    """Select the entries of the flagged samples from a per-sample list."""
    return [value for value, flagged in zip(values, flags, strict=True) if flagged]
