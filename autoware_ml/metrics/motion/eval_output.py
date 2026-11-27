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

"""The typed motion part of the eval output a model hands to its metric suites."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from jaxtyping import Bool, Float, Int, Int64
from pydantic import BaseModel, ConfigDict, InstanceOf, model_validator

from autoware_ml.dataclasses.batch.frame_meta import FrameMetaSample
from autoware_ml.dataclasses.models.detection3d.predictions import Detection3DSamplePredictions


class MotionEvalFrame(BaseModel):
    """
    The predicted and the ground truth trajectories of one frame, positions in the frame.

    Attributes:
      predictions: Boxes, scores and labels of the predicted agents, their velocity as their
        detection estimates it.
      pred_trajectories: Positions of every mode of every predicted agent per future step.
      pred_mode_scores: Probability of every mode of every predicted agent, summing to one.
      gt_boxes: Ground truth boxes of the agents, 7 or more parameters each.
      gt_labels: Class of every ground truth agent.
      gt_num_points: Lidar points inside every ground truth box.
      gt_instance_ids: Object identity of every ground truth agent.
      gt_futures: Future positions of every ground truth agent.
      gt_valid: Whether a future position is known.
      step_seconds: Time between two future steps.
      is_padding: Whether the frame only pads a streaming lane, kept so every rank gathers as many
        frames and left out of the metrics.
      key_frame: Whether the frame is a key frame of its scene, a frame between key frames carries
        no annotations and is left out of the metrics.
      frame_meta: Pose, scene and timestamp of the frame.
    """

    model_config = ConfigDict(frozen=True, strict=True, arbitrary_types_allowed=True)

    predictions: Detection3DSamplePredictions
    pred_trajectories: Float[torch.Tensor, "num_preds num_modes num_steps 2"]
    pred_mode_scores: Float[torch.Tensor, "num_preds num_modes"]
    gt_boxes: Float[torch.Tensor, "num_gts num_box_params"]
    gt_labels: Int[torch.Tensor, " num_gts"]
    gt_num_points: Int[torch.Tensor, " num_gts"]
    gt_instance_ids: Int64[torch.Tensor, " num_gts"]
    gt_futures: Float[torch.Tensor, "num_gts num_steps 2"]
    gt_valid: Bool[torch.Tensor, "num_gts num_steps"]
    step_seconds: float
    is_padding: bool
    key_frame: bool
    frame_meta: InstanceOf[FrameMetaSample]

    @property
    def scored(self) -> bool:
        """Whether the metrics score the frame, a key frame that pads no lane."""
        return self.key_frame and not self.is_padding

    @model_validator(mode="after")
    def _check_rows(self) -> MotionEvalFrame:
        """Check that every prediction and every ground truth column describes the same agents."""
        num_preds = self.predictions.bboxes_3d.shape[0]
        num_gts = self.gt_boxes.shape[0]
        columns = (
            ("pred_trajectories", self.pred_trajectories, num_preds),
            ("pred_mode_scores", self.pred_mode_scores, num_preds),
            ("gt_labels", self.gt_labels, num_gts),
            ("gt_num_points", self.gt_num_points, num_gts),
            ("gt_instance_ids", self.gt_instance_ids, num_gts),
            ("gt_futures", self.gt_futures, num_gts),
            ("gt_valid", self.gt_valid, num_gts),
        )
        for name, value, rows in columns:
            if value.shape[0] != rows:
                raise ValueError(f"{name} describes {value.shape[0]} agents, expected {rows}.")
        if self.pred_mode_scores.shape[1] != self.pred_trajectories.shape[1]:
            raise ValueError(
                f"pred_mode_scores scores {self.pred_mode_scores.shape[1]} modes, "
                f"pred_trajectories holds {self.pred_trajectories.shape[1]}."
            )
        if self.step_seconds <= 0:
            raise ValueError(f"step_seconds must be positive, got {self.step_seconds}.")
        return self


class MotionEvalOutput(BaseModel):
    """
    The motion frames of one batch.

    Attributes:
      frames: One entry per sample of the batch, padding samples included.
    """

    model_config = ConfigDict(frozen=True, strict=True)

    frames: Sequence[MotionEvalFrame]
