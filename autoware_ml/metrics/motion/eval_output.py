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
from jaxtyping import Bool, Float, Int
from pydantic import BaseModel, ConfigDict, model_validator


class MotionEvalFrame(BaseModel):
    """
    The predicted and the ground truth trajectories of one frame, positions in the frame.

    Attributes:
      pred_xy: BEV center of every predicted agent.
      pred_labels: Class of every predicted agent.
      pred_scores: Detection score of every predicted agent.
      pred_trajectories: Positions of every mode of every predicted agent per future step.
      pred_mode_scores: Probability of every mode of every predicted agent, summing to one.
      pred_velocities: BEV velocity in m/s of every predicted agent, as its detection estimates it.
      gt_xy: BEV center of every ground truth agent.
      gt_labels: Class of every ground truth agent.
      gt_futures: Future positions of every ground truth agent.
      gt_valid: Whether a future position is known.
      gt_observed: Whether the stream observed every ground truth agent so far.
      scored: Whether the metrics score the frame. A frame padding a streaming lane or lying
        between key frames is kept so every rank gathers as many frames, and left out.
    """

    model_config = ConfigDict(frozen=True, strict=True, arbitrary_types_allowed=True)

    pred_xy: Float[torch.Tensor, "num_preds 2"]
    pred_labels: Int[torch.Tensor, " num_preds"]
    pred_scores: Float[torch.Tensor, " num_preds"]
    pred_trajectories: Float[torch.Tensor, "num_preds num_modes num_steps 2"]
    pred_mode_scores: Float[torch.Tensor, "num_preds num_modes"]
    pred_velocities: Float[torch.Tensor, "num_preds 2"]
    gt_xy: Float[torch.Tensor, "num_gts 2"]
    gt_labels: Int[torch.Tensor, " num_gts"]
    gt_futures: Float[torch.Tensor, "num_gts num_steps 2"]
    gt_valid: Bool[torch.Tensor, "num_gts num_steps"]
    gt_observed: Bool[torch.Tensor, " num_gts"]
    scored: bool

    @model_validator(mode="after")
    def _check_rows(self) -> MotionEvalFrame:
        """Check that every prediction and every ground truth column describes the same agents."""
        for prefix, rows in (("pred_", self.pred_xy.shape[0]), ("gt_", self.gt_xy.shape[0])):
            for name in type(self).model_fields:
                value = getattr(self, name)
                if name.startswith(prefix) and value.shape[0] != rows:
                    raise ValueError(
                        f"{name} describes {value.shape[0]} agents, {prefix}xy {rows}."
                    )
        if self.pred_mode_scores.shape[1] != self.pred_trajectories.shape[1]:
            raise ValueError(
                f"pred_mode_scores scores {self.pred_mode_scores.shape[1]} modes, "
                f"pred_trajectories holds {self.pred_trajectories.shape[1]}."
            )
        return self


class MotionEvalOutput(BaseModel):
    """
    The motion frames of one batch.

    Attributes:
      frames: One entry per sample of the batch, padding samples included.
    """

    model_config = ConfigDict(frozen=True, strict=True)

    frames: Sequence[MotionEvalFrame]
