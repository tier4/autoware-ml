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

"""Builders of the typed motion eval output for the motion metric tests."""

from __future__ import annotations

import torch

from autoware_ml.dataclasses.batch.frame_meta import FrameMetaSample
from autoware_ml.dataclasses.models.detection3d.predictions import Detection3DSamplePredictions
from autoware_ml.metrics.eval_output import EvalOutput
from autoware_ml.metrics.motion.eval_output import MotionEvalFrame, MotionEvalOutput


def _boxes(xy: torch.Tensor, velocities: torch.Tensor | None = None) -> torch.Tensor:
    """Unit boxes at BEV centers, at rest unless the velocities are given."""
    boxes = torch.zeros((xy.shape[0], 9), dtype=torch.float32, device=xy.device)
    boxes[:, :2] = xy
    boxes[:, 3:6] = 1.0
    if velocities is not None:
        boxes[:, 7:9] = velocities
    return boxes


def motion_frame(
    fields: dict[str, torch.Tensor],
    observed: torch.Tensor,
    scored: bool = True,
    step_seconds: float = 0.5,
) -> MotionEvalFrame:
    """A typed motion frame from flat test fields.

    Args:
        fields: ``pred_xy``, ``pred_labels``, ``pred_scores``, ``pred_velocities``,
            ``pred_trajectories``, ``pred_mode_scores``, ``gt_xy``, ``gt_labels``, ``gt_futures``
            and ``gt_valid``.
        observed: Whether the stream observes every ground truth agent, its box then holds
            points.
        scored: Whether the metrics score the frame, a frame between key frames otherwise.
        step_seconds: Time between two future steps.

    Returns:
        The motion frame of one scene of its own.
    """
    device = fields["gt_xy"].device
    num_gts = fields["gt_xy"].shape[0]
    return MotionEvalFrame(
        predictions=Detection3DSamplePredictions(
            bboxes_3d=_boxes(fields["pred_xy"], fields["pred_velocities"]),
            scores_3d=fields["pred_scores"].float(),
            labels_3d=fields["pred_labels"].long(),
        ),
        pred_trajectories=fields["pred_trajectories"],
        pred_mode_scores=fields["pred_mode_scores"],
        gt_boxes=_boxes(fields["gt_xy"]),
        gt_labels=fields["gt_labels"],
        gt_num_points=torch.where(observed.to(device), 5, 0).long(),
        gt_instance_ids=torch.arange(num_gts, dtype=torch.long, device=device),
        gt_futures=fields["gt_futures"],
        gt_valid=fields["gt_valid"],
        step_seconds=step_seconds,
        is_padding=False,
        key_frame=scored,
        frame_meta=FrameMetaSample(
            ego2global=torch.eye(4, dtype=torch.float64),
            scene_token="scene",
            timestamp=0.0,
            key_frame=scored,
        ),
    )


def motion_output(
    frames: list[dict[str, torch.Tensor]],
    observed: list[torch.Tensor],
    scored: list[bool] | None = None,
) -> EvalOutput:
    """Typed motion frames of a batch, every frame scored unless told otherwise.

    Args:
        frames: The fields of every frame, see :func:`motion_frame`.
        observed: Whether the stream observes every ground truth agent, per frame.
        scored: Whether the metrics score each frame, every frame when ``None``.

    Returns:
        The eval output holding the motion frames.
    """
    return EvalOutput(
        motion=MotionEvalOutput(
            frames=[
                motion_frame(fields, frame_observed, frame_scored)
                for fields, frame_observed, frame_scored in zip(
                    frames, observed, scored or [True] * len(frames), strict=True
                )
            ]
        )
    )
