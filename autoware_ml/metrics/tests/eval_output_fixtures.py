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

"""Builders of the typed eval output the metric tests feed to the suites."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from autoware_ml.dataclasses.batch.frame_meta import FrameMetaSample
from autoware_ml.dataclasses.models.detection3d.predictions import Detection3DSamplePredictions
from autoware_ml.metrics.eval_output import (
    Detection3DEvalFrame,
    Detection3DEvalOutput,
    EvalOutput,
    Segmentation3DEvalFrame,
    Segmentation3DEvalOutput,
)


def frame_meta(
    lanelet_map_path: str | None, scene_token: str = "scene", ego2global: torch.Tensor | None = None
) -> FrameMetaSample:
    """Metadata of one frame, the ego at the map origin unless a pose is given.

    Args:
        lanelet_map_path: Path of the scene's lanelet map, ``None`` for a scene without one.
        scene_token: Scene identifier of the frame.
        ego2global: 4x4 pose of the frame in the map, the identity when ``None``.

    Returns:
        The frame metadata.
    """
    return FrameMetaSample(
        ego2global=torch.eye(4, dtype=torch.float32) if ego2global is None else ego2global,
        scene_token=scene_token,
        timestamp=0.0,
        lanelet_map_path=lanelet_map_path,
    )


def prediction(
    bboxes_3d: torch.Tensor, scores_3d: torch.Tensor, labels_3d: torch.Tensor
) -> Detection3DSamplePredictions:
    """Decoded predictions of one frame, cast to the dtypes the typed output takes.

    Args:
        bboxes_3d: ``(N, 7+)`` boxes.
        scores_3d: ``(N,)`` scores.
        labels_3d: ``(N,)`` class indices.

    Returns:
        The predictions of the frame.
    """
    return Detection3DSamplePredictions(
        bboxes_3d=bboxes_3d.to(torch.float32),
        scores_3d=scores_3d.to(torch.float32),
        labels_3d=labels_3d.to(torch.long),
    )


def empty_prediction(num_box_params: int = 9) -> Detection3DSamplePredictions:
    """Predictions of a frame without a single box.

    Args:
        num_box_params: Parameters per box, 7 or 9.

    Returns:
        The empty predictions.
    """
    return prediction(
        torch.zeros((0, num_box_params)), torch.zeros((0,)), torch.zeros((0,), dtype=torch.long)
    )


def detection_frame(
    predictions: Detection3DSamplePredictions,
    gt_boxes: torch.Tensor,
    gt_labels: torch.Tensor,
    gt_num_points: torch.Tensor | None = None,
    is_padding: bool = False,
    meta: FrameMetaSample | None = None,
) -> Detection3DEvalFrame:
    """One detection frame, the ground truth cast to the dtypes the typed output takes.

    Args:
        predictions: Decoded predictions of the frame.
        gt_boxes: ``(M, 7+)`` ground truth boxes.
        gt_labels: ``(M,)`` ground truth class indices.
        gt_num_points: ``(M,)`` lidar points per box, or ``None``.
        is_padding: Whether the frame only pads a streaming lane.
        meta: Metadata of the frame, or ``None``.

    Returns:
        The frame.
    """
    return Detection3DEvalFrame(
        predictions=predictions,
        gt_boxes=gt_boxes.to(torch.float32),
        gt_labels=gt_labels.to(torch.long),
        gt_num_points=gt_num_points,
        is_padding=is_padding,
        frame_meta=meta,
    )


def detection_eval_output(frames: Sequence[Detection3DEvalFrame]) -> EvalOutput:
    """An eval output holding detection frames only.

    Args:
        frames: The frames of the batch.

    Returns:
        The eval output.
    """
    return EvalOutput(detection3d=Detection3DEvalOutput(frames=frames))


def segmentation_frame(
    pred: torch.Tensor,
    target: torch.Tensor,
    coord: torch.Tensor,
    scores: torch.Tensor | None = None,
    meta: FrameMetaSample | None = None,
    gt_boxes: torch.Tensor | None = None,
    gt_box_labels: torch.Tensor | None = None,
) -> Segmentation3DEvalFrame:
    """One segmentation frame.

    Args:
        pred: ``(N,)`` predicted class per point.
        target: ``(N,)`` ground truth class per point.
        coord: ``(N, 3+)`` point coordinates.
        scores: ``(N, C)`` per-class probabilities, or ``None`` when no metric reads them.
        meta: Metadata of the frame, or ``None``.
        gt_boxes: ``(M, 7+)`` detection ground truth boxes, or ``None``.
        gt_box_labels: ``(M,)`` detection ground truth class indices, or ``None``.

    Returns:
        The frame.
    """
    return Segmentation3DEvalFrame(
        coord=coord,
        pred=pred,
        target=target,
        scores=scores,
        frame_meta=meta,
        gt_boxes=None if gt_boxes is None else gt_boxes.to(torch.float32),
        gt_box_labels=None if gt_box_labels is None else gt_box_labels.to(torch.long),
    )


def segmentation_eval_output(frames: Sequence[Segmentation3DEvalFrame]) -> EvalOutput:
    """An eval output holding segmentation frames only.

    Args:
        frames: The frames of the batch.

    Returns:
        The eval output.
    """
    return EvalOutput(segmentation3d=Segmentation3DEvalOutput(frames=frames))
