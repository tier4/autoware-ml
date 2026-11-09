"""The typed eval output a model hands to its metric suites.

A model's ``build_eval_output`` returns one :class:`EvalOutput` per batch, holding the
detection and the segmentation part a suite of each task reads. Every part is a tuple of
per-frame models, so a suite loops over frames and reads typed fields instead of
indexing a flat dict by key. The per-frame metadata the region and collision filters
need (ego pose, scene token, lanelet map path) travels as the dataset's own
:class:`~autoware_ml.dataclasses.batch.frame_meta.FrameMetaSample`.

:class:`MetricRequirements` is what a suite declares it needs from the output at its
stage. The eval mixin checks it against the first batch, so a model whose output lacks a
part a configured metric reads fails loud with the model and the part named.
"""

from __future__ import annotations

import torch
from jaxtyping import Float, Float32, Int, Int64
from pydantic import BaseModel, ConfigDict, InstanceOf, model_validator

from autoware_ml.dataclasses.batch.frame_meta import FrameMetaSample
from autoware_ml.dataclasses.models.detection3d.predictions import Detection3DSamplePredictions

MIN_BOX_PARAMS = 7


class Detection3DEvalFrame(BaseModel):
    """
    The predictions and the ground truth of one detection frame.

    Attributes:
      predictions: Decoded boxes of the frame with their scores and labels.
      gt_boxes: Ground truth boxes, 7 or more parameters each, in the frame of the points.
      gt_labels: Class index of every ground truth box.
      gt_num_points: Lidar points inside every ground truth box, when the dataset counts them.
      is_padding: Whether the frame only pads a streaming lane. It is kept so every rank holds
        as many frames as the others for the gather, and the metrics leave it out.
      frame_meta: Pose, scene and lanelet map of the frame, when the dataset attaches them.
    """

    model_config = ConfigDict(frozen=True, strict=True, arbitrary_types_allowed=True)

    predictions: Detection3DSamplePredictions
    gt_boxes: Float32[torch.Tensor, "num_gt_boxes num_box_params"]
    gt_labels: Int64[torch.Tensor, " num_gt_boxes"]
    gt_num_points: Int[torch.Tensor, " num_gt_boxes"] | None = None
    is_padding: bool = False
    frame_meta: InstanceOf[FrameMetaSample] | None = None

    @model_validator(mode="after")
    def _check_ground_truth(self) -> Detection3DEvalFrame:
        """Check that the ground truth columns describe the same boxes."""
        num_gt_boxes = self.gt_boxes.shape[0]
        if self.gt_boxes.shape[1] < MIN_BOX_PARAMS:
            raise ValueError(
                f"gt_boxes need at least {MIN_BOX_PARAMS} parameters per box, got "
                f"{self.gt_boxes.shape[1]}."
            )
        if self.gt_labels.shape[0] != num_gt_boxes:
            raise ValueError(
                f"gt_labels describe {self.gt_labels.shape[0]} boxes, gt_boxes {num_gt_boxes}."
            )
        if self.gt_num_points is not None and self.gt_num_points.shape[0] != num_gt_boxes:
            raise ValueError(
                f"gt_num_points describe {self.gt_num_points.shape[0]} boxes, gt_boxes "
                f"{num_gt_boxes}."
            )
        return self


class Detection3DEvalOutput(BaseModel):
    """
    The detection frames of one batch.

    Attributes:
      frames: One entry per sample of the batch, padding samples included.
    """

    model_config = ConfigDict(frozen=True, strict=True)

    frames: tuple[Detection3DEvalFrame, ...]

    @property
    def carries_frame_meta(self) -> bool:
        """Whether the frames carry their pose, scene and lanelet map."""
        return all(frame.frame_meta is not None for frame in self.frames)

    @property
    def carries_gt_num_points(self) -> bool:
        """Whether the frames carry the lidar point count of every ground truth box."""
        return all(frame.gt_num_points is not None for frame in self.frames)


class Segmentation3DEvalFrame(BaseModel):
    """
    The scored points of one segmentation frame.

    Attributes:
      coord: Point coordinates in the frame of the points, ``x``, ``y``, ``z`` first.
      pred: Predicted class of every point.
      target: Ground truth class of every point, the ignore index included.
      scores: Softmax probability of every class per point, when the metrics read them.
      frame_meta: Pose, scene and lanelet map of the frame, when the dataset attaches them.
      gt_boxes: Detection ground truth boxes of the frame, for the cross-task metrics.
      gt_box_labels: Class index of every detection ground truth box.
    """

    model_config = ConfigDict(frozen=True, strict=True, arbitrary_types_allowed=True)

    coord: Float[torch.Tensor, "num_points num_coord"]
    pred: Int[torch.Tensor, " num_points"]
    target: Int[torch.Tensor, " num_points"]
    scores: Float[torch.Tensor, "num_points num_classes"] | None = None
    frame_meta: InstanceOf[FrameMetaSample] | None = None
    gt_boxes: Float32[torch.Tensor, "num_gt_boxes num_box_params"] | None = None
    gt_box_labels: Int64[torch.Tensor, " num_gt_boxes"] | None = None

    @model_validator(mode="after")
    def _check_points_and_boxes(self) -> Segmentation3DEvalFrame:
        """Check that every per-point array describes the same points, and the boxes agree."""
        num_points = self.coord.shape[0]
        if self.coord.shape[1] < 3:
            raise ValueError(
                f"coord needs x, y and z per point, got {self.coord.shape[1]} columns."
            )
        for name, tensor in (("pred", self.pred), ("target", self.target), ("scores", self.scores)):
            if tensor is not None and tensor.shape[0] != num_points:
                raise ValueError(
                    f"{name} carries {tensor.shape[0]} rows for {num_points} points, every "
                    "per-point array must describe the same points."
                )
        if (self.gt_boxes is None) != (self.gt_box_labels is None):
            raise ValueError("gt_boxes and gt_box_labels come together, one of them is missing.")
        if self.gt_boxes is not None and self.gt_box_labels is not None:
            if self.gt_boxes.shape[1] < MIN_BOX_PARAMS:
                raise ValueError(
                    f"gt_boxes need at least {MIN_BOX_PARAMS} parameters per box, got "
                    f"{self.gt_boxes.shape[1]}."
                )
            if self.gt_box_labels.shape[0] != self.gt_boxes.shape[0]:
                raise ValueError(
                    f"gt_box_labels describe {self.gt_box_labels.shape[0]} boxes, gt_boxes "
                    f"{self.gt_boxes.shape[0]}."
                )
        return self


class Segmentation3DEvalOutput(BaseModel):
    """
    The segmentation frames of one batch.

    Attributes:
      frames: One entry per frame of the batch.
    """

    model_config = ConfigDict(frozen=True, strict=True)

    frames: tuple[Segmentation3DEvalFrame, ...]

    @property
    def carries_frame_meta(self) -> bool:
        """Whether the frames carry their pose, scene and lanelet map."""
        return all(frame.frame_meta is not None for frame in self.frames)

    @property
    def carries_scores(self) -> bool:
        """Whether the frames carry the per-class scores of their points."""
        return all(frame.scores is not None for frame in self.frames)

    @property
    def carries_gt_boxes(self) -> bool:
        """Whether the frames carry the detection ground truth boxes."""
        return all(frame.gt_boxes is not None for frame in self.frames)


class EvalOutput(BaseModel):
    """
    What a model hands its metric suites for one batch.

    Attributes:
      detection3d: The detection frames, when the model detects.
      segmentation3d: The segmentation frames, when the model segments.
    """

    model_config = ConfigDict(frozen=True, strict=True)

    detection3d: Detection3DEvalOutput | None = None
    segmentation3d: Segmentation3DEvalOutput | None = None


class MetricRequirements(BaseModel):
    """
    What a metric suite needs from the eval output at its stage.

    Attributes:
      detection3d: The suite reads the detection frames.
      segmentation3d: The suite reads the segmentation frames.
      frame_meta: A filter or a collision provider reads the pose and lanelet map of the frames.
      gt_num_points: A ground truth point filter reads the lidar point count of every box.
      scores: A point metric reads the per-class scores of the points.
      gt_boxes: A cross-task point metric reads the detection ground truth of the frames.
    """

    model_config = ConfigDict(frozen=True, strict=True)

    detection3d: bool = False
    segmentation3d: bool = False
    frame_meta: bool = False
    gt_num_points: bool = False
    scores: bool = False
    gt_boxes: bool = False

    def unmet(self, eval_output: EvalOutput) -> tuple[str, ...]:
        """
        Name what the eval output fails to provide.

        Args:
          eval_output: The output of the model for one batch.

        Returns:
          tuple[str, ...]: The missing parts, empty when the output satisfies the requirements.
        """
        missing: list[str] = []
        detection = eval_output.detection3d
        segmentation = eval_output.segmentation3d
        if self.detection3d and detection is None:
            missing.append("detection3d frames")
        if self.segmentation3d and segmentation is None:
            missing.append("segmentation3d frames")
        if self.frame_meta and not _carries(detection, segmentation, "carries_frame_meta"):
            missing.append("frame_meta (ego pose, scene token and lanelet map path per frame)")
        if self.gt_num_points and not (detection is not None and detection.carries_gt_num_points):
            missing.append("gt_num_points (lidar points per ground truth box)")
        if self.scores and not (segmentation is not None and segmentation.carries_scores):
            missing.append("scores (per-class probabilities per point)")
        if self.gt_boxes and not (segmentation is not None and segmentation.carries_gt_boxes):
            missing.append("gt_boxes (detection ground truth inside the segmentation frames)")
        return tuple(missing)


def _carries(
    detection: Detection3DEvalOutput | None,
    segmentation: Segmentation3DEvalOutput | None,
    attribute: str,
) -> bool:
    """Whether every present part carries the attribute, and at least one part is present."""
    parts = [part for part in (detection, segmentation) if part is not None]
    return bool(parts) and all(getattr(part, attribute) for part in parts)
