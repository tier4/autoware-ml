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

"""Pure data structures for the detection metrics.

``Detection3DSample`` is the per-frame input. ``MatchCurve`` carries one
class's score-ordered matching results and ``CurveMetrics`` its AP-style
summary. ``TpErrorSet`` holds the nuScenes true-positive errors of one selection
of matches and ``RecallTarget`` names a recall operating point to select them at.
The memoizing ``Detection3DState`` that orchestrates the matching lives in
:mod:`autoware_ml.metrics.detection3d.matching`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from jaxtyping import Bool, Float32, Float64, Int, Int64
from pydantic import BaseModel, ConfigDict, model_validator

from autoware_ml.metrics.eval_output import MIN_BOX_PARAMS

# nuScenes center-distance AP thresholds (meters). The AP-family metrics own this
# set. Metrics that need a single operating point share DEFAULT_TP_THRESHOLD (the
# nuScenes dist_th_tp), so the whole report moves together when it changes. Nds
# and TpErrors additionally require tp_threshold to be a configured threshold, so
# the aggregate operating point always appears among the detail keys.
DEFAULT_MATCH_THRESHOLDS = (0.5, 1.0, 2.0, 4.0)
DEFAULT_TP_THRESHOLD = 2.0


class Detection3DSample(BaseModel):
    """
    The predictions and the ground truth of one detection frame, as the metrics read them.

    Attributes:
      pred_boxes: Predicted boxes, 7 or more parameters each, in the frame of the points.
      pred_scores: Confidence of every predicted box.
      pred_labels: Class index of every predicted box.
      gt_boxes: Ground truth boxes kept by the ground truth filters, laid out as the predictions.
      gt_labels: Class index of every ground truth box.
      gt_ttc: Reachability time-to-collision to ego of every ground truth box in seconds,
        ``inf`` when unreachable. Present only when the suite is built with a collision
        provider, and read by the criticality metrics.
      pred_ttc: The same time-to-collision of every predicted box. Set together with ``gt_ttc``
        or not at all.
      ttc_covered: False for a frame whose scene has no lanelet map, so its time-to-collision
        is all ``inf`` by construction. The criticality metrics leave such frames out of their
        denominators, mirroring the region filter coverage rule.
    """

    model_config = ConfigDict(frozen=True, strict=True, arbitrary_types_allowed=True)

    pred_boxes: Float32[torch.Tensor, "num_predictions num_box_params"]
    pred_scores: Float32[torch.Tensor, " num_predictions"]
    pred_labels: Int64[torch.Tensor, " num_predictions"]
    gt_boxes: Float32[torch.Tensor, "num_gt_boxes num_box_params"]
    gt_labels: Int64[torch.Tensor, " num_gt_boxes"]
    gt_ttc: Float32[torch.Tensor, " num_gt_boxes"] | None = None
    pred_ttc: Float32[torch.Tensor, " num_predictions"] | None = None
    ttc_covered: bool = True

    @model_validator(mode="after")
    def _check_rows(self) -> Detection3DSample:
        """Check that every per-box tensor describes the boxes of its side."""
        for side, boxes, columns in (
            (
                "pred",
                self.pred_boxes,
                (
                    ("pred_scores", self.pred_scores),
                    ("pred_labels", self.pred_labels),
                    ("pred_ttc", self.pred_ttc),
                ),
            ),
            ("gt", self.gt_boxes, (("gt_labels", self.gt_labels), ("gt_ttc", self.gt_ttc))),
        ):
            if boxes.shape[1] < MIN_BOX_PARAMS:
                raise ValueError(
                    f"{side}_boxes need at least {MIN_BOX_PARAMS} parameters per box, got "
                    f"{boxes.shape[1]}."
                )
            for name, column in columns:
                if column is not None and column.shape[0] != boxes.shape[0]:
                    raise ValueError(
                        f"{name} describes {column.shape[0]} boxes, {side}_boxes {boxes.shape[0]}."
                    )
        if (self.gt_ttc is None) != (self.pred_ttc is None):
            given, missing = (
                ("pred_ttc", "gt_ttc") if self.gt_ttc is None else ("gt_ttc", "pred_ttc")
            )
            raise ValueError(
                f"{given} is set but {missing} is None, the time-to-collision comes for both "
                "sides of a frame or for neither."
            )
        return self


class MatchCurve(BaseModel):
    """
    The score-ordered matching results of one class at one threshold.

    Every array holds one entry per prediction of the class, in descending score order. The
    error channels are NaN for a prediction without a match.

    Attributes:
      total_gt: Number of ground truth boxes of the class.
      scores: Confidence of every prediction.
      true_positive: Whether the prediction matched a ground truth box.
      false_positive: The complement of ``true_positive``.
      heading_score: Heading-weighted credit of a match in ``[0, 1]``, 0 without a match.
      translation_error: BEV center distance of a match in meters.
      orientation_error: Absolute yaw gap of a match in radians.
      scale_error: One minus the aligned 3D IoU of a match.
      velocity_error: Velocity gap of a match in meters per second.
      attribute_error: Attribute error of a match, the neutral 1.0 while attributes are not
        evaluated.
      corner_error: Mean BEV corner distance of a match in meters, filled alongside the
        nuScenes channels so the driving-aware components read it without matching again.
      nearest_surface_error: Signed near-face error of a match in meters, filled likewise.
    """

    model_config = ConfigDict(frozen=True, strict=True, arbitrary_types_allowed=True)

    total_gt: int
    scores: Float64[np.ndarray, " num_predictions"]
    true_positive: Bool[np.ndarray, " num_predictions"]
    false_positive: Bool[np.ndarray, " num_predictions"]
    heading_score: Float64[np.ndarray, " num_predictions"]
    translation_error: Float64[np.ndarray, " num_predictions"]
    orientation_error: Float64[np.ndarray, " num_predictions"]
    scale_error: Float64[np.ndarray, " num_predictions"]
    velocity_error: Float64[np.ndarray, " num_predictions"]
    attribute_error: Float64[np.ndarray, " num_predictions"]
    corner_error: Float64[np.ndarray, " num_predictions"]
    nearest_surface_error: Float64[np.ndarray, " num_predictions"]

    @model_validator(mode="after")
    def _check_rows(self) -> MatchCurve:
        """Check that every array describes the same predictions."""
        num_predictions = self.scores.shape[0]
        for name in type(self).model_fields:
            value = getattr(self, name)
            if isinstance(value, np.ndarray) and value.shape[0] != num_predictions:
                raise ValueError(
                    f"{name} describes {value.shape[0]} predictions, scores {num_predictions}."
                )
        return self

    @property
    def num_predictions(self) -> int:
        """Return the number of scored predictions in this match curve."""
        return int(self.scores.shape[0])

    @property
    def num_match(self) -> int:
        """Return the number of true-positive matches in this curve."""
        return int(np.sum(self.true_positive))

    @property
    def cumulative_tp(self) -> Int[np.ndarray, " num_predictions"]:
        """Return cumulative true positives in descending score order."""
        return np.cumsum(self.true_positive)

    @property
    def cumulative_fp(self) -> Int[np.ndarray, " num_predictions"]:
        """Return cumulative false positives in descending score order."""
        return np.cumsum(self.false_positive)

    @property
    def cumulative_heading_tp(self) -> Float64[np.ndarray, " num_predictions"]:
        """Return cumulative heading-weighted true positives in score order."""
        return np.cumsum(self.heading_score)


@dataclass(frozen=True)
class CurveMetrics:
    """AP-style summary values derived from one match curve."""

    ap: float
    aph: float
    max_f1: float
    optimal_conf: float
    optimal_index: int
    optimal_recall: float
    optimal_precision: float


class TpErrorSet(BaseModel):
    """
    The nuScenes true-positive errors of one selection of matches.

    Attributes:
      ate: Average translation error, the BEV center distance in meters.
      aoe: Average orientation error, the absolute yaw gap in radians.
      ase: Average scale error, one minus the aligned 3D IoU.
      ave: Average velocity error in meters per second.
      aae: Average attribute error, the neutral 1.0 while attributes are not evaluated.
    """

    model_config = ConfigDict(frozen=True, strict=True)

    ate: float
    aoe: float
    ase: float
    ave: float
    aae: float

    @classmethod
    def nan(cls) -> TpErrorSet:
        """
        The error set of a selection without a measurable match.

        Returns:
          TpErrorSet: Every error NaN.
        """
        nan = float("nan")
        return cls(ate=nan, aoe=nan, ase=nan, ave=nan, aae=nan)

    def named(self) -> tuple[tuple[str, float], ...]:
        """
        The errors with their nuScenes names, in the nuScenes order.

        Returns:
          tuple[tuple[str, float], ...]: ``(name, value)`` pairs, ``ATE`` first.
        """
        return (
            ("ATE", self.ate),
            ("AOE", self.aoe),
            ("ASE", self.ase),
            ("AVE", self.ave),
            ("AAE", self.aae),
        )


class SelectedTpErrors(BaseModel):
    """
    True-positive errors selected at one operating point.

    Attributes:
      count: Number of true positives the errors are averaged over.
      errors: The mean errors over them, every error 1.0 when no true positive was selected.
    """

    model_config = ConfigDict(frozen=True, strict=True)

    count: int
    errors: TpErrorSet


class RecallTarget(BaseModel):
    """
    A recall operating point the true-positive errors are selected up to.

    Attributes:
      name: Variant name carried by the metric keys, for example ``default``. ``optimal`` is
        reserved for the max-F1 operating point that is always reported.
      recall: Recall level in ``[0, 1]`` up to which the matches are pooled.
    """

    model_config = ConfigDict(frozen=True, strict=True)

    name: str
    recall: float

    @model_validator(mode="after")
    def _check_name_and_range(self) -> RecallTarget:
        """Reject the reserved name and a recall outside the unit range."""
        if self.name == "optimal":
            raise ValueError('"optimal" is the reserved max-F1 operating point, pick another name.')
        if not 0.0 <= self.recall <= 1.0:
            raise ValueError(f"recall target {self.name!r} must be in [0, 1], got {self.recall}.")
        return self
