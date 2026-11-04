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

"""Open-loop motion metrics.

Predictions are matched to the ground truth of their class by BEV center distance in score order,
the matched pairs with a known future report the minimum average displacement over their known
steps. The final displacement, the miss rate and the end-to-end prediction accuracy (EPA) score only
the pairs whose ground truth reaches the last step of the horizon, as in SparseDrive, and the EPA
divides by the boxes known at that step. An endpoint farther than the miss distance is a miss, as in
the Argoverse 2 benchmark. The forecast summary scores the same pairs as the Argoverse 2 motion
forecasting benchmark: the best of the modes, the most likely mode and a constant velocity
baseline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from jaxtyping import Bool, Float, Int

from autoware_ml.metrics.base import EvalStage, Metric, MetricFilter, MetricRange, MetricSuite
from autoware_ml.metrics.detection3d.criticality import greedy_match
from autoware_ml.metrics.detection3d.matching import gt_keep_mask, ignored_prediction_mask
from autoware_ml.metrics.detection3d.observation import StreamObserver
from autoware_ml.metrics.eval_output import EvalOutput, MetricRequirements
from autoware_ml.metrics.report import MetricReport
from autoware_ml.types.geometry import Box3DFieldIndex


@dataclass(frozen=True)
class MotionFrame:
    """Predicted and ground truth trajectories of one frame, positions in the frame.

    Attributes:
        pred_xy: BEV centers of the predicted agents ``(P, 2)``.
        pred_labels: Classes of the predicted agents ``(P,)``.
        pred_scores: Detection scores of the predicted agents ``(P,)``.
        pred_trajectories: Trajectory of every mode ``(P, M, T, 2)``.
        pred_mode_scores: Probability of every mode ``(P, M)``.
        pred_velocities: BEV velocity in m/s the detection estimates ``(P, 2)``.
        gt_xy: BEV centers of the ground truth agents ``(G, 2)``.
        gt_labels: Classes of the ground truth agents ``(G,)``.
        gt_futures: Future positions ``(G, T, 2)``.
        gt_valid: Whether a future position is known ``(G, T)``.
        step_seconds: Time between two future steps.
    """

    pred_xy: Float[np.ndarray, "num_preds 2"]
    pred_labels: Int[np.ndarray, " num_preds"]
    pred_scores: Float[np.ndarray, " num_preds"]
    pred_trajectories: Float[np.ndarray, "num_preds num_modes num_steps 2"]
    pred_mode_scores: Float[np.ndarray, "num_preds num_modes"]
    pred_velocities: Float[np.ndarray, "num_preds 2"]
    gt_xy: Float[np.ndarray, "num_gts 2"]
    gt_labels: Int[np.ndarray, " num_gts"]
    gt_futures: Float[np.ndarray, "num_gts num_steps 2"]
    gt_valid: Bool[np.ndarray, "num_gts num_steps"]
    step_seconds: float


@dataclass(frozen=True)
class MotionState:
    """Frames of the evaluation and the settings of the metrics."""

    frames: list[MotionFrame]
    class_names: tuple[str, ...]
    match_distance: float
    miss_distance: float


class AgentMotionSummary(Metric[MotionState]):
    """minADE, minFDE, miss rate and EPA per class and their mean over the classes."""

    def evaluate(self, state: MotionState, stage: EvalStage) -> MetricReport:
        """Score the confident predictions against every box of their class, as SparseDrive's EPA.

        Args:
            state: Frames of the evaluation.
            stage: Evaluation stage, unused.

        Returns:
            The values per class and their mean over the classes.
        """
        report = MetricReport()
        means: dict[str, list[float]] = {"minADE": [], "minFDE": [], "MR": [], "EPA": []}
        for label, name in enumerate(state.class_names):
            ade, fde, hits, false_positives, num_boxes, num_gt = [], [], 0, 0, 0, 0
            for frame in state.frames:
                pred = frame.pred_labels == label
                gt = frame.gt_labels == label
                num_boxes += int(gt.sum())
                # Only an agent known at the horizon end can be a hit, so only such agents count
                num_gt += int(frame.gt_valid[gt][:, -1].sum())
                # The suite left out the predictions on ignored boxes already
                is_tp, matched, _ = greedy_match(
                    frame.gt_xy[gt],
                    np.zeros((0, 2)),
                    frame.pred_xy[pred],
                    frame.pred_scores[pred],
                    state.match_distance,
                )
                matched_pred = np.flatnonzero(is_tp)
                matched_gt = matched[matched_pred]
                false_positives += int(pred.sum()) - len(matched_pred)
                # A match without a known future is no false positive but can never be a hit
                known = frame.gt_valid[gt][matched_gt].any(axis=1)
                trajectories = frame.pred_trajectories[pred][matched_pred][known]
                futures = frame.gt_futures[gt][matched_gt][known]
                valid = frame.gt_valid[gt][matched_gt][known]
                for modes, future, mask in zip(trajectories, futures, valid, strict=True):
                    errors = np.linalg.norm(modes - future[None], axis=-1)
                    ade.append(float((errors[:, mask]).mean(axis=1).min()))
                    if mask[-1]:
                        fde.append(float(errors[:, -1].min()))
                        hits += int(fde[-1] <= state.miss_distance)
            if num_boxes == 0:
                continue
            values = {
                "minADE": float(np.mean(ade)) if ade else float("nan"),
                "minFDE": float(np.mean(fde)) if fde else float("nan"),
                "MR": float(np.mean(np.asarray(fde) > state.miss_distance))
                if fde
                else float("nan"),
                "EPA": (hits - 0.5 * false_positives) / num_gt if num_gt else float("nan"),
            }
            for key, value in values.items():
                report.record(f"{key}_{name}", value)
                if np.isfinite(value):
                    means[key].append(value)
        for key, values in means.items():
            if values:
                report.record(key, float(np.mean(values)))
        return report


@dataclass(frozen=True)
class ForecastPair:
    """One matched agent with a known future, as the forecast summary scores it.

    Attributes:
        label: Class of the agent.
        modes: Predicted trajectory of every mode ``(M, T, 2)``.
        mode_scores: Probability of every mode ``(M,)``.
        constant_velocity: Trajectory of the detection at its own velocity ``(T, 2)``.
        future: Ground truth future ``(T, 2)``.
        valid: Whether a future position is known ``(T,)``.
    """

    label: int
    modes: Float[np.ndarray, "num_modes num_steps 2"]
    mode_scores: Float[np.ndarray, " num_modes"]
    constant_velocity: Float[np.ndarray, "num_steps 2"]
    future: Float[np.ndarray, "num_steps 2"]
    valid: Bool[np.ndarray, " num_steps"]


@dataclass(frozen=True)
class ForecastErrors:
    """Displacement errors of one agent.

    Attributes:
        best_ade: Average displacement of the mode whose endpoint is closest.
        best_fde: Endpoint displacement of that mode.
        top_ade: Average displacement of the most likely mode.
        top_fde: Endpoint displacement of the most likely mode.
        cv_ade: Average displacement of the constant velocity baseline.
        cv_fde: Endpoint displacement of the constant velocity baseline.
    """

    best_ade: float
    best_fde: float
    top_ade: float
    top_fde: float
    cv_ade: float
    cv_fde: float


class AgentForecastSummary(Metric[MotionState]):
    """Motion forecasting metrics of the Argoverse 2 benchmark over the matched agents.

    Every matched agent whose position at the last step is known counts once, as the benchmark
    scores the agents with a full future.
    ``minADE<K>`` and ``minFDE<K>`` are the average and endpoint displacement of the mode whose
    endpoint is closest, ``MR<K>`` the share of agents whose endpoint is farther than the miss
    distance. ``minADE1``, ``minFDE1`` and ``MR1`` score the most likely mode alone. ``cvADE``,
    ``cvFDE`` and ``cvMR`` score the detection extrapolated at its own velocity, the straight line
    baseline, and ``advantage`` is ``cvADE`` over ``minADE<K>``. Every value is reported over all
    agents and per class, with the number of agents.
    """

    def evaluate(self, state: MotionState, stage: EvalStage) -> MetricReport:
        """Score every matched agent with a full future.

        Args:
            state: Frames of the evaluation.
            stage: Evaluation stage, unused.

        Returns:
            The values over all agents and per class.
        """
        errors = [(pair.label, forecast_errors(pair)) for pair in self.pairs(state)]
        report = MetricReport()
        if not errors:
            return report
        num_modes = int(state.frames[0].pred_trajectories.shape[1])
        record_forecast(
            report, [error for _, error in errors], num_modes, state.miss_distance, suffix=""
        )
        for label, name in enumerate(state.class_names):
            of_class = [error for error_label, error in errors if error_label == label]
            if of_class:
                record_forecast(report, of_class, num_modes, state.miss_distance, f"_{name}")
        return report

    def pairs(self, state: MotionState) -> list[ForecastPair]:
        """Match the confident predictions to the boxes of their class, frame by frame.

        Args:
            state: Frames of the evaluation.

        Returns:
            The matched agents the summary counts.
        """
        pairs = []
        for frame in state.frames:
            steps = frame.pred_trajectories.shape[2]
            times = frame.step_seconds * np.arange(1, steps + 1)
            for label in range(len(state.class_names)):
                pred = np.flatnonzero(frame.pred_labels == label)
                gt = np.flatnonzero(frame.gt_labels == label)
                # The suite left out the predictions on ignored boxes already
                is_tp, matched, _ = greedy_match(
                    frame.gt_xy[gt],
                    np.zeros((0, 2)),
                    frame.pred_xy[pred],
                    frame.pred_scores[pred],
                    state.match_distance,
                )
                for pred_row in np.flatnonzero(is_tp):
                    p, g = pred[pred_row], gt[matched[pred_row]]
                    valid = frame.gt_valid[g]
                    if not valid[-1]:
                        continue
                    pairs.append(
                        ForecastPair(
                            label=label,
                            modes=frame.pred_trajectories[p],
                            mode_scores=frame.pred_mode_scores[p],
                            constant_velocity=frame.pred_xy[p]
                            + times[:, None] * frame.pred_velocities[p],
                            future=frame.gt_futures[g],
                            valid=valid,
                        )
                    )
        return pairs


def record_forecast(
    report: MetricReport,
    errors: list[ForecastErrors],
    num_modes: int,
    miss_distance: float,
    suffix: str,
) -> None:
    """Record the means and miss rates of a group of agents.

    Args:
        report: Report receiving the values.
        errors: Errors of every agent of the group.
        num_modes: Number of modes, which names the best of the modes values.
        miss_distance: Endpoint displacement beyond which an agent is a miss.
        suffix: Suffix of the value names, empty for all agents.
    """

    def mean(values: list[float]) -> float:
        return float(np.mean(values))

    def misses(endpoint_errors: list[float]) -> float:
        return float(np.mean(np.asarray(endpoint_errors) > miss_distance))

    best_ade = mean([error.best_ade for error in errors])
    cv_ade = mean([error.cv_ade for error in errors])
    values = {
        f"minADE{num_modes}": best_ade,
        f"minFDE{num_modes}": mean([error.best_fde for error in errors]),
        f"MR{num_modes}": misses([error.best_fde for error in errors]),
        "minADE1": mean([error.top_ade for error in errors]),
        "minFDE1": mean([error.top_fde for error in errors]),
        "MR1": misses([error.top_fde for error in errors]),
        "cvADE": cv_ade,
        "cvFDE": mean([error.cv_fde for error in errors]),
        "cvMR": misses([error.cv_fde for error in errors]),
        # An exact forecast has no finite ratio
        "advantage": cv_ade / best_ade if best_ade > 0 else float("nan"),
        "agents": float(len(errors)),
    }
    for key, value in values.items():
        report.record(f"{key}{suffix}", value)


def forecast_errors(pair: ForecastPair) -> ForecastErrors:
    """Displacement errors of the best mode, the most likely mode and the baseline of an agent.

    Args:
        pair: The matched agent.

    Returns:
        Its errors over the known steps, the endpoint being the last step.
    """
    errors = np.linalg.norm(pair.modes - pair.future[None], axis=-1)
    best = int(np.argmin(errors[:, -1]))
    top = int(np.argmax(pair.mode_scores))
    cv_errors = np.linalg.norm(pair.constant_velocity - pair.future, axis=-1)
    return ForecastErrors(
        best_ade=float(errors[best, pair.valid].mean()),
        best_fde=float(errors[best, -1]),
        top_ade=float(errors[top, pair.valid].mean()),
        top_fde=float(errors[top, -1]),
        cv_ade=float(cv_errors[pair.valid].mean()),
        cv_fde=float(cv_errors[-1]),
    )


class MotionMetricSuite(MetricSuite[MotionState]):
    """Accumulate the predicted and ground truth trajectories of every frame."""

    prefix = "motion"
    supports_filters = False

    def __init__(
        self,
        components: list[Metric[MotionState]],
        class_names: tuple[str, ...],
        score_threshold: float,
        min_observed_points: int,
        match_distance: float = 2.0,
        miss_distance: float = 2.0,
        max_distances: dict[str, float] | None = None,
        **kwargs: Any,
    ) -> None:
        """Register the per frame accumulation state.

        Args:
            components: Injected metrics, as in :class:`MetricSuite`.
            class_names: Class names in label order.
            score_threshold: Detection score from which a prediction counts.
            min_observed_points: Lidar points a box needs in one frame of its stream for the
                metrics to count its agent from then on, see :class:`StreamObserver`.
            match_distance: Largest BEV center distance of a match in meters.
            miss_distance: Final displacement from which a matched agent is a miss.
            max_distances: Class name to maximum evaluation distance in meters, as in the
                tracking suite.
            **kwargs: Forwarded to :class:`MetricSuite`.
        """
        super().__init__(components=components, **kwargs)
        self.observer = StreamObserver(min_observed_points)
        self.class_names = tuple(class_names)
        self.score_threshold = float(score_threshold)
        self.match_distance = float(match_distance)
        self.miss_distance = float(miss_distance)
        self.max_distances = max_distances
        for name in (
            "unscored",
            "step_seconds",
            "pred_xy",
            "pred_labels",
            "pred_scores",
            "pred_trajectories",
            "pred_mode_scores",
            "pred_velocities",
            "gt_xy",
            "gt_labels",
            "gt_futures",
            "gt_valid",
        ):
            self.add_state(name, default=[], dist_reduce_fx=None)

    def reset(self) -> None:
        """Clear the accumulated frames and forget what every stream observed."""
        super().reset()
        self.observer.reset()

    def requirements(self) -> MetricRequirements:
        """The motion frames.

        Returns:
            The requirements at this suite instance's stage.
        """
        return MetricRequirements(motion=True)

    def update(self, eval_output: EvalOutput) -> None:
        """Keep the frames of a batch, one tensor per field and frame.

        The boxes and the predictions follow the tracking suite: a box beyond the distance of its
        class, or not observed in its stream yet, is ignored with the predictions that claim it,
        and a prediction beyond the distance of its class leaves.

        Args:
            eval_output: The typed eval output the model built for one batch.

        Raises:
            ValueError: If the output carries no motion frames.
        """
        if eval_output.motion is None:
            raise ValueError("The motion suite needs the motion frames, the eval output has none.")
        for lane, frame in enumerate(eval_output.motion.frames):
            prediction = frame.predictions
            pred_boxes = prediction.bboxes_3d.detach().float()
            gt_boxes = frame.gt_boxes.detach().float()
            device = gt_boxes.device
            self.unscored.append(torch.tensor([not frame.scored], device=device))
            self.step_seconds.append(
                torch.tensor([frame.step_seconds], dtype=torch.float64, device=device)
            )
            if frame.scored:
                gt_labels = frame.gt_labels.long()
                pred_labels = prediction.labels_3d.long()
                gt_keep = gt_keep_mask(
                    gt_boxes, gt_labels, None, self.class_names, self.max_distances, 0
                ) & self.observer.observe(
                    lane,
                    frame.frame_meta.scene_token,
                    frame.gt_instance_ids,
                    frame.gt_num_points,
                ).to(device)
                # The metrics read the predictions at or above the score threshold only
                confident = (prediction.scores_3d >= self.score_threshold) & gt_keep_mask(
                    pred_boxes, pred_labels, None, self.class_names, self.max_distances, 0
                )
                pred_index = confident.nonzero().flatten()
                claimed = ignored_prediction_mask(
                    pred_boxes[pred_index],
                    prediction.scores_3d[pred_index],
                    pred_labels[pred_index],
                    gt_boxes[gt_keep],
                    gt_labels[gt_keep],
                    gt_boxes[~gt_keep],
                    gt_labels[~gt_keep],
                    self.match_distance,
                )
                pred_index = pred_index[~claimed.to(pred_index.device)]
            else:
                # An unscored frame keeps empty entries so every rank gathers as many
                gt_keep = torch.zeros(gt_boxes.shape[0], dtype=torch.bool, device=device)
                pred_index = torch.zeros(0, dtype=torch.long, device=pred_boxes.device)
            self.pred_xy.append(pred_boxes[pred_index, :2])
            self.pred_labels.append(prediction.labels_3d[pred_index].long())
            self.pred_scores.append(prediction.scores_3d[pred_index].detach().float())
            self.pred_trajectories.append(frame.pred_trajectories[pred_index].detach().float())
            self.pred_mode_scores.append(frame.pred_mode_scores[pred_index].detach().float())
            self.pred_velocities.append(
                pred_boxes[pred_index, Box3DFieldIndex.VELOCITY_X : Box3DFieldIndex.VELOCITY_Y + 1]
            )
            self.gt_xy.append(gt_boxes[gt_keep, :2])
            self.gt_labels.append(frame.gt_labels[gt_keep].long())
            self.gt_futures.append(frame.gt_futures[gt_keep].detach().float())
            self.gt_valid.append(frame.gt_valid[gt_keep])

    def state_for(
        self, metric_range: MetricRange | None, metric_filter: MetricFilter
    ) -> MotionState:
        """Collect the frames with the settings of the metrics, without unscored frames.

        Args:
            metric_range: Unused, the suite reports no range windows.
            metric_filter: Unused, the suite reports no filters.

        Returns:
            The frames of the evaluation.
        """
        frames = [
            self.frame(index)
            for index in range(len(self.unscored))
            if not bool(self.unscored[index])
        ]
        return MotionState(
            frames=frames,
            class_names=self.class_names,
            match_distance=self.match_distance,
            miss_distance=self.miss_distance,
        )

    def frame(self, index: int) -> MotionFrame:
        """Build the accumulated frame at an index."""
        return MotionFrame(
            pred_xy=self.pred_xy[index].cpu().numpy(),
            pred_labels=self.pred_labels[index].cpu().numpy(),
            pred_scores=self.pred_scores[index].cpu().numpy(),
            pred_trajectories=self.pred_trajectories[index].cpu().numpy(),
            pred_mode_scores=self.pred_mode_scores[index].cpu().numpy(),
            pred_velocities=self.pred_velocities[index].cpu().numpy(),
            gt_xy=self.gt_xy[index].cpu().numpy(),
            gt_labels=self.gt_labels[index].cpu().numpy(),
            gt_futures=self.gt_futures[index].cpu().numpy(),
            gt_valid=self.gt_valid[index].cpu().numpy(),
            step_seconds=float(self.step_seconds[index]),
        )
