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

"""Multi-object tracking metrics of a temporal 3D detector.

The suite groups the frames of every stream in time order and matches the tracked boxes to the
ground truth of the same class by BEV center distance, keeping last frame's pairs while they
stay in reach (CLEAR MOT). It reports MOTA, ID switches and fragmentations overall and per class
at the track score threshold, and the best MOTA over a sweep of score thresholds.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from jaxtyping import Float, Int64
from scipy.optimize import linear_sum_assignment

from autoware_ml.metrics.base import EvalStage, Metric, MetricFilter, MetricRange, MetricSuite
from autoware_ml.geometry.bbox_3d.instance_ids import hash_instance_ids
from autoware_ml.metrics.detection3d.matching import gt_keep_mask, ignored_prediction_mask
from autoware_ml.metrics.detection3d.observation import StreamObserver, observe_detection_frame
from autoware_ml.metrics.eval_output import EvalOutput, MetricRequirements
from autoware_ml.metrics.report import MetricReport


@dataclass(frozen=True)
class TrackingFrame:
    """Ground truth and tracked boxes of one frame.

    Attributes:
        gt_xy: BEV centers of the ground truth boxes ``(G, 2)``.
        gt_labels: Classes of the ground truth boxes ``(G,)``.
        gt_ids: Instance ids of the ground truth boxes ``(G,)``.
        pred_xy: BEV centers of the tracked boxes ``(P, 2)``.
        pred_labels: Classes of the tracked boxes ``(P,)``.
        pred_ids: Track ids of the tracked boxes ``(P,)``.
        pred_scores: Scores of the tracked boxes ``(P,)``.
    """

    gt_xy: Float[np.ndarray, "num_gts 2"]
    gt_labels: Int64[np.ndarray, " num_gts"]
    gt_ids: Int64[np.ndarray, " num_gts"]
    pred_xy: Float[np.ndarray, "num_preds 2"]
    pred_labels: Int64[np.ndarray, " num_preds"]
    pred_ids: Int64[np.ndarray, " num_preds"]
    pred_scores: Float[np.ndarray, " num_preds"]

    def above(self, score_threshold: float) -> TrackingFrame:
        """Return the frame without the tracked boxes scored below ``score_threshold``."""
        keep = self.pred_scores >= score_threshold
        return TrackingFrame(
            gt_xy=self.gt_xy,
            gt_labels=self.gt_labels,
            gt_ids=self.gt_ids,
            pred_xy=self.pred_xy[keep],
            pred_labels=self.pred_labels[keep],
            pred_ids=self.pred_ids[keep],
            pred_scores=self.pred_scores[keep],
        )


@dataclass
class ClearMotCounts:
    """CLEAR MOT event counts, which the matching of every frame adds to in place.

    Attributes:
        tp: Ground truth boxes matched to a tracked box.
        fp: Tracked boxes matched to no ground truth box.
        fn: Ground truth boxes matched to no tracked box.
        id_switches: Matches whose track differs from the last track of their object.
        fragmentations: Matches of an object that its previous frame left unmatched.
        num_gt: Ground truth boxes.
        num_gt_tracks: Ground truth objects.
    """

    tp: int = 0
    fp: int = 0
    fn: int = 0
    id_switches: int = 0
    fragmentations: int = 0
    num_gt: int = 0
    num_gt_tracks: int = 0

    def __add__(self, other: ClearMotCounts) -> ClearMotCounts:
        """Sum of the counts of two sets of events."""
        return ClearMotCounts(
            tp=self.tp + other.tp,
            fp=self.fp + other.fp,
            fn=self.fn + other.fn,
            id_switches=self.id_switches + other.id_switches,
            fragmentations=self.fragmentations + other.fragmentations,
            num_gt=self.num_gt + other.num_gt,
            num_gt_tracks=self.num_gt_tracks + other.num_gt_tracks,
        )

    @property
    def mota(self) -> float:
        """Multi-object tracking accuracy, NaN without ground truth."""
        return 1.0 - _ratio(self.fn + self.fp + self.id_switches, self.num_gt)

    @property
    def recall(self) -> float:
        """Share of the ground truth boxes that are matched, NaN without ground truth."""
        return _ratio(self.tp, self.num_gt)

    @property
    def precision(self) -> float:
        """Share of the tracked boxes that are matched, NaN without a tracked box."""
        return _ratio(self.tp, self.tp + self.fp)


@dataclass(frozen=True)
class TrackingState:
    """Frames of every stream, each stream in time order."""

    streams: list[list[TrackingFrame]]
    class_names: tuple[str, ...]
    match_distance: float
    score_threshold: float
    sweep_score_thresholds: tuple[float, ...]


class TrackingSummary(Metric[TrackingState]):
    """MOTA, ID switches and fragmentations, overall and per class, NaN where a count is zero."""

    def evaluate(self, state: TrackingState, stage: EvalStage) -> MetricReport:
        """Run the CLEAR MOT matching over every stream.

        Args:
            state: Frames of every stream.
            stage: Evaluation stage being reported.

        Returns:
            The tracking report.
        """
        counts = _clear_mot_counts(state, state.score_threshold)
        report = MetricReport()
        totals = _totals(counts)
        for label, label_counts in counts.items():
            if label_counts.num_gt > 0:
                name = state.class_names[label]
                report.record(f"mota_{name}", label_counts.mota)
                report.record(f"id_switches_{name}", float(label_counts.id_switches))
        mota = totals.mota
        report.record("mota", mota)
        report.record("recall", totals.recall)
        report.record("precision", totals.precision)
        report.record("id_switches", float(totals.id_switches))
        report.record("fragmentations", float(totals.fragmentations))
        report.record("id_switches_per_gt_track", _ratio(totals.id_switches, totals.num_gt_tracks))
        report.record("num_gt", float(totals.num_gt))
        # Best MOTA over the sweep, independent of the score calibration
        sweep = {
            threshold: _totals(_clear_mot_counts(state, threshold)).mota
            for threshold in state.sweep_score_thresholds
        }
        sweep[state.score_threshold] = mota
        best_threshold = max(
            sweep, key=lambda threshold: np.nan_to_num(sweep[threshold], nan=-np.inf)
        )
        report.record("mota_best", sweep[best_threshold])
        report.record("mota_best_score_threshold", best_threshold)
        return report


def _clear_mot_counts(state: TrackingState, score_threshold: float) -> dict[int, ClearMotCounts]:
    """Match the tracked boxes from ``score_threshold`` on and count the CLEAR MOT events per class.

    Args:
        state: Frames of every stream.
        score_threshold: Score from which a tracked box counts.

    Returns:
        Event counts of every class.
    """
    counts: dict[int, ClearMotCounts] = defaultdict(ClearMotCounts)
    for stream in state.streams:
        last_match: dict[int, int] = {}
        tracked: dict[int, bool] = {}
        previous_pairs: dict[int, int] = {}
        for full_frame in stream:
            frame = full_frame.above(score_threshold)
            pairs: dict[int, int] = {}
            for label in np.union1d(frame.gt_labels, frame.pred_labels).tolist():
                pairs |= _match_class(
                    frame,
                    int(label),
                    state.match_distance,
                    previous_pairs,
                    last_match,
                    tracked,
                    counts[int(label)],
                )
            previous_pairs = pairs
    return counts


def _totals(counts: dict[int, ClearMotCounts]) -> ClearMotCounts:
    """Sum the event counts over the classes."""
    return sum(counts.values(), ClearMotCounts())


def _ratio(numerator: int, denominator: int) -> float:
    """Quotient of two counts, NaN for a zero denominator."""
    return numerator / denominator if denominator else math.nan


def _match_class(
    frame: TrackingFrame,
    label: int,
    match_distance: float,
    previous_pairs: dict[int, int],
    last_match: dict[int, int],
    tracked: dict[int, bool],
    counts: ClearMotCounts,
) -> dict[int, int]:
    """Match the boxes of one class in one frame and update the counts.

    Args:
        frame: Boxes of the frame.
        label: Class to match.
        match_distance: Largest BEV center distance of a match in meters.
        previous_pairs: Track every ground truth instance was matched to in the previous frame
            of the stream, the only pairs that carry over.
        last_match: Track every ground truth instance was last matched to, updated in place.
        tracked: Whether every ground truth instance was matched in its previous frame, updated
            in place.
        counts: Event counts of the class, updated in place.

    Returns:
        The track every matched ground truth instance of the class was paired with.
    """
    gt_mask = frame.gt_labels == label
    pred_mask = frame.pred_labels == label
    gt_xy, gt_ids = frame.gt_xy[gt_mask], frame.gt_ids[gt_mask]
    pred_xy, pred_ids = frame.pred_xy[pred_mask], frame.pred_ids[pred_mask]
    distance = np.linalg.norm(gt_xy[:, None] - pred_xy[None], axis=-1)
    reachable = distance <= match_distance

    pairs: dict[int, int] = {}
    claimed: set[int] = set()
    for gt_index, gt_id in enumerate(gt_ids.tolist()):
        previous = np.flatnonzero(pred_ids == previous_pairs.get(gt_id, -1))
        if previous.size and int(previous[0]) not in claimed and reachable[gt_index, previous[0]]:
            pairs[gt_index] = int(previous[0])
            claimed.add(int(previous[0]))
    free_gt = [index for index in range(gt_ids.shape[0]) if index not in pairs]
    free_pred = [index for index in range(pred_ids.shape[0]) if index not in claimed]
    if free_gt and free_pred:
        cost = np.where(
            reachable[np.ix_(free_gt, free_pred)], distance[np.ix_(free_gt, free_pred)], 1e6
        )
        for row, col in zip(*linear_sum_assignment(cost), strict=True):
            if cost[row, col] < 1e6:
                pairs[free_gt[row]] = free_pred[col]

    for gt_index, gt_id in enumerate(gt_ids.tolist()):
        if gt_id not in tracked:
            counts.num_gt_tracks += 1
        matched = gt_index in pairs
        if matched:
            track_id = int(pred_ids[pairs[gt_index]])
            if gt_id in last_match and last_match[gt_id] != track_id:
                counts.id_switches += 1
            if gt_id in tracked and not tracked[gt_id] and gt_id in last_match:
                counts.fragmentations += 1
            last_match[gt_id] = track_id
        tracked[gt_id] = matched
    counts.num_gt += int(gt_ids.shape[0])
    counts.tp += len(pairs)
    counts.fn += int(gt_ids.shape[0]) - len(pairs)
    counts.fp += int(pred_ids.shape[0]) - len(pairs)
    return {
        int(gt_ids[gt_index]): int(pred_ids[pred_index]) for gt_index, pred_index in pairs.items()
    }


class TrackingMetricSuite(MetricSuite[TrackingState]):
    """Accumulate the tracked boxes of a temporal detector per stream."""

    prefix = "track"
    supports_filters = False

    def __init__(
        self,
        components: list[Metric[TrackingState]],
        class_names: tuple[str, ...],
        score_threshold: float,
        min_observed_points: int,
        sweep_score_thresholds: tuple[float, ...] = (),
        match_distance: float = 2.0,
        max_distances: dict[str, float] | None = None,
        **kwargs: Any,
    ) -> None:
        """Register the per frame accumulation states.

        Args:
            components: Injected metrics, as in :class:`MetricSuite`.
            class_names: Class names in label order.
            score_threshold: Score from which a tracked box counts.
            min_observed_points: Lidar points a box needs in one frame of its stream for the
                metrics to count its object from then on, see :class:`StreamObserver`.
            sweep_score_thresholds: Further score thresholds searched for the best MOTA.
            match_distance: Largest BEV center distance of a match in meters.
            max_distances: Class name to maximum evaluation distance in meters, as in the
                detection suites. A tracked box beyond it leaves, and a ground truth box beyond
                it leaves with the tracked boxes that claim it.
            **kwargs: Forwarded to :class:`MetricSuite`.
        """
        super().__init__(components=components, **kwargs)
        self.observer = StreamObserver(min_observed_points)
        self.class_names = tuple(class_names)
        self.score_threshold = float(score_threshold)
        self.sweep_score_thresholds = tuple(float(value) for value in sweep_score_thresholds)
        self.min_score_threshold = min((self.score_threshold, *self.sweep_score_thresholds))
        self.match_distance = float(match_distance)
        self.max_distances = max_distances
        for name in (
            "unscored",
            "scene_ids",
            "timestamps",
            "gt_xy",
            "gt_labels",
            "gt_ids",
            "pred_xy",
            "pred_labels",
            "pred_ids",
            "pred_scores",
        ):
            self.add_state(name, default=[], dist_reduce_fx=None)

    def reset(self) -> None:
        """Clear the accumulated frames and forget what every stream observed."""
        super().reset()
        self.observer.reset()

    def requirements(self) -> MetricRequirements:
        """The detection frames with the identities, the point counts and the scene of the boxes.

        Returns:
            The requirements at this suite instance's stage.
        """
        return MetricRequirements(
            detection3d=True, frame_meta=True, gt_num_points=True, gt_instance_ids=True
        )

    def update(self, eval_output: EvalOutput) -> None:
        """Accumulate the observed ground truth and the confident tracked boxes of a batch.

        A query decodes into one box, so a track has at most one box per frame. A tracked box that
        claims a ground truth box the stream has not observed yet, or one beyond the distance of
        its class, leaves the evaluation with it.

        Args:
            eval_output: The typed eval output the model built for one batch.

        Raises:
            ValueError: If the output carries no detection frames, a frame lacks its tracks or
                timestamp, a label lies outside the classes, or a track holds two boxes of one
                frame.
        """
        output = eval_output.detection3d
        if output is None:
            raise ValueError(
                "The tracking suite needs the detection3d frames, the eval output carries none."
            )
        for index, frame in enumerate(output.frames):
            prediction = frame.predictions
            if prediction.instance_ids is None:
                raise ValueError("Tracking metrics need the track id of every predicted box.")
            if frame.gt_instance_ids is None or frame.frame_meta is None:
                raise ValueError(
                    f"Frame {index} lacks its object identities or its scene and timestamp."
                )
            boxes = frame.gt_boxes.detach().float()
            device = boxes.device
            self.unscored.append(torch.tensor([not frame.scored], device=device))
            # A scene streams in time order on one batch position, so it is one stream
            scene_id = hash_instance_ids([frame.frame_meta.scene_token])
            self.scene_ids.append(torch.from_numpy(scene_id).to(device))
            self.timestamps.append(
                torch.tensor([frame.frame_meta.timestamp], dtype=torch.float64, device=device)
            )
            if not frame.scored:
                # An unscored frame keeps empty entries so every rank gathers as many
                self._append_empty_frame(boxes, prediction.bboxes_3d)
                continue
            labels = frame.gt_labels.long()
            num_classes = len(self.class_names)
            for name, frame_labels in (("ground truth", labels), ("tracked", prediction.labels_3d)):
                if bool(((frame_labels < 0) | (frame_labels >= num_classes)).any()):
                    raise ValueError(
                        f"Frame {index} carries {name} labels outside the {num_classes} classes."
                    )
            in_range = gt_keep_mask(boxes, labels, None, self.class_names, self.max_distances, 0)
            observed = observe_detection_frame(self.observer, index, frame).to(device) & in_range
            tracked = prediction.instance_ids >= 0
            track_ids = prediction.instance_ids[tracked]
            if torch.unique(track_ids).numel() != track_ids.numel():
                raise ValueError(
                    f"Frame {index} gives a track more than one box, a query decodes into one box."
                )
            keep = tracked & (prediction.scores_3d >= self.min_score_threshold)
            pred_boxes = prediction.bboxes_3d[keep].detach().float()
            pred_labels = prediction.labels_3d[keep].long()
            # The class distance bounds the tracked boxes as it bounds the ground truth
            near = gt_keep_mask(
                pred_boxes, pred_labels, None, self.class_names, self.max_distances, 0
            )
            pred_boxes = pred_boxes[near]
            pred_labels = pred_labels[near]
            pred_scores = prediction.scores_3d[keep][near].detach().float()
            pred_ids = prediction.instance_ids[keep][near].long()
            # The claims run greedily in score order, one pass holds for every higher threshold
            claimed = ignored_prediction_mask(
                pred_boxes,
                pred_scores,
                pred_labels,
                boxes[observed],
                labels[observed],
                boxes[~observed],
                labels[~observed],
                self.match_distance,
            )
            self.gt_xy.append(boxes[observed, :2])
            self.gt_labels.append(labels[observed])
            self.gt_ids.append(frame.gt_instance_ids[observed].long())
            self.pred_xy.append(pred_boxes[~claimed, :2])
            self.pred_labels.append(pred_labels[~claimed])
            self.pred_ids.append(pred_ids[~claimed])
            self.pred_scores.append(pred_scores[~claimed])

    def _append_empty_frame(
        self,
        gt_boxes: Float[torch.Tensor, "num_gts box_dim"],
        pred_boxes: Float[torch.Tensor, "num_preds box_dim"],
    ) -> None:
        """Give an unscored frame empty ground truth and prediction entries.

        Args:
            gt_boxes: Ground truth boxes of the frame, for their device.
            pred_boxes: Predicted boxes of the frame, for their device.
        """
        self.gt_xy.append(gt_boxes[:0, :2])
        self.gt_labels.append(torch.zeros(0, dtype=torch.long, device=gt_boxes.device))
        self.gt_ids.append(torch.zeros(0, dtype=torch.long, device=gt_boxes.device))
        self.pred_xy.append(pred_boxes.detach().float()[:0, :2])
        self.pred_labels.append(torch.zeros(0, dtype=torch.long, device=pred_boxes.device))
        self.pred_ids.append(torch.zeros(0, dtype=torch.long, device=pred_boxes.device))
        self.pred_scores.append(torch.zeros(0, device=pred_boxes.device))

    def state_for(
        self, metric_range: MetricRange | None, metric_filter: MetricFilter
    ) -> TrackingState:
        """Group the frames by scene and order every scene in time, without unscored frames.

        Args:
            metric_range: Unused, the suite reports no range windows.
            metric_filter: Unused, the suite reports no filters.

        Returns:
            Frames of every stream.
        """
        streams: dict[int, list[tuple[float, TrackingFrame]]] = defaultdict(list)
        for index in range(len(self.scene_ids)):
            if bool(self.unscored[index]):
                continue
            frame = TrackingFrame(
                gt_xy=self.gt_xy[index].cpu().numpy(),
                gt_labels=self.gt_labels[index].cpu().numpy(),
                gt_ids=self.gt_ids[index].cpu().numpy(),
                pred_xy=self.pred_xy[index].cpu().numpy(),
                pred_labels=self.pred_labels[index].cpu().numpy(),
                pred_ids=self.pred_ids[index].cpu().numpy(),
                pred_scores=self.pred_scores[index].cpu().numpy(),
            )
            streams[int(self.scene_ids[index])].append((float(self.timestamps[index]), frame))
        return TrackingState(
            streams=[
                [frame for _, frame in sorted(frames, key=lambda item: item[0])]
                for frames in streams.values()
            ],
            class_names=self.class_names,
            match_distance=self.match_distance,
            score_threshold=self.score_threshold,
            sweep_score_thresholds=self.sweep_score_thresholds,
        )
