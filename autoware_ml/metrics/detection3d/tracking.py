"""Multi-object tracking metrics of a temporal 3D detector.

The suite groups the frames of every stream in time order and matches the tracked boxes to the
ground truth of the same class by BEV center distance, keeping last frame's pairs while they
stay in reach (CLEAR MOT). It reports MOTA, ID switches and fragmentations overall and per class
at the track score threshold, and the best MOTA over a sweep of score thresholds.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from autoware_ml.metrics.base import EvalStage, Metric, MetricFilter, MetricRange, MetricSuite


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

    gt_xy: np.ndarray
    gt_labels: np.ndarray
    gt_ids: np.ndarray
    pred_xy: np.ndarray
    pred_labels: np.ndarray
    pred_ids: np.ndarray
    pred_scores: np.ndarray

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


@dataclass(frozen=True)
class TrackingState:
    """Frames of every stream, each stream in time order."""

    streams: list[list[TrackingFrame]]
    class_names: tuple[str, ...]
    match_distance: float
    score_threshold: float
    sweep_score_thresholds: tuple[float, ...]


class TrackingSummary(Metric[TrackingState]):
    """MOTA, ID switches and fragmentations, overall and per class."""

    def evaluate(self, state: TrackingState, stage: EvalStage) -> dict[str, float]:
        """Run the CLEAR MOT matching over every stream.

        Args:
            state: Frames of every stream.
            stage: Evaluation stage being reported.

        Returns:
            Tracking metric keys mapped to values.
        """
        counts = _clear_mot_counts(state, state.score_threshold)
        report: dict[str, float] = {}
        totals = _totals(counts)
        for label, label_counts in counts.items():
            if label < len(state.class_names) and label_counts["num_gt"] > 0:
                report[f"mota_{state.class_names[label]}"] = _mota(label_counts)
                report[f"id_switches_{state.class_names[label]}"] = float(
                    label_counts["id_switches"]
                )
        num_gt = max(totals["num_gt"], 1)
        report.update(
            {
                "mota": _mota(totals),
                "recall": totals["tp"] / num_gt,
                "precision": totals["tp"] / max(totals["tp"] + totals["fp"], 1),
                "id_switches": float(totals["id_switches"]),
                "fragmentations": float(totals["fragmentations"]),
                "id_switches_per_gt_track": totals["id_switches"] / max(totals["num_gt_tracks"], 1),
                "num_gt": float(totals["num_gt"]),
            }
        )
        # Best MOTA over the sweep, independent of the score calibration
        sweep = {
            threshold: _mota(_totals(_clear_mot_counts(state, threshold)))
            for threshold in state.sweep_score_thresholds
        }
        sweep[state.score_threshold] = report["mota"]
        best_threshold = max(sweep, key=sweep.__getitem__)
        report["mota_best"] = sweep[best_threshold]
        report["mota_best_score_threshold"] = best_threshold
        return report


def _clear_mot_counts(state: TrackingState, score_threshold: float) -> dict[int, dict[str, int]]:
    """Match the tracked boxes from ``score_threshold`` on and count the CLEAR MOT events per class.

    Args:
        state: Frames of every stream.
        score_threshold: Score from which a tracked box counts.

    Returns:
        Event counters of every class.
    """
    counts: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for stream in state.streams:
        last_match: dict[int, int] = {}
        tracked: dict[int, bool] = {}
        for full_frame in stream:
            frame = full_frame.above(score_threshold)
            for label in np.union1d(frame.gt_labels, frame.pred_labels).tolist():
                _match_class(
                    frame,
                    int(label),
                    state.match_distance,
                    last_match,
                    tracked,
                    counts[int(label)],
                )
    return counts


def _totals(counts: dict[int, dict[str, int]]) -> dict[str, int]:
    """Sum the event counters over the classes."""
    totals: dict[str, int] = defaultdict(int)
    for label_counts in counts.values():
        for key, value in label_counts.items():
            totals[key] += value
    return totals


def _mota(counts: dict[str, int]) -> float:
    """Multi-object tracking accuracy of the given counts."""
    errors = counts["fn"] + counts["fp"] + counts["id_switches"]
    return 1.0 - errors / max(counts["num_gt"], 1)


def _match_class(
    frame: TrackingFrame,
    label: int,
    match_distance: float,
    last_match: dict[int, int],
    tracked: dict[int, bool],
    counts: dict[str, int],
) -> None:
    """Match the boxes of one class in one frame and update the counts.

    Args:
        frame: Boxes of the frame.
        label: Class to match.
        match_distance: Largest BEV center distance of a match in meters.
        last_match: Track every ground truth instance was last matched to, updated in place.
        tracked: Whether every ground truth instance was matched in its previous frame, updated
            in place.
        counts: Counters of the class, updated in place.
    """
    gt_mask = frame.gt_labels == label
    pred_mask = frame.pred_labels == label
    gt_xy, gt_ids = frame.gt_xy[gt_mask], frame.gt_ids[gt_mask]
    pred_xy, pred_ids = frame.pred_xy[pred_mask], frame.pred_ids[pred_mask]
    distance = np.linalg.norm(gt_xy[:, None] - pred_xy[None], axis=-1)
    reachable = distance <= match_distance

    pairs: dict[int, int] = {}
    for gt_index, gt_id in enumerate(gt_ids.tolist()):
        previous = np.flatnonzero(pred_ids == last_match.get(gt_id, -1))
        if previous.size and reachable[gt_index, previous[0]]:
            pairs[gt_index] = int(previous[0])
    free_gt = [index for index in range(gt_ids.shape[0]) if index not in pairs]
    free_pred = [index for index in range(pred_ids.shape[0]) if index not in pairs.values()]
    if free_gt and free_pred:
        cost = np.where(
            reachable[np.ix_(free_gt, free_pred)], distance[np.ix_(free_gt, free_pred)], 1e6
        )
        for row, col in zip(*linear_sum_assignment(cost), strict=True):
            if cost[row, col] < 1e6:
                pairs[free_gt[row]] = free_pred[col]

    for gt_index, gt_id in enumerate(gt_ids.tolist()):
        if gt_id not in tracked:
            counts["num_gt_tracks"] += 1
        matched = gt_index in pairs
        if matched:
            track_id = int(pred_ids[pairs[gt_index]])
            if gt_id in last_match and last_match[gt_id] != track_id:
                counts["id_switches"] += 1
            if gt_id in tracked and not tracked[gt_id] and gt_id in last_match:
                counts["fragmentations"] += 1
            last_match[gt_id] = track_id
        tracked[gt_id] = matched
    counts["num_gt"] += int(gt_ids.shape[0])
    counts["tp"] += len(pairs)
    counts["fn"] += int(gt_ids.shape[0]) - len(pairs)
    counts["fp"] += int(pred_ids.shape[0]) - len(pairs)


class TrackingMetricSuite(MetricSuite[TrackingState]):
    """Accumulate the tracked boxes of a temporal detector per stream."""

    prefix = "track"
    _required_keys = (
        "predictions",
        "gt_boxes",
        "gt_labels",
        "gt_instance_ids",
        "gt_observed",
        "stream_ids",
        "timestamp",
        "padding_mask",
    )

    def __init__(
        self,
        components: list[Metric[TrackingState]],
        class_names: tuple[str, ...],
        score_threshold: float,
        sweep_score_thresholds: tuple[float, ...] = (),
        match_distance: float = 2.0,
        **kwargs: Any,
    ) -> None:
        """Register the per frame accumulation states.

        Args:
            components: Injected metrics, as in :class:`MetricSuite`.
            class_names: Class names in label order.
            score_threshold: Score from which a tracked box counts.
            sweep_score_thresholds: Further score thresholds searched for the best MOTA.
            match_distance: Largest BEV center distance of a match in meters.
            **kwargs: Forwarded to :class:`MetricSuite`.
        """
        super().__init__(components=components, ranges=(), **kwargs)
        self.class_names = tuple(class_names)
        self.score_threshold = float(score_threshold)
        self.sweep_score_thresholds = tuple(float(value) for value in sweep_score_thresholds)
        self.min_score_threshold = min((self.score_threshold, *self.sweep_score_thresholds))
        self.match_distance = float(match_distance)
        for name in (
            "padding",
            "stream_ids",
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

    def update(self, eval_out: dict[str, Any]) -> None:
        """Accumulate the observed ground truth and the confident tracked boxes of a batch.

        Args:
            eval_out: Flat eval-output dict built by the model for one batch.
        """
        for index, prediction in enumerate(eval_out["predictions"]):
            if prediction["instance_ids"] is None:
                raise ValueError("Tracking metrics need the track id of every predicted box.")
            observed = eval_out["gt_observed"][index]
            boxes = eval_out["gt_boxes"][index].detach()
            device = boxes.device
            keep = (prediction["scores_3d"] >= self.min_score_threshold) & (
                prediction["instance_ids"] >= 0
            )
            # A padding frame keeps its entry so every rank gathers as many, the score skips it
            self.padding.append(
                torch.tensor([bool(eval_out["padding_mask"][index])], device=device)
            )
            self.stream_ids.append(eval_out["stream_ids"][index].reshape(1).to(device))
            self.timestamps.append(
                torch.as_tensor(
                    eval_out["timestamp"][index], dtype=torch.float64, device=device
                ).reshape(1)
            )
            self.gt_xy.append(boxes[observed, :2].float())
            self.gt_labels.append(eval_out["gt_labels"][index][observed].long())
            self.gt_ids.append(eval_out["gt_instance_ids"][index][observed].long())
            self.pred_xy.append(prediction["bboxes_3d"][keep, :2].detach().float())
            self.pred_labels.append(prediction["labels_3d"][keep].long())
            self.pred_ids.append(prediction["instance_ids"][keep].long())
            self.pred_scores.append(prediction["scores_3d"][keep].detach().float())

    def state_for(
        self, metric_range: MetricRange | None, metric_filter: MetricFilter
    ) -> TrackingState:
        """Group the frames by stream and order every stream in time, without padding frames.

        Args:
            metric_range: Unused, the suite reports no range windows.
            metric_filter: Unused, the suite reports no filters.

        Returns:
            Frames of every stream.
        """
        streams: dict[int, list[tuple[float, TrackingFrame]]] = defaultdict(list)
        for index in range(len(self.stream_ids)):
            if bool(self.padding[index]):
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
            streams[int(self.stream_ids[index])].append((float(self.timestamps[index]), frame))
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
