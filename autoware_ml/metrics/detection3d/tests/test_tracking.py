"""CLEAR MOT counts of the tracking metric suite."""

from __future__ import annotations

import torch

from autoware_ml.metrics.base import EvalStage
from autoware_ml.metrics.detection3d.tracking import TrackingMetricSuite, TrackingSummary


def _frame(
    gt_x: list[float],
    gt_ids: list[int],
    pred_x: list[float],
    pred_ids: list[int],
    pred_scores: list[float] | None = None,
) -> dict:
    """One frame of class 0 boxes on the x axis, the tracked boxes scored 0.9 by default."""

    def boxes(xs: list[float]) -> torch.Tensor:
        out = torch.zeros((len(xs), 9))
        out[:, 0] = torch.tensor(xs)
        out[:, 3:6] = 1.0
        return out

    return {
        "prediction": {
            "bboxes_3d": boxes(pred_x),
            "scores_3d": torch.tensor(pred_scores or [0.9] * len(pred_x)),
            "labels_3d": torch.zeros(len(pred_x), dtype=torch.long),
            "instance_ids": torch.tensor(pred_ids, dtype=torch.long),
        },
        "gt_boxes": boxes(gt_x),
        "gt_labels": torch.zeros(len(gt_x), dtype=torch.long),
        "gt_instance_ids": torch.tensor(gt_ids, dtype=torch.long),
        "gt_observed": torch.ones(len(gt_x), dtype=torch.bool),
    }


def _report(
    frames: list[dict], stream_id: int = 0, sweep_score_thresholds: tuple[float, ...] = ()
) -> dict[str, float]:
    suite = TrackingMetricSuite(
        components=[TrackingSummary()],
        class_names=("car",),
        score_threshold=0.5,
        sweep_score_thresholds=sweep_score_thresholds,
    )
    suite.update(
        {
            "predictions": [frame["prediction"] for frame in frames],
            "gt_boxes": [frame["gt_boxes"] for frame in frames],
            "gt_labels": [frame["gt_labels"] for frame in frames],
            "gt_instance_ids": [frame["gt_instance_ids"] for frame in frames],
            "gt_observed": [frame["gt_observed"] for frame in frames],
            "stream_ids": [torch.tensor(stream_id)] * len(frames),
            # Fed out of order, the suite sorts every stream by time
            "timestamp": [float(len(frames) - index) for index in range(len(frames))],
            "padding_mask": [False] * len(frames),
        }
    )
    return suite.result(EvalStage.VAL)


def test_a_track_switch_counts_once_and_lowers_mota() -> None:
    # Reversed time: the object is followed by track 5, then track 6 takes it over
    frames = [
        _frame([0.3], [1], [0.3], [6]),
        _frame([0.2], [1], [0.2], [6]),
        _frame([0.1], [1], [0.1], [5]),
        _frame([0.0], [1], [0.0], [5]),
    ]
    report = _report(frames)

    assert report["id_switches"] == 1.0
    assert report["mota"] == 0.75
    assert report["recall"] == 1.0
    assert report["fragmentations"] == 0.0


def test_a_missed_frame_counts_a_fragmentation_and_a_miss() -> None:
    frames = [
        _frame([0.3], [1], [0.3], [5]),
        _frame([0.2], [1], [9.0], [5]),
        _frame([0.1], [1], [0.1], [5]),
    ]
    report = _report(frames)

    assert report["id_switches"] == 0.0
    assert report["fragmentations"] == 1.0
    # One miss and one far false positive over three ground truth boxes
    assert abs(report["mota"] - (1.0 - 2.0 / 3.0)) < 1e-6


def test_the_sweep_finds_the_threshold_that_drops_weak_false_positives() -> None:
    frames = [
        _frame([0.2], [1], [0.2, 9.0], [5, 7], [0.9, 0.6]),
        _frame([0.1], [1], [0.1, 9.0], [5, 7], [0.9, 0.6]),
    ]
    report = _report(frames, sweep_score_thresholds=(0.3, 0.7))

    # The weak far tracks count at the track threshold 0.5 and vanish from 0.7 on
    assert report["mota"] == 0.0
    assert report["mota_best"] == 1.0
    assert report["mota_best_score_threshold"] == 0.7
