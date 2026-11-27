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

"""CLEAR MOT counts of the tracking metric suite."""

from __future__ import annotations

import math
import unittest

import torch

from autoware_ml.dataclasses.batch.frame_meta import FrameMetaSample
from autoware_ml.dataclasses.models.detection3d.predictions import Detection3DSamplePredictions
from autoware_ml.metrics.base import EvalStage
from autoware_ml.metrics.filters import CorridorFilter
from autoware_ml.metrics.detection3d.tracking import TrackingMetricSuite, TrackingSummary
from autoware_ml.metrics.eval_output import (
    Detection3DEvalFrame,
    Detection3DEvalOutput,
    EvalOutput,
)


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
        "gt_num_points": torch.full((len(gt_x),), 5, dtype=torch.long),
    }


def _report(
    frames: list[dict],
    sweep_score_thresholds: tuple[float, ...] = (),
    max_distances: dict[str, float] | None = None,
) -> dict[str, float]:
    suite = TrackingMetricSuite(
        components=[TrackingSummary()],
        class_names=("car",),
        score_threshold=0.5,
        sweep_score_thresholds=sweep_score_thresholds,
        min_observed_points=1,
        max_distances=max_distances,
    )
    suite.update(
        EvalOutput(
            detection3d=Detection3DEvalOutput(
                frames=[
                    Detection3DEvalFrame(
                        predictions=Detection3DSamplePredictions(**frame["prediction"]),
                        gt_boxes=frame["gt_boxes"],
                        gt_labels=frame["gt_labels"],
                        gt_instance_ids=frame["gt_instance_ids"],
                        gt_num_points=frame["gt_num_points"],
                        is_padding=frame.get("is_padding", False),
                        key_frame=frame.get("key_frame", True),
                        # Fed out of order, the suite sorts every stream by time
                        frame_meta=FrameMetaSample(
                            ego2global=torch.eye(4, dtype=torch.float64),
                            scene_token=frame.get("scene", "scene"),
                            timestamp=float(len(frames) - index),
                            key_frame=frame.get("key_frame", True),
                        ),
                    )
                    for index, frame in enumerate(frames)
                ]
            )
        )
    )
    report = suite.result(EvalStage.VAL)
    return {str(key): report.metric_value(key) for key in report.metric_names()}


class TestTrackingMetricSuite(unittest.TestCase):
    def test_interleaved_streams_sharing_an_instance_id_never_switch_into_each_other(
        self,
    ) -> None:
        """
        Input: two scenes fed interleaved in batch order whose objects share instance id 1, one
          followed by track 5 and the other by track 6.
        Expected: every stream is matched on its own, so no ID switch and full recall.
        """
        frames = [
            {**_frame([0.3], [1], [0.3], [5]), "scene": "a"},
            {**_frame([0.3], [1], [0.3], [6]), "scene": "b"},
            {**_frame([0.2], [1], [0.2], [5]), "scene": "a"},
            {**_frame([0.2], [1], [0.2], [6]), "scene": "b"},
        ]
        report = _report(frames)

        self.assertEqual(report["id_switches"], 0.0)
        self.assertEqual(report["recall"], 1.0)
        self.assertEqual(report["mota"], 1.0)

    def test_frames_between_key_frames_and_padding_leave_the_counts_unchanged(self) -> None:
        """
        Input: a stream of two key frames followed by track 5, with a frame between key frames
          and a padding frame in between whose track 9 would be a switch if they were scored.
        Expected: the unscored frames are skipped, the pair of the first key frame carries over
          to the second, and the counts equal those of the stream without them.
        """
        key_frames = [_frame([0.3], [1], [0.3], [5]), _frame([0.0], [1], [0.0], [5])]
        with_unscored = [
            key_frames[0],
            {**_frame([0.2], [1], [0.2], [9]), "key_frame": False},
            {**_frame([0.1], [1], [0.1], [9]), "is_padding": True},
            key_frames[1],
        ]

        self.assertEqual(_report(with_unscored), _report(key_frames))
        self.assertEqual(_report(with_unscored)["id_switches"], 0.0)

    def test_a_track_switch_counts_once_and_lowers_mota(self) -> None:
        # Reversed time: the object is followed by track 5, then track 6 takes it over
        frames = [
            _frame([0.3], [1], [0.3], [6]),
            _frame([0.2], [1], [0.2], [6]),
            _frame([0.1], [1], [0.1], [5]),
            _frame([0.0], [1], [0.0], [5]),
        ]
        report = _report(frames)

        self.assertEqual(report["id_switches"], 1.0)
        self.assertEqual(report["mota"], 0.75)
        self.assertEqual(report["recall"], 1.0)
        self.assertEqual(report["fragmentations"], 0.0)

    def test_a_missed_frame_counts_a_fragmentation_and_a_miss(self) -> None:
        frames = [
            _frame([0.3], [1], [0.3], [5]),
            _frame([0.2], [1], [9.0], [5]),
            _frame([0.1], [1], [0.1], [5]),
        ]
        report = _report(frames)

        self.assertEqual(report["id_switches"], 0.0)
        self.assertEqual(report["fragmentations"], 1.0)
        # One miss and one far false positive over three ground truth boxes
        self.assertAlmostEqual(report["mota"], 1.0 - 2.0 / 3.0, delta=1e-6)

    def test_the_sweep_finds_the_threshold_that_drops_weak_false_positives(self) -> None:
        frames = [
            _frame([0.2], [1], [0.2, 9.0], [5, 7], [0.9, 0.6]),
            _frame([0.1], [1], [0.1, 9.0], [5, 7], [0.9, 0.6]),
        ]
        report = _report(frames, sweep_score_thresholds=(0.3, 0.7))

        # The weak far tracks count at the track threshold 0.5 and vanish from 0.7 on
        self.assertEqual(report["mota"], 0.0)
        self.assertEqual(report["mota_best"], 1.0)
        self.assertEqual(report["mota_best_score_threshold"], 0.7)

    def test_a_track_continues_only_the_pair_of_the_previous_frame(self) -> None:
        # A is matched to T, then gone while B takes T, then both are back within reach of T
        frames = [
            _frame([0.0], [1], [0.0], [7]),
            _frame([1.0], [2], [1.0], [7]),
            _frame([0.5, 1.5], [1, 2], [1.0], [7]),
        ]
        report = _report(frames[::-1])

        # T continues with B alone, A is a miss instead of a second pair of T
        self.assertEqual(report["num_gt"], 4.0)
        self.assertEqual(report["recall"], 0.75)
        self.assertEqual(report["precision"], 1.0)
        self.assertEqual(report["mota"], 0.75)
        self.assertEqual(report["id_switches"], 0.0)

    def test_a_pair_lost_in_the_previous_frame_does_not_carry_over(self) -> None:
        # A is matched to T, out of reach while B takes T, then both are in reach of T
        frames = [
            _frame([0.0], [1], [0.0], [7]),
            _frame([0.0, 3.0], [1, 2], [2.6], [7]),
            _frame([0.8, 2.2], [1, 2], [1.5], [7]),
        ]
        report = _report(frames[::-1])

        self.assertEqual(report["num_gt"], 5.0)
        self.assertEqual(report["recall"], 0.6)
        self.assertEqual(report["precision"], 1.0)
        self.assertAlmostEqual(report["mota"], 0.6, delta=1e-9)

    def test_a_track_on_an_unobserved_box_leaves_with_it(self) -> None:
        # The object at 30 m has not been observed in the stream yet, the model tracks it already
        frame = _frame([10.0, 30.0], [1, 2], [10.5, 30.5], [5, 6], [0.8, 0.9])
        frame["gt_num_points"] = torch.tensor([5, 0])
        report = _report([frame])

        self.assertEqual(report["num_gt"], 1.0)
        self.assertEqual(report["precision"], 1.0)
        self.assertEqual(report["mota"], 1.0)

    def test_a_box_beyond_the_class_distance_leaves_with_its_track(self) -> None:
        """
        Input: a car at 10 m and a car at 60 m with their tracks, cars evaluated up to 50 m.
        Expected: the far car and its track leave the evaluation, as in the detection suites.
        Check: one ground truth box and a clean MOTA.
        """
        frame = _frame([10.0, 60.0], [1, 2], [10.5, 60.5], [5, 6], [0.8, 0.9])
        report = _report([frame], max_distances={"car": 50.0})

        self.assertEqual(report["num_gt"], 1.0)
        self.assertEqual(report["precision"], 1.0)
        self.assertEqual(report["mota"], 1.0)

    def test_a_far_tracked_box_without_ground_truth_leaves(self) -> None:
        """
        Input: a car at 10 m with its track and a track at 140 m with no box, cars up to 50 m.
        Expected: the far track leaves the evaluation, the class distance bounds the predictions.
        Check: a clean precision and MOTA.
        """
        frame = _frame([10.0], [1], [10.5, 140.0], [5, 6])
        report = _report([frame], max_distances={"car": 50.0})

        self.assertEqual(report["precision"], 1.0)
        self.assertEqual(report["mota"], 1.0)

    def test_a_track_with_two_boxes_in_a_frame_is_refused(self) -> None:
        """
        Input: one track on two boxes in a frame, which a decoding of one box per query never gives.
        Expected: the suite refuses the frame.
        """
        frame = _frame([10.0], [1], [10.5, 10.6], [5, 5], [0.9, 0.6])

        with self.assertRaisesRegex(ValueError, "more than one box"):
            _report([frame])

    def test_frames_without_the_tracking_fields_are_refused(self) -> None:
        """
        Input: an eval output without detection frames, a frame without track ids on its
          predictions and a frame without the identities of its boxes.
        Expected: ValueError naming the missing part each time.
        """
        suite = TrackingMetricSuite(
            components=[TrackingSummary()],
            class_names=("car",),
            score_threshold=0.5,
            min_observed_points=1,
        )
        with self.assertRaisesRegex(ValueError, "needs the detection3d frames"):
            suite.update(EvalOutput())

        untracked = _frame([10.0], [1], [10.0], [5])
        untracked["prediction"]["instance_ids"] = None
        without_identities = {**_frame([10.0], [1], [10.0], [5]), "gt_instance_ids": None}
        for frame, message in (
            (untracked, "track id of every predicted box"),
            (without_identities, "lacks its object identities"),
        ):
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                _report([frame])

    def test_labels_outside_the_classes_are_refused(self) -> None:
        """
        Input: a ground truth box, then a tracked box, of the class after the last class.
        Expected: ValueError naming the labels, no box is left out of the counts silently.
        """
        gt_outside = _frame([10.0], [1], [10.0], [5])
        gt_outside["gt_labels"] = torch.tensor([1])
        tracked_outside = _frame([10.0], [1], [10.0], [5])
        tracked_outside["prediction"]["labels_3d"] = torch.tensor([1])
        for frame, name in ((gt_outside, "ground truth"), (tracked_outside, "tracked")):
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValueError, f"{name} labels outside the 1 classes"):
                    _report([frame])

    def test_tracking_without_observed_ground_truth_is_undefined(self) -> None:
        """
        Input: one frame with a tracked box and no ground truth box.
        Expected: MOTA and recall have no ground truth to divide by,
          the false positive stays counted.
        Check: NaN MOTA, recall and best MOTA, zero precision.
        """
        report = _report([_frame([], [], [10.0], [5])], sweep_score_thresholds=(0.3,))

        self.assertTrue(math.isnan(report["mota"]))
        self.assertTrue(math.isnan(report["recall"]))
        self.assertTrue(math.isnan(report["mota_best"]))
        self.assertEqual(report["precision"], 0.0)

    def test_a_component_with_a_filter_is_refused(self) -> None:
        """
        Input: a tracking summary configured with a corridor filter.
        Expected: ValueError, the suite reports no filtered slice and would label the whole scene.
        """
        with self.assertRaisesRegex(ValueError, "reports no filtered slices"):
            TrackingMetricSuite(
                components=[TrackingSummary(filter=CorridorFilter())],
                class_names=("car",),
                score_threshold=0.5,
                min_observed_points=1,
            )
