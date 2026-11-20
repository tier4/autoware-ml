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

"""Unit tests for scoring only the key frames of a stream."""

from __future__ import annotations

import unittest

import torch

from autoware_ml.metrics.base import EvalStage, MetricRange
from autoware_ml.metrics.detection3d.box_counts import BoxCounts
from autoware_ml.metrics.detection3d.frame_count import FrameCount as DetectionFrameCount
from autoware_ml.metrics.detection3d.suite import Detection3DMetricSuite
from autoware_ml.metrics.segmentation3d.error_clusters import ErrorClusters
from autoware_ml.metrics.segmentation3d.frame_count import FrameCount
from autoware_ml.metrics.segmentation3d.point_cloud import Segmentation3DPointCloudMetricSuite
from autoware_ml.metrics.eval_output import EvalOutput, Segmentation3DEvalFrame
from autoware_ml.metrics.report import MetricKey
from autoware_ml.metrics.tests.eval_output_fixtures import (
    detection_eval_output,
    detection_frame,
    prediction,
    segmentation_eval_output,
    segmentation_frame,
)


def _car(x: float) -> torch.Tensor:
    """One unit car box on the x axis."""
    box = torch.zeros((1, 9), dtype=torch.float32)
    box[0, 0] = x
    box[0, 3:6] = 1.0
    return box


def _eval_output(key_frames: list[bool], padding: list[bool]) -> EvalOutput:
    """Three frames with one car and one car prediction each."""
    return detection_eval_output(
        [
            detection_frame(
                prediction(_car(10.0 * (index + 1)), torch.tensor([0.9]), torch.tensor([0])),
                _car(10.0 * (index + 1)),
                torch.tensor([0]),
                is_padding=is_padding,
                key_frame=key_frame,
            )
            for index, (key_frame, is_padding) in enumerate(zip(key_frames, padding))
        ]
    )


def _seg_frame(
    pred: list[int], target: list[int], is_padding: bool = False, key_frame: bool = True
) -> Segmentation3DEvalFrame:
    """One segmentation frame with points on the x axis and confident scores."""
    num_points = len(pred)
    scores = torch.full((num_points, 2), 0.05)
    scores[torch.arange(num_points), torch.tensor(pred)] = 0.95
    coord = torch.zeros((num_points, 3))
    coord[:, 0] = torch.arange(num_points, dtype=torch.float32) * 0.1
    return segmentation_frame(
        torch.tensor(pred),
        torch.tensor(target),
        coord,
        scores,
        is_padding=is_padding,
        key_frame=key_frame,
    )


class TestKeyFrameScoring(unittest.TestCase):
    def test_the_unscored_frames_are_left_out(self) -> None:
        """
        Input: a key frame, then a frame between key frames and a padding slot, both unscored.
        Expected: only the key frame counts, the unscored frames neither in the boxes nor the frames.
        Check: one GT box, one prediction and one frame.
        """
        suite = Detection3DMetricSuite(
            components=[BoxCounts(), DetectionFrameCount()],
            class_names=("car", "pedestrian"),
            ranges=(MetricRange("0-100m", 0.0, 100.0),),
        )
        suite.update(_eval_output([True, False, True], [False, False, True]))
        report = suite.result(EvalStage.VAL)

        self.assertEqual(report.metric_value(MetricKey(name="num_gts_car")), 1.0)
        self.assertEqual(report.metric_value(MetricKey(name="num_preds_car")), 1.0)
        self.assertEqual(report.metric_value(MetricKey(name="num_frames")), 1.0)

    def test_the_point_cloud_suite_scores_the_key_frames_only(self) -> None:
        """
        Input: a key frame with one wrong point of four, its padding copy and an unannotated frame.
        Expected: only the key frame counts, as in the confusion suite.
        Check: one frame and an error rate of a quarter.
        """
        suite = Segmentation3DPointCloudMetricSuite(
            components=[FrameCount(stages=("test",)), ErrorClusters(stages=("test",))],
            num_classes=2,
            ignore_index=-1,
            class_names=("a", "b"),
            ranges=(),
        )
        suite.bind_stage(EvalStage.TEST)
        suite.update(
            segmentation_eval_output(
                [
                    _seg_frame([0, 0, 1, 1], [0, 0, 1, 0]),
                    _seg_frame([0, 0, 1, 1], [0, 0, 1, 0], is_padding=True),
                    _seg_frame([1, 1, 1, 1], [-1, -1, -1, -1], key_frame=False),
                ]
            )
        )
        report = suite.result(EvalStage.TEST)

        self.assertEqual(report.metric_value(MetricKey(name="num_frames")), 1.0)
        self.assertEqual(report.metric_value(MetricKey(name="error_rate")), 0.25)
