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

"""Unit tests for what the stream of a temporal model has observed, as the metrics count it."""

import unittest

import torch

from autoware_ml.dataclasses.batch.frame_meta import FrameMetaSample
from autoware_ml.dataclasses.models.detection3d.predictions import Detection3DSamplePredictions
from autoware_ml.metrics.base import EvalStage, MetricRange
from autoware_ml.metrics.detection3d.box_counts import BoxCounts
from autoware_ml.metrics.detection3d.mean_ap import MeanAP
from autoware_ml.metrics.detection3d.observation import StreamObserver, observe_detection_frame
from autoware_ml.metrics.detection3d.suite import Detection3DMetricSuite
from autoware_ml.metrics.eval_output import Detection3DEvalFrame, Detection3DEvalOutput, EvalOutput
from autoware_ml.metrics.report import MetricKey


def _boxes(*xs: float) -> torch.Tensor:
    """Unit boxes on the x axis."""
    boxes = torch.zeros((len(xs), 9), dtype=torch.float32)
    boxes[:, 0] = torch.tensor(xs)
    boxes[:, 3:6] = 1.0
    return boxes


def _frame(
    scene: str,
    gt_xs: tuple[float, ...],
    instance_ids: list[int],
    num_points: list[int],
    pred_xs: tuple[float, ...] = (),
) -> Detection3DEvalFrame:
    """One key frame of cars of a scene."""
    return Detection3DEvalFrame(
        predictions=Detection3DSamplePredictions(
            bboxes_3d=_boxes(*pred_xs),
            scores_3d=torch.linspace(0.9, 0.5, len(pred_xs)),
            labels_3d=torch.zeros(len(pred_xs), dtype=torch.long),
        ),
        gt_boxes=_boxes(*gt_xs),
        gt_labels=torch.zeros(len(gt_xs), dtype=torch.long),
        gt_num_points=torch.tensor(num_points, dtype=torch.long),
        gt_instance_ids=torch.tensor(instance_ids, dtype=torch.long),
        key_frame=True,
        frame_meta=FrameMetaSample(
            ego2global=torch.eye(4, dtype=torch.float64),
            scene_token=scene,
            timestamp=0.0,
            key_frame=True,
        ),
    )


class TestStreamObserver(unittest.TestCase):
    def test_an_object_counts_from_its_first_frame_with_enough_points(self) -> None:
        """
        Input: one lane over three frames of a scene, the object seen with 1, 3 and 0 points.
        Expected: unobserved below 2 points, observed from the frame with 3 on, also when its
          points vanish again.
        """
        observer = StreamObserver(min_observed_points=2)

        observed = [
            observe_detection_frame(observer, 0, _frame("a", (10.0,), [7], [points])).tolist()
            for points in (1, 3, 0)
        ]

        self.assertEqual(observed, [[False], [True], [True]])

    def test_a_lane_forgets_its_objects_with_a_new_scene(self) -> None:
        """
        Input: a lane observing object 7, then a frame of another scene with object 7 occluded,
          and a second lane of the first scene.
        Expected: the new scene starts unobserved, every lane follows its own stream.
        """
        observer = StreamObserver(min_observed_points=1)
        observe_detection_frame(observer, 0, _frame("a", (10.0,), [7], [5]))

        self.assertEqual(
            observe_detection_frame(observer, 0, _frame("b", (10.0,), [7], [0])).tolist(), [False]
        )
        self.assertEqual(
            observe_detection_frame(observer, 1, _frame("a", (10.0,), [7], [0])).tolist(), [False]
        )

    def test_a_frame_without_identities_is_refused(self) -> None:
        frame = _frame("a", (10.0,), [7], [5]).model_copy(update={"gt_instance_ids": None})

        with self.assertRaisesRegex(ValueError, "instance ids and the point counts"):
            observe_detection_frame(StreamObserver(min_observed_points=1), 0, frame)

    def test_a_negative_threshold_is_refused(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-negative"):
            StreamObserver(min_observed_points=-1)


class TestObservedDetectionSuite(unittest.TestCase):
    def test_the_detection_of_an_unobserved_box_counts_neither_way(self) -> None:
        """
        Input: a seen car at 10 m and a car at 30 m without points its stream never observed,
          both detected, the second detection scored higher.
        Expected: the unobserved car is ignored with the detection that claims it.
        Check: one GT box and an AP of 1, then a fresh epoch forgets the stream.
        """
        suite = Detection3DMetricSuite(
            components=[BoxCounts(), MeanAP(stages=("val",), thresholds=(2.0,))],
            class_names=("car",),
            min_observed_points=1,
            ranges=(MetricRange("0-100m", 0.0, 100.0),),
        )
        frame = _frame("a", (10.0, 30.0), [1, 2], [5, 0], pred_xs=(30.5, 10.5))
        suite.update(EvalOutput(detection3d=Detection3DEvalOutput(frames=[frame])))

        report = suite.result(EvalStage.VAL)

        self.assertEqual(report.metric_value(MetricKey(name="num_gts_car")), 1.0)
        self.assertAlmostEqual(report.metric_value(MetricKey(name="mAP_car")), 1.0)
        self.assertTrue(suite.requirements().gt_instance_ids)


if __name__ == "__main__":
    unittest.main()
