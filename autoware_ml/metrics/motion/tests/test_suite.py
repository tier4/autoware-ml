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

"""Tests for the open-loop motion metrics."""

from __future__ import annotations

import unittest

import torch

from autoware_ml.metrics.base import EvalStage
from autoware_ml.metrics.eval_output import EvalOutput
from autoware_ml.metrics.filters import CorridorFilter
from autoware_ml.metrics.motion.eval_output import MotionEvalFrame
from autoware_ml.metrics.motion.tests.eval_output_fixtures import motion_output
from autoware_ml.metrics.motion.suite import (
    AgentForecastSummary,
    AgentMotionSummary,
    MotionMetricSuite,
)
from autoware_ml.metrics.report import MetricKey


def frame(prediction_offset: float) -> dict[str, torch.Tensor]:
    """One car moving 1 m per step, predicted with an offset."""
    future = torch.tensor([[[11.0, 0.0], [12.0, 0.0]]])
    return {
        "pred_xy": torch.tensor([[10.0, 0.0]]),
        "pred_labels": torch.tensor([0]),
        "pred_scores": torch.tensor([0.9]),
        "pred_trajectories": (future + torch.tensor([0.0, prediction_offset]))[:, None].repeat(
            1, 2, 1, 1
        ),
        "pred_mode_scores": torch.full((1, 2), 0.5),
        "pred_velocities": torch.tensor([[2.0, 0.0]]),
        "gt_xy": torch.tensor([[10.0, 0.0]]),
        "gt_labels": torch.zeros(1, dtype=torch.long),
        "gt_futures": future,
        "gt_valid": torch.ones(1, 2, dtype=torch.bool),
    }


def evaluate(frames: list[dict[str, torch.Tensor]]) -> dict[str, float]:
    """Run the agent metric of a suite over the frames."""
    suite = MotionMetricSuite(
        components=[AgentMotionSummary()], class_names=("car",), score_threshold=0.2
    )
    observed = [torch.ones(len(frame["gt_xy"]), dtype=torch.bool) for frame in frames]
    suite.update(motion_output(frames, observed))
    state = suite.state_for(None, suite.components[0].filter)
    report = suite.components[0].evaluate(state, EvalStage.VAL)
    return {str(key): report.metric_value(key) for key in report.metric_names()}


class TestMotionSuite(unittest.TestCase):
    """The open-loop motion metrics."""

    def test_a_perfect_prediction_scores_zero_errors(self) -> None:
        """
        Input: the car's future predicted exactly.
        Expected: zero minADE and minFDE, no miss, EPA 1.
        """
        report = evaluate([frame(0.0)])
        self.assertEqual(report["minADE_car"], 0.0)
        self.assertEqual(report["minFDE_car"], 0.0)
        self.assertEqual(report["MR_car"], 0.0)
        self.assertEqual(report["EPA_car"], 1.0)

    def test_offsets_become_displacement_errors_and_misses(self) -> None:
        """
        Input: the car predicted 3 m to the side.
        Expected: minADE and minFDE 3, a miss, EPA 0.
        """
        report = evaluate([frame(3.0)])
        self.assertEqual(report["minADE_car"], 3.0)
        self.assertEqual(report["minFDE_car"], 3.0)
        self.assertEqual(report["MR_car"], 1.0)
        self.assertEqual(report["EPA_car"], 0.0)

    def test_a_prediction_on_a_car_without_a_known_future_is_no_false_positive(self) -> None:
        """
        Input: three cars detected exactly, the first two with a known future predicted exactly, the
          third at the end of its scene without one.
        Expected: no false positive, two hits over the two boxes known at the horizon end, the third
          box neither a hit nor counted, zero displacement over the cars with a future.
        """
        known = frame(0.0)
        unknown = frame(0.0)
        unknown["pred_xy"] = unknown["gt_xy"] = torch.tensor([[30.0, 0.0]])
        unknown["gt_futures"] = torch.zeros(1, 2, 2)
        unknown["gt_valid"] = torch.zeros(1, 2, dtype=torch.bool)
        other = frame(0.0)
        other["pred_xy"] = other["gt_xy"] = torch.tensor([[-10.0, 0.0]])
        merged = {name: torch.cat([known[name], other[name], unknown[name]]) for name in known}

        report = evaluate([merged])

        self.assertEqual(report["EPA_car"], 1.0)
        self.assertEqual(report["minADE_car"], 0.0)
        self.assertEqual(report["minFDE_car"], 0.0)
        self.assertEqual(report["MR_car"], 0.0)

    def test_a_future_that_ends_before_the_horizon_scores_only_its_average_displacement(
        self,
    ) -> None:
        """
        Input: a car known to the end of the horizon predicted exactly, a second car known only at
          the first step and predicted 1.9 m to the side.
        Expected: minADE averages both cars, minFDE, MR and EPA count only the first car, so a
          perfect forecast of it scores EPA 1.
        """
        full = frame(0.0)
        truncated = frame(1.9)
        truncated["pred_xy"] = truncated["gt_xy"] = torch.tensor([[-10.0, 0.0]])
        truncated["pred_trajectories"] = truncated["pred_trajectories"] - torch.tensor([20.0, 0.0])
        truncated["gt_futures"] = truncated["gt_futures"] - torch.tensor([20.0, 0.0])
        truncated["gt_valid"] = torch.tensor([[True, False]])
        merged = {name: torch.cat([full[name], truncated[name]]) for name in full}

        report = evaluate([merged])

        self.assertAlmostEqual(report["minADE_car"], 0.95, places=6)
        self.assertEqual(report["minFDE_car"], 0.0)
        self.assertEqual(report["MR_car"], 0.0)
        self.assertEqual(report["EPA_car"], 1.0)

    def test_an_endpoint_at_the_miss_distance_is_no_miss_in_either_summary(self) -> None:
        """
        Input: the car predicted 2 m to the side, the miss distance being 2 m.
        Expected: a miss is an endpoint farther than the miss distance, as in Argoverse 2, so both
          summaries count a hit and no miss.
        """
        report = evaluate([frame(2.0)])
        forecast_report = forecast([frame(2.0)])
        self.assertEqual(report["MR_car"], 0.0)
        self.assertEqual(report["EPA_car"], 1.0)
        self.assertEqual(forecast_report["MR2"], 0.0)
        self.assertEqual(forecast_report["MR1"], 0.0)

    @unittest.skipUnless(torch.cuda.is_available(), "needs a second device")
    def test_unscored_flags_follow_the_device_of_the_frames(self) -> None:
        """
        Input: one frame whose tensors live on the GPU.
        Expected: its unscored flag lives there too, so ranks can gather it with the frame fields.
        """
        suite = MotionMetricSuite(
            components=[AgentMotionSummary()], class_names=("car",), score_threshold=0.2
        )
        sample = {name: value.cuda() for name, value in frame(0.0).items()}
        suite.update(motion_output([sample], [torch.ones(1, dtype=torch.bool).cuda()]))
        self.assertEqual(suite.unscored[0].device, sample["pred_xy"].device)

    def test_only_the_confident_predictions_are_kept(self) -> None:
        """
        Input: a scored frame with a prediction above and one below the score threshold, and an
          unscored copy of it.
        Expected: the scored frame keeps the confident prediction only, the metrics read no other,
          and the unscored frame keeps empty entries.
        """
        suite = MotionMetricSuite(
            components=[AgentMotionSummary()], class_names=("car",), score_threshold=0.2
        )
        sample = frame(0.0)
        weak = frame(0.0)
        weak["pred_scores"] = torch.tensor([0.1])
        merged = {name: torch.cat([sample[name], weak[name]]) for name in sample}

        observed = [torch.ones(2, dtype=torch.bool)] * 2
        suite.update(motion_output([merged, merged], observed, [True, False]))

        self.assertEqual([len(entry) for entry in suite.pred_trajectories], [1, 0])
        self.assertAlmostEqual(suite.pred_scores[0].item(), 0.9, places=6)
        self.assertEqual([len(entry) for entry in suite.gt_xy], [2, 0])

    def test_boxes_the_tracking_suite_leaves_out_leave_with_their_predictions(self) -> None:
        """
        Input: a car predicted exactly, a second one at 30 m not observed in its stream yet and a
          third one at 60 m, each with its prediction, cars evaluated up to 50 m.
        Expected: the unobserved and the far car leave with their predictions, as in the tracking
          suite, so EPA scores the first car alone.
        """
        suite = MotionMetricSuite(
            components=[AgentMotionSummary()],
            class_names=("car",),
            score_threshold=0.2,
            max_distances={"car": 50.0},
        )
        frames = [frame(0.0), frame(0.0), frame(0.0)]
        for moved, x in zip(frames[1:], (30.0, 60.0), strict=True):
            moved["pred_xy"] = moved["gt_xy"] = torch.tensor([[x, 0.0]])
        merged = {name: torch.cat([item[name] for item in frames]) for name in frames[0]}

        suite.update(motion_output([merged], [torch.tensor([True, False, True])]))
        report = suite.components[0].evaluate(
            suite.state_for(None, suite.components[0].filter), EvalStage.VAL
        )

        self.assertEqual(len(suite.gt_xy[0]), 1)
        self.assertEqual(len(suite.pred_xy[0]), 1)
        self.assertEqual(report.metric_value(MetricKey(name="EPA_car")), 1.0)

    def test_a_component_with_a_filter_is_refused(self) -> None:
        """
        Input: an agent motion summary configured with a corridor filter.
        Expected: ValueError, the suite reports no filtered slice and would label the whole scene.
        """
        with self.assertRaisesRegex(ValueError, "reports no filtered slices"):
            MotionMetricSuite(
                components=[AgentMotionSummary(filter=CorridorFilter())],
                class_names=("car",),
                score_threshold=0.2,
            )

    def test_an_eval_output_without_motion_frames_is_refused(self) -> None:
        """
        Input: an eval output without motion frames.
        Expected: ValueError naming the missing frames.
        """
        suite = MotionMetricSuite(
            components=[AgentMotionSummary()], class_names=("car",), score_threshold=0.2
        )

        with self.assertRaisesRegex(ValueError, "needs the motion frames"):
            suite.update(EvalOutput())

    def test_a_prediction_column_of_another_agent_count_is_refused(self) -> None:
        """
        Input: a motion frame of one predicted agent with the labels of two.
        Expected: ValueError naming the column and both counts.
        """
        sample = frame(0.0)
        sample["pred_labels"] = torch.tensor([0, 0])

        with self.assertRaisesRegex(ValueError, "pred_labels describes 2 agents, pred_xy 1"):
            MotionEvalFrame(**sample, gt_observed=torch.ones(1, dtype=torch.bool), scored=True)


def forecast(frames: list[dict[str, torch.Tensor]]) -> dict[str, float]:
    """Run the forecast summary of a suite over the frames, 0.5 s steps."""
    summary = AgentForecastSummary(step_seconds=0.5)
    suite = MotionMetricSuite(components=[summary], class_names=("car",), score_threshold=0.2)
    observed = [torch.ones(len(frame["gt_xy"]), dtype=torch.bool) for frame in frames]
    suite.update(motion_output(frames, observed))
    report = summary.evaluate(suite.state_for(None, summary.filter), EvalStage.TEST)
    return {str(key): report.metric_value(key) for key in report.metric_names()}


def two_mode_frame() -> dict[str, torch.Tensor]:
    """The car of :func:`frame` with a likely mode 3 m to the side and an unlikely mode 1 m off
    at the endpoint only, at 1 m per step while its detection estimates 1 m/s."""
    sample = frame(0.0)
    future = sample["gt_futures"]
    likely = future + torch.tensor([0.0, 3.0])
    unlikely = future + torch.tensor([[[0.0, 0.0], [0.0, 1.0]]])
    sample["pred_trajectories"] = torch.stack([likely, unlikely], dim=1)
    sample["pred_mode_scores"] = torch.tensor([[0.8, 0.2]])
    sample["pred_velocities"] = torch.tensor([[1.0, 0.0]])
    return sample


class TestForecastSummary(unittest.TestCase):
    """The motion forecasting metrics of the Argoverse 2 benchmark."""

    def test_a_perfect_forecast_scores_zero_errors(self) -> None:
        """
        Input: the car's future predicted exactly by both modes, the detection at 2 m/s.
        Expected: zero errors and no miss for the best and the likely mode, a perfect baseline as
          the car covers 1 m per 0.5 s step, one agent.
        """
        report = forecast([frame(0.0)])
        for key in ("minADE2", "minFDE2", "MR2", "minADE1", "minFDE1", "MR1", "cvADE", "cvFDE"):
            self.assertEqual(report[key], 0.0, key)
        self.assertEqual(report["agents"], 1.0)
        self.assertEqual(report["agents_car"], 1.0)

    def test_the_best_mode_is_the_closest_endpoint_and_the_top_mode_the_likeliest(self) -> None:
        """
        Input: a likely mode 3 m to the side and an unlikely one exact at 0.5 s and 1 m off at
          1.0 s, the detection at 1 m/s while the car covers 2 m/s.
        Expected: the best mode is the unlikely one (ADE 0.5, FDE 1, no miss), the top mode the
          likely one (ADE 3, FDE 3, a miss), the baseline lags 0.5 and 1 m (ADE 0.75, FDE 1),
          the advantage is 0.75 over 0.5.
        """
        report = forecast([two_mode_frame()])
        self.assertAlmostEqual(report["minADE2"], 0.5, places=6)
        self.assertAlmostEqual(report["minFDE2"], 1.0, places=6)
        self.assertEqual(report["MR2"], 0.0)
        self.assertAlmostEqual(report["minADE1"], 3.0, places=6)
        self.assertAlmostEqual(report["minFDE1"], 3.0, places=6)
        self.assertEqual(report["MR1"], 1.0)
        self.assertAlmostEqual(report["cvADE"], 0.75, places=6)
        self.assertAlmostEqual(report["cvFDE"], 1.0, places=6)
        self.assertAlmostEqual(report["advantage"], 1.5, places=6)

    def test_only_agents_known_at_the_last_step_count(self) -> None:
        """
        Input: the perfect car and a car 3 m off whose last step is unknown.
        Expected: only the perfect car counts, as the benchmark scores full futures.
        """
        partial = frame(3.0)
        partial["pred_xy"] = partial["gt_xy"] = torch.tensor([[-10.0, 0.0]])
        partial["gt_valid"] = torch.tensor([[True, False]])
        merged = {name: torch.cat([frame(0.0)[name], partial[name]]) for name in partial}

        report = forecast([merged])

        self.assertEqual(report["agents"], 1.0)
        self.assertEqual(report["minFDE2"], 0.0)
        self.assertEqual(report["MR2"], 0.0)

    def test_a_non_positive_step_is_rejected(self) -> None:
        """
        Input: a forecast summary with 0 s steps.
        Expected: ValueError, the baseline could not be placed.
        """
        with self.assertRaisesRegex(ValueError, "step_seconds must be positive"):
            AgentForecastSummary(step_seconds=0.0)

    def test_mode_scores_of_another_mode_count_are_rejected(self) -> None:
        """
        Input: a motion frame with two trajectory modes and three mode scores.
        Expected: ValueError naming both counts.
        """
        sample = frame(0.0)
        sample["pred_mode_scores"] = torch.full((1, 3), 1.0 / 3.0)
        with self.assertRaisesRegex(ValueError, "scores 3 modes, pred_trajectories holds 2"):
            MotionEvalFrame(**sample, gt_observed=torch.ones(1, dtype=torch.bool), scored=True)
