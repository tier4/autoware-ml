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

"""Unit tests for the evaluated frame count."""

import unittest

import torch

from autoware_ml.metrics.base import EvalStage, MetricRange
from autoware_ml.metrics.detection3d.frame_count import FrameCount
from autoware_ml.metrics.detection3d.suite import Detection3DMetricSuite

_CLASS_NAMES = ("car", "pedestrian")


def _box(x: float) -> list[float]:
    """Box at distance ``x`` along the X axis, in the 9-value box layout."""
    return [x, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0]


def _prediction(xs: list[float]) -> dict[str, torch.Tensor]:
    """One frame of car predictions at the given distances."""
    return {
        "bboxes_3d": torch.tensor([_box(x) for x in xs], dtype=torch.float32).reshape(-1, 9),
        "scores_3d": torch.full((len(xs),), 0.9, dtype=torch.float32),
        "labels_3d": torch.zeros((len(xs),), dtype=torch.long),
    }


class TestFrameCount(unittest.TestCase):
    """Unit tests for FrameCount, driven through the detection suite."""

    def _build_suite(self) -> Detection3DMetricSuite:
        """Suite with one near bucket and one far bucket, the frame count as its only metric."""
        return Detection3DMetricSuite(
            components=[FrameCount()],
            class_names=_CLASS_NAMES,
            ranges=(MetricRange("0-50m", 0.0, 50.0), MetricRange("50-100m", 50.0, 100.0)),
        )

    def test_counts_every_accumulated_frame(self) -> None:
        """Test that frames are summed across batches, an empty frame included."""
        suite = self._build_suite()
        suite.update(
            {
                "predictions": [_prediction([10.0]), _prediction([60.0])],
                "gt_boxes": [
                    torch.tensor([_box(10.0)], dtype=torch.float32),
                    torch.tensor([_box(60.0)], dtype=torch.float32),
                ],
                "gt_labels": [torch.tensor([0]), torch.tensor([0])],
            }
        )
        # A frame with no boxes on either side is still an evaluated frame
        suite.update(
            {
                "predictions": [_prediction([])],
                "gt_boxes": [torch.zeros((0, 9), dtype=torch.float32)],
                "gt_labels": [torch.zeros((0,), dtype=torch.long)],
            }
        )

        report = suite.result(EvalStage.VAL)

        self.assertEqual(report["num_frames"], 3.0)

    def test_range_buckets_keep_the_frame_count(self) -> None:
        """Test that range clipping drops boxes, not frames, so every bucket counts all frames."""
        suite = self._build_suite()
        suite.update(
            {
                "predictions": [_prediction([10.0]), _prediction([60.0])],
                "gt_boxes": [
                    torch.tensor([_box(10.0)], dtype=torch.float32),
                    torch.tensor([_box(60.0)], dtype=torch.float32),
                ],
                "gt_labels": [torch.tensor([0]), torch.tensor([0])],
            }
        )

        report = suite.result(EvalStage.TEST)

        self.assertEqual(report["num_frames"], 2.0)
        self.assertEqual(report["num_frames_0m_50m"], 2.0)
        self.assertEqual(report["num_frames_50m_100m"], 2.0)

    def test_reset_drops_the_frames_of_the_previous_epoch(self) -> None:
        """Test that the count starts over after the suite is reset."""
        suite = self._build_suite()
        batch = {
            "predictions": [_prediction([10.0])],
            "gt_boxes": [torch.tensor([_box(10.0)], dtype=torch.float32)],
            "gt_labels": [torch.tensor([0])],
        }
        suite.update(batch)
        suite.update(batch)
        suite.reset()
        suite.update(batch)

        report = suite.result(EvalStage.VAL)

        self.assertEqual(report["num_frames"], 1.0)


if __name__ == "__main__":
    unittest.main()
