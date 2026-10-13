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

"""Unit tests for the ground-truth and prediction box counts."""

import unittest

import torch

from autoware_ml.metrics.base import EvalStage, MetricRange
from autoware_ml.metrics.detection3d.box_counts import BoxCounts
from autoware_ml.metrics.detection3d.suite import Detection3DMetricSuite

_CLASS_NAMES = ("car", "pedestrian")


def _box(x: float) -> list[float]:
    """Box at distance ``x`` along the X axis, in the 9-value box layout."""
    return [x, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0]


class TestBoxCounts(unittest.TestCase):
    """Unit tests for BoxCounts, driven through the detection suite."""

    def _build_suite(self, min_num_points: int = 2) -> Detection3DMetricSuite:
        """Suite with one near bucket and one far bucket, counts as its only metric."""
        return Detection3DMetricSuite(
            components=[BoxCounts()],
            class_names=_CLASS_NAMES,
            min_num_points=min_num_points,
            ranges=(MetricRange("0-50m", 0.0, 50.0), MetricRange("50-100m", 50.0, 100.0)),
        )

    def _update(self, suite: Detection3DMetricSuite) -> None:
        """Accumulate one frame: 3 GT (one below min_num_points) and 3 predictions."""
        suite.update(
            {
                "predictions": [
                    {
                        "bboxes_3d": torch.tensor(
                            [_box(10.0), _box(20.0), _box(60.0)], dtype=torch.float32
                        ),
                        "scores_3d": torch.tensor([0.9, 0.8, 0.7], dtype=torch.float32),
                        "labels_3d": torch.tensor([0, 1, 0], dtype=torch.long),
                    }
                ],
                "gt_boxes": [
                    torch.tensor([_box(10.0), _box(30.0), _box(70.0)], dtype=torch.float32)
                ],
                "gt_labels": [torch.tensor([0, 1, 0], dtype=torch.long)],
                # The pedestrian at 30m has too few points and must not be counted
                "gt_num_points": [torch.tensor([5, 1, 4], dtype=torch.long)],
            }
        )

    def test_counts_exclude_gt_below_min_num_points(self) -> None:
        """Test that GT dropped by the min_num_points filter is not counted."""
        suite = self._build_suite()
        self._update(suite)

        report = suite.result(EvalStage.VAL)

        self.assertEqual(report["total_num_gts"], 2.0)
        self.assertEqual(report["num_gts_car"], 2.0)
        self.assertEqual(report["num_gts_pedestrian"], 0.0)
        self.assertEqual(report["total_num_preds"], 3.0)
        self.assertEqual(report["num_preds_car"], 2.0)
        self.assertEqual(report["num_preds_pedestrian"], 1.0)

    def test_counts_are_reported_per_range(self) -> None:
        """Test that each range bucket counts only the boxes clipped into it."""
        suite = self._build_suite()
        self._update(suite)

        report = suite.result(EvalStage.VAL)

        # Near bucket: the car GT at 10m, the car and pedestrian predictions at 10m and 20m
        self.assertEqual(report["num_gts_car_0m_50m"], 1.0)
        self.assertEqual(report["num_preds_car_0m_50m"], 1.0)
        self.assertEqual(report["num_preds_pedestrian_0m_50m"], 1.0)

        # Far bucket: the car GT at 70m and the car prediction at 60m
        self.assertEqual(report["num_gts_car_50m_100m"], 1.0)
        self.assertEqual(report["num_preds_car_50m_100m"], 1.0)
        self.assertEqual(report["num_preds_pedestrian_50m_100m"], 0.0)

    def test_counts_keep_all_gt_without_a_point_filter(self) -> None:
        """Test that the sparse GT box is counted once min_num_points is disabled."""
        suite = self._build_suite(min_num_points=0)
        self._update(suite)

        report = suite.result(EvalStage.VAL)

        self.assertEqual(report["total_num_gts"], 3.0)
        self.assertEqual(report["num_gts_pedestrian"], 1.0)


if __name__ == "__main__":
    unittest.main()
