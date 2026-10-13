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

"""Unit tests for the ground truth labels the detection suite accepts."""

import unittest

import torch

from autoware_ml.metrics.detection3d.box_counts import BoxCounts
from autoware_ml.metrics.detection3d.suite import Detection3DMetricSuite


class TestSuiteGroundTruthLabels(unittest.TestCase):
    """A box of an ignored class never reaches the metrics."""

    def test_rejects_ground_truth_of_an_ignored_class(self) -> None:
        """
        Input: one frame whose ground truth holds a box labelled -1, the ignore index.
        Expected: the suite refuses it, since scoring it would count a box of no class.
        Check: ValueError naming the classes.
        """
        suite = Detection3DMetricSuite(
            components=[BoxCounts()], class_names=("car", "pedestrian"), min_num_points=0
        )
        boxes = torch.zeros((2, 9), dtype=torch.float32)
        boxes[:, 3:6] = 1.0

        with self.assertRaisesRegex(ValueError, "outside the 2 classes"):
            suite.update(
                {
                    "predictions": [
                        {
                            "bboxes_3d": torch.zeros((0, 9)),
                            "scores_3d": torch.zeros(0),
                            "labels_3d": torch.zeros(0, dtype=torch.long),
                        }
                    ],
                    "gt_boxes": [boxes],
                    "gt_labels": [torch.tensor([0, -1])],
                }
            )


if __name__ == "__main__":
    unittest.main()
