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

"""Tests for the grouped circle NMS shared by the detection heads."""

import unittest

import torch

from autoware_ml.models.detection3d.task_modules.nms_groups import (
    NMSGroupConfig,
    filter_nms_groups,
    resolve_nms_groups,
)

CLASS_NAMES = ["car", "bicycle", "cone", "barrier"]


class TestNMSGroups(unittest.TestCase):
    """Grouped NMS resolution and filtering."""

    def test_groups_suppress_across_their_classes_only(self) -> None:
        """
        Input: car and bicycle in one group, cones in a zero radius group capped at two, barrier
        in none.
        Expected: the bicycle next to the car is suppressed, the third cone is capped and the
        barrier is kept.
        Check: compare the keep mask.
        """
        centers = torch.tensor(
            [[[0.0, 0.0], [0.3, 0.0], [0.2, 0.0], [5.0, 0.0], [5.1, 0.0], [5.2, 0.0]]]
        )
        scores = torch.tensor([[0.9, 0.8, 0.7, 0.6, 0.5, 0.4]])
        labels = torch.tensor([[0, 1, 3, 2, 2, 2]])
        groups = resolve_nms_groups(
            [
                NMSGroupConfig(class_names=["car", "bicycle"], nms_radius=0.5, max_size=10),
                NMSGroupConfig(class_names=["cone"], nms_radius=0.0, max_size=2),
            ],
            CLASS_NAMES,
            default_radius=1.0,
            default_max_size=100,
        )

        keep = filter_nms_groups(
            centers, scores, labels, torch.ones_like(scores, dtype=torch.bool), groups, 4
        )

        self.assertEqual(keep[0].nonzero().flatten().tolist(), [0, 2, 3, 4])

    def test_resolve_rejects_a_class_in_two_groups(self) -> None:
        """
        Input: car named by two groups.
        Expected: a class weighed against two radii is a config mistake.
        Check: a ValueError naming both groups is raised.
        """
        with self.assertRaisesRegex(ValueError, "groups 0 and 1"):
            resolve_nms_groups(
                [
                    NMSGroupConfig(class_names=["car"], nms_radius=0.5, max_size=10),
                    NMSGroupConfig(class_names=["car", "cone"], nms_radius=0.5, max_size=10),
                ],
                CLASS_NAMES,
                default_radius=1.0,
                default_max_size=100,
            )


if __name__ == "__main__":
    unittest.main()
