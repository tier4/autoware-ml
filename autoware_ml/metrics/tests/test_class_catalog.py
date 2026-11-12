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

"""Tests for the tables the metrics read from a class catalog."""

from __future__ import annotations

import unittest

from autoware_ml.metrics import class_catalog
from autoware_ml.types.metrics import MotionModel

DETECTION3D_CATALOG = {
    "car": {"group": "grouped_vehicle", "max_distance": 121.0, "motion_model": "wheeled"},
    "bicycle": {
        "group": "grouped_two_wheeler",
        "max_distance": 121.0,
        "motion_model": "omnidirectional",
        "max_speed": 6.0,
        "partial_detection": True,
    },
    "barrier": {"group": "grouped_obstacle", "max_distance": 121.0, "motion_model": "static"},
    "truck": {"group": "grouped_vehicle", "max_distance": 90.0, "motion_model": "wheeled"},
}
SEGMENTATION3D_CATALOG = {
    "car": {"group": "grouped_vehicle"},
    "vegetation": {"group": "grouped_static_structure"},
    "truck": {"group": "grouped_vehicle"},
}


class TestDetection3DTables(unittest.TestCase):
    def test_tables_cover_the_trained_classes_only(self) -> None:
        class_names = ["truck", "car", "bicycle"]

        self.assertEqual(
            class_catalog.max_distances(DETECTION3D_CATALOG, class_names),
            {"truck": 90.0, "car": 121.0, "bicycle": 121.0},
        )
        self.assertEqual(
            class_catalog.motion_models(DETECTION3D_CATALOG, class_names),
            {
                "truck": MotionModel.WHEELED,
                "car": MotionModel.WHEELED,
                "bicycle": MotionModel.OMNIDIRECTIONAL,
            },
        )
        self.assertEqual(
            class_catalog.max_speeds(DETECTION3D_CATALOG, class_names), {"bicycle": 6.0}
        )
        self.assertEqual(
            class_catalog.partial_detection_classes(DETECTION3D_CATALOG, class_names), ["bicycle"]
        )

    def test_groups_follow_the_catalog_order(self) -> None:
        groups = class_catalog.detection3d_class_groups(
            DETECTION3D_CATALOG, ["bicycle", "truck", "car"]
        )

        self.assertEqual(
            groups, {"grouped_vehicle": ["car", "truck"], "grouped_two_wheeler": ["bicycle"]}
        )
        self.assertEqual(list(groups), ["grouped_vehicle", "grouped_two_wheeler"])

    def test_a_trained_class_the_catalog_does_not_list_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "no row for the trained classes \\['bus'\\]"):
            class_catalog.max_distances(DETECTION3D_CATALOG, ["car", "bus"])

    def test_rejects_inconsistent_rows(self) -> None:
        cases = (
            ({"group": "g", "max_distance": 0.0, "motion_model": "static"}, "max_distance"),
            ({"group": "g", "max_distance": 1.0, "motion_model": "omnidirectional"}, "max_speed"),
            (
                {"group": "g", "max_distance": 1.0, "motion_model": "static", "max_speed": 1.0},
                "max_speed",
            ),
            ({"group": "g", "max_distance": 1.0, "motion_model": "flying"}, "flying"),
        )
        for row, message in cases:
            with self.subTest(row=row):
                with self.assertRaisesRegex(ValueError, message):
                    class_catalog.max_distances({"car": row}, ["car"])

    def test_rejects_an_unknown_setting(self) -> None:
        row = {"group": "g", "max_distance": 1.0, "motion_model": "static", "colour": "red"}

        with self.assertRaises(TypeError):
            class_catalog.max_distances({"car": row}, ["car"])


class TestSegmentation3DTables(unittest.TestCase):
    def test_groups_follow_the_catalog_order(self) -> None:
        groups = class_catalog.segmentation3d_class_groups(
            SEGMENTATION3D_CATALOG, ["vegetation", "truck", "car"]
        )

        self.assertEqual(
            groups,
            {"grouped_vehicle": ["car", "truck"], "grouped_static_structure": ["vegetation"]},
        )
        self.assertEqual(list(groups), ["grouped_vehicle", "grouped_static_structure"])


if __name__ == "__main__":
    unittest.main()
