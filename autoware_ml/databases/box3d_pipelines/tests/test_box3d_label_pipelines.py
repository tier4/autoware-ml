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

"""Tests for the box pipeline that resolves the raw label names of the boxes."""

from __future__ import annotations

import unittest

import numpy as np

from autoware_ml.databases.box3d_pipelines.box3d_label_resolver import Box3DLabelResolver
from autoware_ml.databases.schemas.box3d_schemas import Box3DDataModel
from autoware_ml.databases.taxonomy import LabelVocabulary

VOCABULARY = LabelVocabulary(
    {
        "vehicle.car": "car",
        "vehicle.truck": "truck",
        "vehicle.trailer": "trailer",
        "unpainted": None,
    }
)


def _box(label_name: str) -> Box3DDataModel:
    return Box3DDataModel(
        box3d_params=np.zeros(10, dtype=np.float64),
        box3d_instance_id="instance",
        box3d_dataset_label_name=label_name,
        box3d_label_name=label_name,
        box3d_num_lidar_points=1,
        box3d_num_radar_points=0,
        box3d_valid=True,
        box3d_attributes=set(),
        box3d_coordinate="base_link",
    )


class TestBox3DLabelResolver(unittest.TestCase):
    def test_turns_raw_names_into_fine_names(self) -> None:
        boxes = Box3DLabelResolver(VOCABULARY)([_box("vehicle.car"), _box("vehicle.trailer")])

        self.assertEqual([box.box3d_label_name for box in boxes], ["car", "trailer"])
        self.assertEqual(
            [box.box3d_dataset_label_name for box in boxes], ["vehicle.car", "vehicle.trailer"]
        )

    def test_drops_a_raw_name_outside_every_level(self) -> None:
        self.assertEqual(Box3DLabelResolver(VOCABULARY)([_box("unpainted")]), [])

    def test_rejects_a_raw_name_the_vocabulary_does_not_list(self) -> None:
        with self.assertRaises(KeyError):
            Box3DLabelResolver(VOCABULARY)([_box("vehicle.bus")])


if __name__ == "__main__":
    unittest.main()
