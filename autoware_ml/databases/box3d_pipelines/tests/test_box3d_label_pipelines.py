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

"""Tests for the box pipelines that resolve and fold the label names of the boxes."""

from __future__ import annotations

import numpy as np
import pytest

from autoware_ml.databases.box3d_pipelines.box3d_label_remapper import Box3DLabelRemapper
from autoware_ml.databases.box3d_pipelines.box3d_label_resolver import Box3DLabelResolver
from autoware_ml.databases.schemas.box3d_schemas import Box3DDataModel
from autoware_ml.databases.taxonomy import LabelTaxonomy, LabelVocabulary

VOCABULARY = LabelVocabulary(
    {
        "vehicle.car": "car",
        "vehicle.truck": "truck",
        "vehicle.trailer": "trailer",
        "unpainted": None,
    }
)
TAXONOMY = LabelTaxonomy(
    VOCABULARY,
    ["car", "truck"],
    {"car": "car", "truck": "truck", "trailer": None},
    -1,
    {"grouped_vehicle": ["car", "truck"]},
)


def _box(label_name: str) -> Box3DDataModel:
    return Box3DDataModel(
        box3d_params=np.zeros(10, dtype=np.float64),
        box3d_instance_id="instance",
        box3d_dataset_label_name=label_name,
        box3d_label_name=label_name,
        box3d_label_index=-1,
        box3d_num_lidar_points=1,
        box3d_num_radar_points=0,
        box3d_valid=True,
        box3d_attributes=set(),
        box3d_coordinate="base_link",
    )


def test_resolver_turns_raw_names_into_fine_names_and_indices() -> None:
    boxes = Box3DLabelResolver(TAXONOMY)([_box("vehicle.car"), _box("vehicle.trailer")])

    assert [box.box3d_label_name for box in boxes] == ["car", "trailer"]
    # The trailer is a fine name the level drops, so it takes the ignore index
    assert [box.box3d_label_index for box in boxes] == [0, -1]


def test_resolver_drops_a_raw_name_outside_every_level() -> None:
    assert Box3DLabelResolver(TAXONOMY)([_box("unpainted")]) == []


def test_resolver_rejects_a_raw_name_the_vocabulary_does_not_list() -> None:
    with pytest.raises(KeyError):
        Box3DLabelResolver(TAXONOMY)([_box("vehicle.bus")])


def test_remapper_folds_the_listed_names_and_keeps_the_others() -> None:
    boxes = Box3DLabelRemapper(TAXONOMY, {"trailer": "truck"})([_box("trailer"), _box("car")])

    assert [box.box3d_label_name for box in boxes] == ["truck", "car"]
    assert [box.box3d_label_index for box in boxes] == [1, 0]
