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

"""Tests for the label sets of a taxonomy level and the corpus category aliases."""

from __future__ import annotations

import pytest

from autoware_ml.databases.scenarios import DatasetParams
from autoware_ml.databases.t4dataset.t4scenarios import T4Scenarios
from autoware_ml.databases.taxonomy import LabelTaxonomy, LabelVocabulary

VOCABULARY = LabelVocabulary(
    {
        "pole": "pole",
        "traffic_sign": "traffic_sign",
        "vertical_thin": "vertical_thin_unspecified",
        "manmade": "legacy_manmade",
        "building": "building",
        "ignored": None,
    }
)
CLASSES = ["building", "vertical_thin", "traffic_sign"]
GROUPS = {"structure": ["building", "vertical_thin", "traffic_sign"]}


def _taxonomy(**overrides: object) -> LabelTaxonomy:
    kwargs: dict = {
        "vocabulary": VOCABULARY,
        "class_names": CLASSES,
        "class_mapping": {
            "pole": "vertical_thin",
            "traffic_sign": "traffic_sign",
            "vertical_thin_unspecified": "thin_or_sign",
            "legacy_manmade": "manmade_or_thin",
            "building": "building",
        },
        "ignore_index": -1,
        "class_groups": GROUPS,
        "class_sets": {
            "thin_or_sign": ["vertical_thin", "traffic_sign"],
            "manmade_or_thin": ["building", "vertical_thin", "traffic_sign"],
        },
    }
    kwargs.update(overrides)
    return LabelTaxonomy(**kwargs)


def test_set_fine_names_resolve_past_the_classes() -> None:
    taxonomy = _taxonomy()
    assert taxonomy.num_classes == 3
    assert tuple(taxonomy.class_sets) == ("thin_or_sign", "manmade_or_thin")
    assert taxonomy.resolve_index("pole") == 1
    assert taxonomy.resolve_index("traffic_sign") == 2
    assert taxonomy.resolve_index("vertical_thin") == 3
    assert taxonomy.resolve_index("manmade") == 4
    assert taxonomy.resolve_index("ignored") == -1
    assert taxonomy.class_name("vertical_thin_unspecified") == "thin_or_sign"


def test_sets_are_part_of_the_definition_string() -> None:
    with_sets = _taxonomy()
    without = _taxonomy(
        class_mapping={
            "pole": "vertical_thin",
            "traffic_sign": "traffic_sign",
            "vertical_thin_unspecified": "vertical_thin",
            "legacy_manmade": "building",
            "building": "building",
        },
        class_sets=None,
    )
    assert "thin_or_sign" in str(with_sets)
    assert with_sets != without
    assert dict(without.class_sets) == {}
    # A level without sets keeps the string form, and so the hash, it had before sets existed
    assert "class_sets" not in str(without)


def test_sets_are_validated() -> None:
    with pytest.raises(ValueError, match="collides with a class"):
        _taxonomy(class_sets={"building": ["building", "vertical_thin"]})
    with pytest.raises(ValueError, match="at least two distinct classes"):
        _taxonomy(class_sets={"thin_or_sign": ["vertical_thin"], "manmade_or_thin": CLASSES})
    with pytest.raises(ValueError, match="not classes of the level"):
        _taxonomy(
            class_sets={"thin_or_sign": ["vertical_thin", "flag"], "manmade_or_thin": CLASSES}
        )
    with pytest.raises(ValueError, match="not a class of the level"):
        _taxonomy(class_sets={"manmade_or_thin": CLASSES})


def test_dataset_params_carry_category_aliases_into_the_scenario_and_the_hash() -> None:
    plain = DatasetParams(
        dataset_name="db",
        max_past_sweeps=2,
        max_future_sweeps=0,
        sample_steps=1,
    )
    aliased = DatasetParams(
        dataset_name="db",
        max_past_sweeps=2,
        max_future_sweeps=0,
        sample_steps=1,
        semantic_masks=True,
        category_aliases={"manmade": "legacy_manmade"},
    )
    assert plain.category_aliases == {}
    # Options at their default stay out of the string form, so existing tables keep their hash
    assert str(plain) == (
        "DatasetParams(dataset_name=db, max_past_sweeps=2, max_future_sweeps=0, sample_steps=1)"
    )
    assert plain != aliased
    assert "legacy_manmade" in str(aliased)
    scenario = T4Scenarios._build_scenario_data("scene/0", aliased)
    assert scenario.dataset_params.category_aliases == {"manmade": "legacy_manmade"}
    assert "legacy_manmade" in str(scenario)
    assert scenario.dataset_params.camera_frames is True
    lidar_only = DatasetParams(
        dataset_name="db",
        max_past_sweeps=2,
        max_future_sweeps=0,
        sample_steps=1,
        camera_frames=False,
    )
    assert lidar_only != plain
    assert not T4Scenarios._build_scenario_data("scene/0", lidar_only).dataset_params.camera_frames
