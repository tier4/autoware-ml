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

"""Tests for the label vocabulary and the label taxonomies."""

from __future__ import annotations

import pytest

from autoware_ml.databases.taxonomy import (
    DatabaseTaxonomy,
    DetectionTaxonomy,
    LabelTaxonomy,
    LabelVocabulary,
    SegmentationTaxonomy,
)
from autoware_ml.types.metrics import AgentKind

VOCABULARY = LabelVocabulary(
    {
        "car": "car",
        "vehicle.car": "car",
        "police_car": "emergency_vehicle",
        "truck": "truck",
        "trailer": "trailer",
        "semi_trailer": "trailer",
        "unpainted": None,
    }
)
ONLINE_CLASSES = ["car", "truck"]
ONLINE_COARSENING = {"car": "car", "emergency_vehicle": "car", "truck": "truck", "trailer": None}
ONLINE_GROUPS = {"grouped_vehicle": ["car", "truck"]}
OFFLINE_CLASSES = ["car", "emergency_vehicle", "truck", "trailer"]
OFFLINE_COARSENING = {name: name for name in VOCABULARY.fine_names}
OFFLINE_GROUPS = {
    "grouped_car": ["car", "emergency_vehicle"],
    "grouped_truck": ["truck", "trailer"],
}


def _online() -> LabelTaxonomy:
    return LabelTaxonomy(VOCABULARY, ONLINE_CLASSES, ONLINE_COARSENING, -1, ONLINE_GROUPS)


def _offline() -> LabelTaxonomy:
    return LabelTaxonomy(VOCABULARY, OFFLINE_CLASSES, OFFLINE_COARSENING, -1, OFFLINE_GROUPS)


def _detection(
    eval_range: dict[str, float] | None = None,
    collision_kinds: dict[str, str] | None = None,
    living_speeds: dict[str, float] | None = None,
) -> DetectionTaxonomy:
    return DetectionTaxonomy(
        VOCABULARY,
        ONLINE_CLASSES,
        ONLINE_COARSENING,
        -1,
        ONLINE_GROUPS,
        eval_range={"car": 121.0, "truck": 121.0} if eval_range is None else eval_range,
        collision_kinds=(
            {"car": "wheeled", "truck": "living"} if collision_kinds is None else collision_kinds
        ),
        living_speeds={"truck": 6.0} if living_speeds is None else living_speeds,
        partial_detection_classes=["car"],
        heatmap_pooling_classes=["car", "truck"],
    )


def _segmentation() -> SegmentationTaxonomy:
    return SegmentationTaxonomy(VOCABULARY, OFFLINE_CLASSES, OFFLINE_COARSENING, -1, OFFLINE_GROUPS)


def test_vocabulary_resolves_fine_names_and_rejects_unlisted_raw_names() -> None:
    assert VOCABULARY.fine_names == ("car", "emergency_vehicle", "trailer", "truck")
    assert VOCABULARY.fine_name("police_car") == "emergency_vehicle"
    assert VOCABULARY.fine_name("unpainted") is None
    assert VOCABULARY.unlisted(["car", "tree", "drainage", "unpainted"]) == ["drainage", "tree"]
    with pytest.raises(KeyError, match="not listed in the vocabulary"):
        VOCABULARY.fine_name("drainage")


def test_vocabulary_string_form_is_canonical() -> None:
    reordered = LabelVocabulary(dict(reversed(list(VOCABULARY.name_mapping.items()))))

    assert str(reordered) == str(VOCABULARY)
    assert reordered == VOCABULARY


def test_vocabulary_rejects_empty_definitions() -> None:
    with pytest.raises(ValueError, match="at least one fine label name"):
        LabelVocabulary({})
    with pytest.raises(ValueError, match="at least one fine label name"):
        LabelVocabulary({"background": None})
    with pytest.raises(ValueError, match="non-empty strings or None"):
        LabelVocabulary({"car": ""})


def test_online_level_coarsens_fine_names_and_drops_the_rest() -> None:
    taxonomy = _online()

    assert taxonomy.class_names == ("car", "truck")
    assert taxonomy.num_classes == 2
    assert taxonomy.resolve_index("police_car") == 0
    assert taxonomy.resolve_index("semi_trailer") == -1
    assert taxonomy.resolve_index("unpainted") == -1
    assert taxonomy.class_name("trailer") is None
    assert taxonomy.class_name(None) is None
    assert taxonomy.class_index("truck") == 1
    assert taxonomy.class_groups == {"grouped_vehicle": ("car", "truck")}
    with pytest.raises(KeyError, match="not listed in the vocabulary"):
        taxonomy.resolve_index("drainage")
    with pytest.raises(KeyError, match="not a fine label name"):
        taxonomy.class_index("drainage")


def test_offline_level_trains_every_fine_name() -> None:
    taxonomy = _offline()

    assert taxonomy.resolve_index("police_car") == 1
    assert taxonomy.resolve_index("semi_trailer") == 3


def test_levels_differ_in_their_string_form() -> None:
    assert _online() != _offline()
    assert str(_online()) == str(_online())
    assert "coarsening=(car: car, emergency_vehicle: car" in str(_online())


def test_taxonomy_rejects_inconsistent_definitions() -> None:
    with pytest.raises(ValueError, match="at least one class name"):
        LabelTaxonomy(VOCABULARY, [], {}, -1, {})
    with pytest.raises(ValueError, match="unique"):
        LabelTaxonomy(VOCABULARY, ["car", "car"], {}, -1, {})
    with pytest.raises(ValueError, match="ignore index 1 collides"):
        LabelTaxonomy(VOCABULARY, ONLINE_CLASSES, {}, 1, ONLINE_GROUPS)
    with pytest.raises(ValueError, match="missing \\['trailer'\\]"):
        LabelTaxonomy(
            VOCABULARY,
            ONLINE_CLASSES,
            {"car": "car", "emergency_vehicle": "car", "truck": "truck"},
            -1,
            ONLINE_GROUPS,
        )
    with pytest.raises(ValueError, match="unknown \\['bus'\\]"):
        LabelTaxonomy(
            VOCABULARY, ONLINE_CLASSES, {**ONLINE_COARSENING, "bus": None}, -1, ONLINE_GROUPS
        )
    with pytest.raises(ValueError, match="not a class of the level"):
        LabelTaxonomy(
            VOCABULARY, ONLINE_CLASSES, {**ONLINE_COARSENING, "trailer": "bus"}, -1, ONLINE_GROUPS
        )


def test_taxonomy_rejects_class_groups_that_do_not_partition_the_classes() -> None:
    with pytest.raises(ValueError, match="exactly one class group"):
        LabelTaxonomy(VOCABULARY, ONLINE_CLASSES, ONLINE_COARSENING, -1, {"grouped_car": ["car"]})
    with pytest.raises(ValueError, match="exactly one class group"):
        LabelTaxonomy(
            VOCABULARY,
            ONLINE_CLASSES,
            ONLINE_COARSENING,
            -1,
            {"grouped_car": ["car"], "grouped_vehicle": ["car", "truck"]},
        )


def test_a_class_without_fine_labels_is_a_placeholder() -> None:
    taxonomy = LabelTaxonomy(
        VOCABULARY,
        ["car", "truck", "vertical_thin"],
        ONLINE_COARSENING,
        -1,
        {"grouped_vehicle": ["car", "truck"], "grouped_structure": ["vertical_thin"]},
    )

    assert taxonomy.num_classes == 3
    assert "vertical_thin" not in taxonomy.coarsening.values()


def test_detection_taxonomy_carries_typed_evaluation_tables() -> None:
    taxonomy = _detection()

    assert taxonomy.eval_range == {"car": 121.0, "truck": 121.0}
    assert taxonomy.collision_kinds == {"car": AgentKind.WHEELED, "truck": AgentKind.LIVING}
    assert taxonomy.living_speeds == {"truck": 6.0}
    assert taxonomy.partial_detection_classes == ("car",)
    assert taxonomy.heatmap_pooling_classes == ("car", "truck")


def test_detection_taxonomy_rejects_tables_that_do_not_match_the_classes() -> None:
    with pytest.raises(ValueError, match="eval_range must have one entry per class"):
        _detection(eval_range={"car": 121.0})
    with pytest.raises(ValueError, match="collision_kinds must have one entry per class"):
        _detection(collision_kinds={"car": "wheeled", "truck": "wheeled", "bus": "wheeled"})
    with pytest.raises(ValueError, match="Unknown collision kinds \\['hovercraft'\\]"):
        _detection(collision_kinds={"car": "hovercraft", "truck": "living"})
    with pytest.raises(ValueError, match="missing \\['truck'\\]"):
        _detection(living_speeds={})
    with pytest.raises(ValueError, match="unknown \\['car'\\]"):
        _detection(living_speeds={"car": 3.0, "truck": 6.0})


def test_evaluation_tables_stay_out_of_the_string_form() -> None:
    assert _detection() == _detection(eval_range={"car": 50.0, "truck": 50.0})
    assert str(_detection()) == str(
        _detection(collision_kinds={"car": "static", "truck": "static"}, living_speeds={})
    )


def test_database_taxonomy_compares_both_levels() -> None:
    first = DatabaseTaxonomy(detection3d=_detection(), segmentation3d=_segmentation())
    second = DatabaseTaxonomy(detection3d=_detection(), segmentation3d=_segmentation())
    other = DatabaseTaxonomy(
        detection3d=_detection(),
        segmentation3d=SegmentationTaxonomy(
            VOCABULARY, ONLINE_CLASSES, ONLINE_COARSENING, -1, ONLINE_GROUPS
        ),
    )

    assert first == second
    assert first != other
    assert str(first).startswith("DatabaseTaxonomy(detection3d=DetectionTaxonomy(")


def test_database_taxonomy_rejects_untyped_levels() -> None:
    with pytest.raises(TypeError, match="detection3d must be a DetectionTaxonomy"):
        DatabaseTaxonomy(detection3d=_online(), segmentation3d=_segmentation())
    with pytest.raises(TypeError, match="segmentation3d must be a SegmentationTaxonomy"):
        DatabaseTaxonomy(detection3d=_detection(), segmentation3d=_detection())
