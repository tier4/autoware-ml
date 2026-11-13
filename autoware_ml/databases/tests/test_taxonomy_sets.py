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

"""Unit tests for the label sets of a taxonomy level and the dataset options."""

import unittest
from collections.abc import Mapping, Sequence

from autoware_ml.databases.scenarios import DatasetParams
from autoware_ml.databases.t4dataset.t4scenarios import T4Scenarios
from autoware_ml.databases.taxonomy import LabelTaxonomy, LabelVocabulary

_VOCABULARY = LabelVocabulary(
    {
        "pole": "pole",
        "traffic_sign": "traffic_sign",
        "vertical_thin": "legacy_vertical_thin",
        "manmade": "legacy_manmade",
        "building": "building",
        "ignored": None,
    }
)
_CLASSES = ["building", "vertical_thin", "traffic_sign"]
_SET_MAPPING = {
    "pole": "vertical_thin",
    "traffic_sign": "traffic_sign",
    "legacy_vertical_thin": "legacy_vertical_thin_set",
    "legacy_manmade": "legacy_manmade_set",
    "building": "building",
}
_SETS = {
    "legacy_vertical_thin_set": ["vertical_thin", "traffic_sign"],
    "legacy_manmade_set": ["building", "vertical_thin", "traffic_sign"],
}


def _taxonomy(
    class_mapping: Mapping[str, str | None] = _SET_MAPPING,
    class_sets: Mapping[str, Sequence[str]] | None = _SETS,
    ignore_index: int = -1,
) -> LabelTaxonomy:
    """Taxonomy over the test vocabulary, with label sets unless told otherwise."""
    return LabelTaxonomy(
        vocabulary=_VOCABULARY,
        class_names=_CLASSES,
        class_mapping=class_mapping,
        ignore_index=ignore_index,
        class_sets=class_sets,
    )


def _dataset_params(**options: object) -> DatasetParams:
    """Dataset parameters of a small test corpus with the given dataset options."""
    return DatasetParams(dataset_name="db", sample_steps=1, **options)


class TestLabelSets(unittest.TestCase):
    """Unit tests for the label sets of LabelTaxonomy."""

    def test_set_fine_names_resolve_past_the_classes(self) -> None:
        """Test that fine labels mapped to a set resolve to num_classes + set position."""
        taxonomy = _taxonomy()

        self.assertEqual(taxonomy.num_classes, 3)
        self.assertEqual(
            tuple(taxonomy.class_sets), ("legacy_vertical_thin_set", "legacy_manmade_set")
        )
        expected_indices = {
            "pole": 1,
            "traffic_sign": 2,
            "vertical_thin": 3,
            "manmade": 4,
            "ignored": -1,
        }
        for raw_name, index in expected_indices.items():
            with self.subTest(raw_name=raw_name):
                self.assertEqual(taxonomy.resolve_index(raw_name), index)
        self.assertEqual(taxonomy.class_name("legacy_vertical_thin"), "legacy_vertical_thin_set")

    def test_sets_are_part_of_the_definition_string(self) -> None:
        """Test that sets enter the string form only when the level has some."""
        with_sets = _taxonomy()
        without = _taxonomy(
            class_mapping={
                "pole": "vertical_thin",
                "traffic_sign": "traffic_sign",
                "legacy_vertical_thin": "vertical_thin",
                "legacy_manmade": "building",
                "building": "building",
            },
            class_sets=None,
        )

        self.assertIn("legacy_vertical_thin_set", str(with_sets))
        self.assertNotEqual(with_sets, without)
        self.assertEqual(dict(without.class_sets), {})
        # A level without sets keeps the string form, and so the hash, it had before sets existed
        self.assertNotIn("class_sets", str(without))

    def test_invalid_sets_are_rejected(self) -> None:
        """Test that malformed label sets raise a descriptive error."""
        cases = {
            "collides with a trained class": {"building": ["building", "vertical_thin"]},
            "at least two distinct classes": {
                "legacy_vertical_thin_set": ["vertical_thin"],
                "legacy_manmade_set": _CLASSES,
            },
            "which are not trained": {
                "legacy_vertical_thin_set": ["vertical_thin", "flag"],
                "legacy_manmade_set": _CLASSES,
            },
            "not a trained class": {"legacy_manmade_set": _CLASSES},
        }
        for message, class_sets in cases.items():
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    _taxonomy(class_sets=class_sets)

    def test_ignore_index_must_not_collide_with_a_set_index(self) -> None:
        """Test that the ignore index lies outside the class and the set indices."""
        for ignore_index in (3, 4):
            with self.subTest(ignore_index=ignore_index):
                with self.assertRaisesRegex(ValueError, "class and label set indices 0..4"):
                    _taxonomy(ignore_index=ignore_index)
        self.assertEqual(_taxonomy(ignore_index=5).resolve_index("ignored"), 5)


class TestDatasetParams(unittest.TestCase):
    """Unit tests for the dataset options of DatasetParams and their scenario data."""

    def test_default_options_keep_the_string_form(self) -> None:
        """Test that options at their default stay out of the hash input."""
        plain = _dataset_params()

        self.assertEqual(plain.category_aliases, {})
        self.assertEqual(
            plain.hash_repr,
            "DatasetParams(dataset_name='db', sample_steps=1)",
        )

    def test_set_options_enter_the_string_form(self) -> None:
        """Test the exact hash input of every dataset option."""
        params = _dataset_params(
            require_semantic_mask=True,
            category_aliases={"manmade": "legacy_manmade"},
        )

        self.assertEqual(
            params.hash_repr,
            "DatasetParams(dataset_name='db', sample_steps=1, require_semantic_mask=True, "
            "category_aliases=[('manmade', 'legacy_manmade')])",
        )

    def test_category_aliases_reach_the_scenario(self) -> None:
        """Test that the aliases are carried into the scenario data and its hash input."""
        aliased = _dataset_params(
            require_semantic_mask=True, category_aliases={"manmade": "legacy_manmade"}
        )

        self.assertNotEqual(aliased, _dataset_params())
        self.assertIn("legacy_manmade", aliased.hash_repr)
        scenario = T4Scenarios._build_scenario_data("scene/0", aliased)
        self.assertEqual(scenario.dataset_params.category_aliases, {"manmade": "legacy_manmade"})
        self.assertIn("legacy_manmade", scenario.hash_repr)
