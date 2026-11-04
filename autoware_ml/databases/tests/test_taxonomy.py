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

"""Tests for the label vocabulary and the label taxonomy."""

from __future__ import annotations

import unittest

from autoware_ml.databases.taxonomy import LabelTaxonomy, LabelVocabulary

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
OFFLINE_CLASSES = ["car", "emergency_vehicle", "truck", "trailer"]
OFFLINE_COARSENING = {name: name for name in VOCABULARY.fine_names}


def _coarse() -> LabelTaxonomy:
    return LabelTaxonomy(VOCABULARY, ONLINE_CLASSES, ONLINE_COARSENING, -1)


def _fine() -> LabelTaxonomy:
    return LabelTaxonomy(VOCABULARY, OFFLINE_CLASSES, OFFLINE_COARSENING, -1)


class TestLabelVocabulary(unittest.TestCase):
    def test_resolves_fine_names_and_rejects_unlisted_raw_names(self) -> None:
        self.assertEqual(VOCABULARY.fine_names, ("car", "emergency_vehicle", "trailer", "truck"))
        self.assertEqual(VOCABULARY.fine_name("police_car"), "emergency_vehicle")
        self.assertIsNone(VOCABULARY.fine_name("unpainted"))
        self.assertEqual(
            VOCABULARY.unlisted(["car", "tree", "drainage", "unpainted"]), ["drainage", "tree"]
        )
        with self.assertRaisesRegex(KeyError, "not listed in the vocabulary"):
            VOCABULARY.fine_name("drainage")

    def test_string_form_is_canonical(self) -> None:
        reordered = LabelVocabulary(dict(reversed(list(VOCABULARY.class_renaming.items()))))

        self.assertEqual(str(reordered), str(VOCABULARY))
        self.assertEqual(reordered, VOCABULARY)

    def test_rejects_empty_definitions(self) -> None:
        for class_renaming, message in (
            ({}, "at least one fine label name"),
            ({"background": None}, "at least one fine label name"),
            ({"car": ""}, "non-empty strings or None"),
        ):
            with self.subTest(class_renaming=class_renaming):
                with self.assertRaisesRegex(ValueError, message):
                    LabelVocabulary(class_renaming)


class TestLabelTaxonomy(unittest.TestCase):
    def test_coarsens_fine_names_and_drops_the_rest(self) -> None:
        taxonomy = _coarse()

        self.assertEqual(taxonomy.class_names, ("car", "truck"))
        self.assertEqual(taxonomy.num_classes, 2)
        self.assertEqual(taxonomy.resolve_index("police_car"), 0)
        self.assertEqual(taxonomy.resolve_index("semi_trailer"), -1)
        self.assertEqual(taxonomy.resolve_index("unpainted"), -1)
        self.assertIsNone(taxonomy.class_name("trailer"))
        self.assertIsNone(taxonomy.class_name(None))
        self.assertEqual(taxonomy.class_index("truck"), 1)
        with self.assertRaisesRegex(KeyError, "not listed in the vocabulary"):
            taxonomy.resolve_index("drainage")
        with self.assertRaisesRegex(KeyError, "not a fine label name"):
            taxonomy.class_index("drainage")

    def test_a_taxonomy_can_train_every_fine_name(self) -> None:
        taxonomy = _fine()

        self.assertEqual(taxonomy.resolve_index("police_car"), 1)
        self.assertEqual(taxonomy.resolve_index("semi_trailer"), 3)

    def test_taxonomies_differ_in_their_string_form(self) -> None:
        self.assertNotEqual(_coarse(), _fine())
        self.assertEqual(str(_coarse()), str(_coarse()))
        self.assertIn("class_mapping=(car: car, emergency_vehicle: car", str(_coarse()))

    def test_rejects_inconsistent_definitions(self) -> None:
        cases = (
            ([], {}, -1, "at least one class name"),
            (["car", "car"], {}, -1, "unique"),
            (ONLINE_CLASSES, {}, 1, "ignore index 1 collides"),
            (
                ONLINE_CLASSES,
                {"car": "car", "emergency_vehicle": "car", "truck": "truck"},
                -1,
                "missing \\['trailer'\\]",
            ),
            (ONLINE_CLASSES, {**ONLINE_COARSENING, "bus": None}, -1, "unknown \\['bus'\\]"),
            (ONLINE_CLASSES, {**ONLINE_COARSENING, "trailer": "bus"}, -1, "not a trained class"),
        )
        for class_names, class_mapping, ignore_index, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    LabelTaxonomy(VOCABULARY, class_names, class_mapping, ignore_index)

    def test_a_class_without_fine_labels_is_a_placeholder(self) -> None:
        taxonomy = LabelTaxonomy(
            VOCABULARY, ["car", "truck", "vertical_thin"], ONLINE_COARSENING, -1
        )

        self.assertEqual(taxonomy.num_classes, 3)
        self.assertNotIn("vertical_thin", taxonomy.class_mapping.values())

    def test_the_mappings_cannot_be_mutated_after_construction(self) -> None:
        taxonomy = _coarse()

        with self.assertRaises(TypeError):
            taxonomy.class_mapping["car"] = "truck"
        with self.assertRaises(TypeError):
            taxonomy.vocabulary.class_renaming["car"] = "truck"


if __name__ == "__main__":
    unittest.main()
