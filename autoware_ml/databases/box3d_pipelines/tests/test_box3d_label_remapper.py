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

"""Unit tests for the 3D bounding box label remapper."""

from __future__ import annotations

from typing import Sequence
import unittest

import numpy as np

from autoware_ml.databases.box3d_pipelines.box3d_label_remapper import Box3DLabelRemapper
from autoware_ml.databases.schemas.box3d_schemas import Box3DDataModel
from autoware_ml.databases.taxonomy import LabelTaxonomy, LabelVocabulary

_CLASS_NAMES = ("car", "truck", "bicycle")
_IGNORE_INDEX = -1
# Fine names the level drops, they resolve to the ignore index
_DROPPED_FINE_NAMES = ("motorcycle", "trailer", "heavy_machine")
_TAXONOMY = LabelTaxonomy(
    vocabulary=LabelVocabulary({name: name for name in _CLASS_NAMES + _DROPPED_FINE_NAMES}),
    class_names=_CLASS_NAMES,
    class_mapping={name: name for name in _CLASS_NAMES}
    | {name: None for name in _DROPPED_FINE_NAMES},
    ignore_index=_IGNORE_INDEX,
    class_groups={"vehicle": ["car", "truck"], "cycle": ["bicycle"]},
)


def _build_box(dataset_label_name: str) -> Box3DDataModel:
    """Build a box as the records generator emits it, before any pipeline has run."""
    return Box3DDataModel(
        box3d_params=np.zeros(10, dtype=np.float64),
        box3d_instance_id="instance",
        box3d_dataset_label_name=dataset_label_name,
        box3d_label_name=dataset_label_name,
        box3d_label_index=_IGNORE_INDEX,
        box3d_num_lidar_points=10,
        box3d_num_radar_points=0,
        box3d_valid=True,
        box3d_attributes=set(),
        box3d_coordinate="LIDAR_COMMON",
    )


def _labels(boxes: Sequence[Box3DDataModel]) -> list[tuple[str, int]]:
    """Label name and index of every box, in order."""
    return [(box.box3d_label_name, box.box3d_label_index) for box in boxes]


class TestBox3DLabelRemapper(unittest.TestCase):
    """Unit tests for Box3DLabelRemapper."""

    def _build_remapper(self, label_remapper: dict[str, str | None]) -> Box3DLabelRemapper:
        """Build a remapper against the test taxonomy."""
        return Box3DLabelRemapper(taxonomy=_TAXONOMY, label_remapper=label_remapper)

    def test_remaps_name_and_index(self) -> None:
        """Test that a remapped box gets the target name and its class index in the taxonomy."""
        remapper = self._build_remapper({"motorcycle": "bicycle"})

        remapped = remapper([_build_box("motorcycle")])

        self.assertEqual(_labels(remapped), [("bicycle", 2)])

    def test_label_outside_the_classes_is_ignored(self) -> None:
        """Test that a box remapped to a fine name the level drops gets the ignore index."""
        remapper = self._build_remapper({"forklift": "heavy_machine"})

        remapped = remapper([_build_box("forklift")])

        self.assertEqual(_labels(remapped), [("heavy_machine", _IGNORE_INDEX)])

    def test_null_target_drops_the_box(self) -> None:
        """Test that a label remapped to null is removed from the boxes."""
        remapper = self._build_remapper({"forklift": None})

        remapped = remapper([_build_box("forklift"), _build_box("car")])

        self.assertEqual(_labels(remapped), [("car", 0)])

    def test_name_outside_the_vocabulary_raises(self) -> None:
        """Test that a box remapped to a name the vocabulary does not list is an error."""
        remapper = self._build_remapper({"forklift": "unlisted"})

        with self.assertRaises(KeyError):
            remapper([_build_box("forklift")])

    def test_a_later_pass_keeps_the_earlier_remapping(self) -> None:
        """
        Test that chained remappers compose. The pipeline runs a second remapper to fold the
        remaining trailers into trucks, and that pass must not undo the first remapping: a box
        it does not name keeps the label the earlier pass gave it.
        """
        first = self._build_remapper({"motorcycle": "bicycle", "trailer": "trailer"})
        second = self._build_remapper({"trailer": "truck"})

        remapped = second(first([_build_box("motorcycle"), _build_box("trailer")]))

        self.assertEqual(_labels(remapped), [("bicycle", 2), ("truck", 1)])

    def test_the_dataset_label_name_is_never_rewritten(self) -> None:
        """Test that remapping leaves the original dataset label name in place."""
        remapper = self._build_remapper({"motorcycle": "bicycle"})

        remapped = remapper([_build_box("motorcycle")])

        self.assertEqual(remapped[0].box3d_dataset_label_name, "motorcycle")


if __name__ == "__main__":
    unittest.main()
