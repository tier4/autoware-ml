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

from collections.abc import Sequence
import unittest

import numpy as np

from autoware_ml.databases.box3d_pipelines.box3d_label_remapper import Box3DLabelRemapper
from autoware_ml.databases.schemas.box3d_schemas import Box3DDataModel
from autoware_ml.databases.taxonomy import LabelTaxonomy, LabelVocabulary

_CLASS_NAMES = ("car", "truck", "bicycle")
_IGNORE_INDEX = -1
_TAXONOMY = LabelTaxonomy(
    vocabulary=LabelVocabulary(
        {
            "vehicle.car": "car",
            "car": "car",
            "truck": "truck",
            "bicycle": "bicycle",
            "motorcycle": "motorcycle",
            "trailer": "trailer",
            "heavy_machine": "heavy_machine",
        }
    ),
    class_names=_CLASS_NAMES,
    # Fine names the level folds onto a class, and one it drops.
    class_mapping={
        "car": "car",
        "truck": "truck",
        "bicycle": "bicycle",
        "motorcycle": "bicycle",
        "trailer": "truck",
        "heavy_machine": None,
    },
    ignore_index=_IGNORE_INDEX,
    class_groups={"vehicle": ["car", "truck"], "cycle": ["bicycle"]},
)


def _build_box(label_name: str) -> Box3DDataModel:
    """Build a box as the resolver leaves it, named by its fine name and not yet indexed."""
    return Box3DDataModel(
        box3d_params=np.zeros(10, dtype=np.float64),
        box3d_instance_id="instance",
        box3d_dataset_label_name=label_name,
        box3d_label_name=label_name,
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

    def setUp(self) -> None:
        self.remapper = Box3DLabelRemapper(taxonomy=_TAXONOMY)

    def test_remaps_the_index_through_the_class_mapping(self) -> None:
        """Test that a box gets the index of the class its fine name maps to, name unchanged."""
        remapped = self.remapper([_build_box("motorcycle")])

        self.assertEqual(_labels(remapped), [("motorcycle", 2)])

    def test_a_fine_name_equal_to_its_class_gets_the_class_index(self) -> None:
        """Test that a fine name that is also a class gets that class index."""
        remapped = self.remapper([_build_box("car")])

        self.assertEqual(_labels(remapped), [("car", 0)])

    def test_remaps_the_trailers_a_merger_left(self) -> None:
        """Test that a trailer the merger did not absorb takes the index of the truck class."""
        remapped = self.remapper([_build_box("trailer"), _build_box("truck")])

        self.assertEqual(_labels(remapped), [("trailer", 1), ("truck", 1)])

    def test_a_fine_name_the_level_drops_gets_the_ignore_index(self) -> None:
        """Test that a fine name mapped to null keeps its box with the ignore index."""
        remapped = self.remapper([_build_box("heavy_machine"), _build_box("car")])

        self.assertEqual(_labels(remapped), [("heavy_machine", _IGNORE_INDEX), ("car", 0)])

    def test_name_outside_the_fine_names_raises(self) -> None:
        """Test that a box still carrying a raw name the resolver did not fold is an error."""
        with self.assertRaises(KeyError):
            self.remapper([_build_box("vehicle.car")])

    def test_the_dataset_label_name_is_never_rewritten(self) -> None:
        """Test that remapping leaves the original dataset label name in place."""
        remapped = self.remapper([_build_box("motorcycle")])

        self.assertEqual(remapped[0].box3d_dataset_label_name, "motorcycle")


if __name__ == "__main__":
    unittest.main()
