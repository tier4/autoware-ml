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

"""Unit tests for the 3D bounding box mergers (truck + trailer merging)."""

from __future__ import annotations

from types import MappingProxyType
from typing import Sequence
import unittest

import numpy as np
from omegaconf import OmegaConf

from autoware_ml.databases.box3d_pipelines.box3d_merger import Box3DExtendLongerMerger
from autoware_ml.databases.schemas.box3d_schemas import Box3DDataModel
from autoware_ml.databases.taxonomy import LabelTaxonomy, LabelVocabulary
from autoware_ml.types.geometry import Box3DFieldIndex


class TestBox3DExtendLongerMerger(unittest.TestCase):
    """Unit tests for Box3DExtendLongerMerger."""

    def setUp(self) -> None:
        self.label_names = ("car", "truck", "trailer")
        self.taxonomy = LabelTaxonomy(
            vocabulary=LabelVocabulary({name: name for name in self.label_names}),
            class_names=self.label_names,
            class_mapping={name: name for name in self.label_names},
            ignore_index=-1,
            class_groups={"vehicle": list(self.label_names)},
        )
        self.target_labels = MappingProxyType({"truck": ["truck", "trailer"]})
        # Maximum front/back face-center distance in meters for two boxes to count as one object.
        self.proximity_distance_threshold = 2.0

    def _build_box(
        self,
        box: Sequence[float],
        label_name: str,
        dataset_label_name: str | None = None,
        num_lidar_points: int = 10,
        velocity: Sequence[float] = (0.0, 0.0),
        attributes: Sequence[str] = (),
    ) -> Box3DDataModel:
        """Build a box from ``(x, y, z, length, width, height, yaw)`` and a planar velocity."""
        box3d_params = np.zeros(len(Box3DFieldIndex), dtype=np.float64)
        box3d_params[: Box3DFieldIndex.YAW + 1] = box
        box3d_params[Box3DFieldIndex.VELOCITY_X] = velocity[0]
        box3d_params[Box3DFieldIndex.VELOCITY_Y] = velocity[1]
        return Box3DDataModel(
            box3d_params=box3d_params,
            box3d_instance_id=f"{label_name}_instance",
            box3d_dataset_label_name=dataset_label_name or label_name,
            box3d_label_name=label_name,
            box3d_label_index=self.label_names.index(label_name),
            box3d_num_lidar_points=num_lidar_points,
            box3d_num_radar_points=0,
            box3d_valid=True,
            box3d_attributes=set(attributes),
            box3d_coordinate="LIDAR_COMMON",
        )

    def _label_names(self, boxes: Sequence[Box3DDataModel]) -> list[str]:
        """Label name of every box, in order."""
        return [box.box3d_label_name for box in boxes]

    def _build_merger(
        self, target_labels: MappingProxyType[str, Sequence[str]] | None = None
    ) -> Box3DExtendLongerMerger:
        """Build a merger folding trucks and trailers into trucks against the test taxonomy."""
        return Box3DExtendLongerMerger(
            target_labels=self.target_labels if target_labels is None else target_labels,
            proximity_distance_threshold=self.proximity_distance_threshold,
            taxonomy=self.taxonomy,
        )

    def test_extend_longer_geometry_matches_reference(self) -> None:
        """Test that a truck and a collinear trailer 1 m ahead become one 9 m truck."""
        # truck length=4 at x=0, collinear trailer length=4 at x=5 (1 m gap) -> one 9 m box.
        merged = self._build_merger()(
            [
                self._build_box(
                    [0, 0, 0, 4, 2, 2, 0], "truck", num_lidar_points=10, velocity=(1.0, 0.0)
                ),
                self._build_box(
                    [5, 0, 0, 4, 2, 2, 0], "trailer", num_lidar_points=5, velocity=(1.0, 0.0)
                ),
            ]
        )

        self.assertEqual(len(merged), 1)
        merged_box = merged[0]
        self.assertEqual(merged_box.box3d_label_name, "truck")
        self.assertEqual(merged_box.box3d_label_index, self.label_names.index("truck"))
        np.testing.assert_allclose(
            merged_box.box3d_params,
            [2.5, 0.0, 0.0, 9.0, 2.0, 2.0, 0.0, 1.0, 0.0, 0.0],
            atol=1e-6,
        )
        self.assertEqual(merged_box.box3d_num_lidar_points, 15)

    def test_merged_box_keeps_first_source_identity_and_unions_attributes(self) -> None:
        """Test that the merged box keeps the first source identity and unions the attributes."""
        merged = self._build_merger()(
            [
                self._build_box([0, 0, 0, 4, 2, 2, 0], "truck", attributes=("moving",)),
                self._build_box([5, 0, 0, 4, 2, 2, 0], "trailer", attributes=("occluded",)),
            ]
        )

        self.assertEqual(merged[0].box3d_instance_id, "truck_instance")
        self.assertEqual(merged[0].box3d_dataset_label_name, "truck")
        self.assertEqual(merged[0].box3d_attributes, {"moving", "occluded"})

    def test_overlapping_pair_is_merged(self) -> None:
        """Test that an overlapping truck and trailer merge whatever the dataset label name."""
        merged = self._build_merger()(
            [
                self._build_box([0, 0, 0, 4, 2, 2, 0], "truck"),
                # Overlaps the truck; the merger matches on the (already remapped) label name.
                self._build_box(
                    [1, 0, 0, 4, 2, 2, 0], "trailer", dataset_label_name="vehicle.trailer"
                ),
            ]
        )

        self.assertEqual(self._label_names(merged), ["truck"])

    def test_distant_trailer_is_not_merged(self) -> None:
        """Test that a trailer far from the truck is left untouched."""
        # trailer back face (x=98) is far from truck front face (x=2): no merge.
        merged = self._build_merger()(
            [
                self._build_box([0, 0, 0, 4, 2, 2, 0], "truck"),
                self._build_box([100, 0, 0, 4, 2, 2, 0], "trailer"),
            ]
        )

        self.assertEqual(sorted(self._label_names(merged)), ["trailer", "truck"])

    def test_each_box_merges_at_most_once(self) -> None:
        """Test that one truck between two trailers is consumed by a single merge."""
        merged = self._build_merger()(
            [
                self._build_box([5, 0, 0, 4, 2, 2, 0], "trailer"),
                self._build_box([0, 0, 0, 4, 2, 2, 0], "truck"),
                self._build_box([-5, 0, 0, 4, 2, 2, 0], "trailer"),
            ]
        )

        self.assertEqual(sorted(self._label_names(merged)), ["trailer", "truck"])

    def test_noop_without_target_labels(self) -> None:
        """Test that a merger with no target labels passes every box through unchanged."""
        boxes = [
            self._build_box([0, 0, 0, 4, 2, 2, 0], "truck"),
            self._build_box([5, 0, 0, 4, 2, 2, 0], "trailer"),
        ]

        merged = self._build_merger(target_labels=MappingProxyType({}))(boxes)

        self.assertEqual(len(merged), len(boxes))
        for merged_box, box in zip(merged, boxes):
            self.assertIs(merged_box, box)

    def test_hydra_target_labels_are_accepted(self) -> None:
        """Test that target labels composed by Hydra, a DictConfig, drive the merge like a mapping."""
        merger = Box3DExtendLongerMerger(
            target_labels=OmegaConf.create({"truck": ["truck", "trailer"]}),
            proximity_distance_threshold=self.proximity_distance_threshold,
            taxonomy=self.taxonomy,
        )

        merged = merger(
            [
                self._build_box([0, 0, 0, 4, 2, 2, 0], "truck"),
                self._build_box([5, 0, 0, 4, 2, 2, 0], "trailer"),
            ]
        )

        self.assertEqual(self._label_names(merged), ["truck"])

    def test_target_label_without_exactly_two_source_labels_raises(self) -> None:
        """Test that a target label must be built from exactly two source labels."""
        with self.assertRaisesRegex(ValueError, "exactly 2 labels"):
            self._build_merger(
                target_labels=MappingProxyType({"truck": ["truck", "trailer", "car"]})
            )

    def test_extend_longer_merges_center_z_and_height_from_box_faces(self) -> None:
        """Test that the merged box spans from the lowest bottom face to the highest top face."""
        merged = self._build_merger()(
            [
                self._build_box([0, 0, 1, 4, 2, 2, 0], "truck"),
                self._build_box([1, 0, 3, 4, 2, 2, 0], "trailer"),
            ]
        )

        self.assertAlmostEqual(merged[0].box3d_params[Box3DFieldIndex.Z], 2.0, places=6)
        self.assertAlmostEqual(merged[0].box3d_params[Box3DFieldIndex.HEIGHT], 4.0, places=6)


if __name__ == "__main__":
    unittest.main()
