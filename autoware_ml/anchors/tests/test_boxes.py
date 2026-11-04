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


"""Tests for the box anchors of the Sparse4D head."""

from __future__ import annotations

import tempfile
import unittest
from types import SimpleNamespace

import numpy as np
import polars as pl

from autoware_ml.anchors.boxes import BoxAnchorRecipe, cluster_boxes, load_gt_boxes
from autoware_ml.databases.schemas.box3d_schemas import Box3DDatasetSchema
from autoware_ml.databases.schemas.dataset_schemas import DatasetTableSchema
from autoware_ml.databases.taxonomy import LabelTaxonomy, LabelVocabulary

_CAR_TAXONOMY = LabelTaxonomy(LabelVocabulary({"car": "car"}), ["car"], {"car": "car"}, -1)
_RANGE = [-10.0, -10.0, -5.0, 10.0, 10.0, 5.0]


def _car_records(rows: list[list[float]]) -> pl.DataFrame:
    """
    One record holding a valid car box per row.

    Args:
      rows: Box parameters of every box, as the record table stores them.

    Returns:
      pl.DataFrame: The record.
    """
    return pl.DataFrame(
        {
            DatasetTableSchema.BOXES_3D.name: [
                [
                    {
                        Box3DDatasetSchema.BOX3D_PARAMS.name: row,
                        Box3DDatasetSchema.BOX3D_LABEL_NAME.name: "car",
                        Box3DDatasetSchema.BOX3D_VALID.name: True,
                    }
                    for row in rows
                ]
            ]
        },
        schema={
            DatasetTableSchema.BOXES_3D.name: pl.List(
                pl.Struct(
                    {
                        Box3DDatasetSchema.BOX3D_PARAMS.name: Box3DDatasetSchema.BOX3D_PARAMS.dtype,
                        Box3DDatasetSchema.BOX3D_LABEL_NAME.name: pl.String,
                        Box3DDatasetSchema.BOX3D_VALID.name: pl.Boolean,
                    }
                )
            )
        },
    )


def _recipe(cache_path: str, num_anchors: int, max_samples: int = 10) -> BoxAnchorRecipe:
    """
    Box recipe over the car taxonomy in the test range.

    Args:
      cache_path: Directory of the cached anchors.
      num_anchors: Number of anchors.
      max_samples: Most boxes the clustering reads.

    Returns:
      BoxAnchorRecipe: The recipe.
    """
    return BoxAnchorRecipe(
        cache_path=cache_path,
        taxonomy=_CAR_TAXONOMY,
        num_anchors=num_anchors,
        point_cloud_range=_RANGE,
        max_samples=max_samples,
        iterations=5,
        seed=0,
    )


class TestBoxAnchors(unittest.TestCase):
    def test_box_anchors_sit_at_the_cluster_centers(self) -> None:
        """
        Input: boxes around two centers, small boxes at one and large at the other.
        Expected: one anchor per center with the mean log size of its boxes and an identity heading.
        """
        rng = np.random.default_rng(0)
        near = np.concatenate(
            [rng.normal(0.0, 0.1, (50, 3)), np.ones((50, 3)), np.zeros((50, 1))], 1
        )
        far = np.concatenate(
            [rng.normal(20.0, 0.1, (50, 3)), np.full((50, 3), 4.0), np.zeros((50, 1))], 1
        )

        anchors = cluster_boxes(
            np.concatenate([near, far]).astype(np.float32), 2, iterations=10, seed=0
        )

        order = np.argsort(anchors[:, 0])
        np.testing.assert_allclose(anchors[order, :3], [[0.0] * 3, [20.0] * 3], atol=0.1)
        np.testing.assert_allclose(anchors[order, 3:6], [[0.0] * 3, [np.log(4.0)] * 3], atol=1e-5)
        np.testing.assert_array_equal(anchors[:, 6:8], [[0.0, 1.0], [0.0, 1.0]])

    def test_boxes_without_a_finite_positive_size_seed_no_anchor(self) -> None:
        """
        Input: a car box, one with a zero width and one with a NaN length.
        Expected: only the car box is loaded, the log size of the others would poison the anchors.
        """
        rows = [
            [1.0, 2.0, 0.5, 4.0, 2.0, 1.5, 0.0, 0.0, 0.0, 0.0],
            [3.0, 2.0, 0.5, 4.0, 0.0, 1.5, 0.0, 0.0, 0.0, 0.0],
            [5.0, 2.0, 0.5, float("nan"), 2.0, 1.5, 0.0, 0.0, 0.0, 0.0],
        ]

        boxes = load_gt_boxes(_car_records(rows), _CAR_TAXONOMY, _RANGE)

        np.testing.assert_array_equal(boxes, np.asarray([rows[0][:7]], dtype=np.float32))

    def test_the_recipe_clusters_the_boxes_in_range_of_every_source(self) -> None:
        """
        Input: two sources of one record each, one box in range in each and one out of range.
        Expected: one anchor at the center of each box in range, with its log size, the box out
          of range seeds none.
        """
        first = [[1.0, 2.0, 0.5, 4.0, 2.0, 1.5, 0.0, 0.0, 0.0, 0.0]]
        out_of_range = [50.0, 0.0, 0.0, 4.0, 2.0, 1.5, 0.0, 0.0, 0.0, 0.0]
        second = [[-5.0, -5.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0], out_of_range]
        split = SimpleNamespace(records=lambda: [_car_records(first), _car_records(second)])

        with tempfile.TemporaryDirectory() as cache_path:
            anchors = _recipe(cache_path, num_anchors=2).build(split)["anchors"]

        order = np.argsort(anchors[:, 0])
        np.testing.assert_allclose(anchors[order, :3], [[-5.0, -5.0, 0.0], [1.0, 2.0, 0.5]])
        np.testing.assert_allclose(
            anchors[order, 3:6], np.log([[1.0, 1.0, 1.0], [4.0, 2.0, 1.5]]), rtol=1e-6
        )

    def test_the_recipe_refuses_training_boxes_that_cannot_seed_the_anchors(self) -> None:
        """
        Input: a split whose only box lies out of range, then two boxes in range for three
          anchors, then two boxes in range of which the draw keeps one for two anchors.
        Expected: ValueError naming the range, then the box count, every time.
        """
        in_range = [[1.0, 2.0, 0.5, 4.0, 2.0, 1.5, 0.0, 0.0, 0.0, 0.0]] * 2
        out_of_range = [[50.0, 0.0, 0.0, 4.0, 2.0, 1.5, 0.0, 0.0, 0.0, 0.0]]
        cases = (
            (out_of_range, 2, 10, "No training box lies inside"),
            (in_range, 3, 10, "2 boxes cannot seed 3 anchors"),
            (in_range, 2, 1, "1 boxes cannot seed 2 anchors"),
        )
        for rows, num_anchors, max_samples, message in cases:
            with self.subTest(message=message), tempfile.TemporaryDirectory() as cache_path:
                split = SimpleNamespace(records=lambda rows=rows: [_car_records(rows)])
                recipe = _recipe(cache_path, num_anchors, max_samples)
                with self.assertRaisesRegex(ValueError, message):
                    recipe.build(split)
