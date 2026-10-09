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

"""Unit tests for the semantic labels the 3D segmentation task serves."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np
import polars as pl

from autoware_ml.databases.schemas.dataset_schemas import DatasetTableSchema
from autoware_ml.databases.taxonomy import LabelVocabulary, SegmentationTaxonomy
from autoware_ml.datamodule.t4dataset.segmentation3d import T4Segmentation3DTask

VOCABULARY = LabelVocabulary(
    {"car": "car", "police_car": "car", "tree": "vegetation", "unpainted": None}
)
TAXONOMY = SegmentationTaxonomy(
    VOCABULARY,
    ("car", "vegetation"),
    {"car": "car", "vegetation": "vegetation"},
    -1,
    {"dynamic": ("car",), "static": ("vegetation",)},
)

# The record stores the mask path six segments deep, the task resolves it against the root
MASK_RELATIVE_PATH = "db_v1/scene/dataset_v1/data/lidar_token/0.bin"


def build_records_dataframe(mask_path: str | None, categories: dict[str, int]) -> pl.DataFrame:
    """Build a one row record table holding a lidar frame and a category mapping."""
    lidar_frame = {
        "lidar_frame_id": "frame",
        "lidar_keyframe": True,
        "lidar_sensor_id": "lidar",
        "lidar_sensor_channel_name": "LIDAR_TOP",
        "lidar_timestamp_seconds": 0.0,
        "lidar_pointcloud_path": "/absolute/" + MASK_RELATIVE_PATH,
        "lidar_pointcloud_source_path": None,
        "lidar_pointcloud_num_features": 5,
        "lidar_sensor_to_ego_pose_matrix": np.eye(4).tolist(),
        "lidar_frame_ego_pose_to_global_matrix": np.eye(4).tolist(),
        "lidar_sensor_to_lidar_sweep_matrix": np.eye(4).tolist(),
        "lidar_pointcloud_semantic_mask_path": mask_path,
    }
    return pl.DataFrame(
        {
            DatasetTableSchema.LIDAR_FRAMES.name: [[lidar_frame]],
            DatasetTableSchema.CATEGORY_MAPPING.name: [
                {
                    "category_names": list(categories),
                    "category_indices": list(categories.values()),
                }
            ],
        },
        schema={
            DatasetTableSchema.LIDAR_FRAMES.name: DatasetTableSchema.LIDAR_FRAMES.dtype,
            DatasetTableSchema.CATEGORY_MAPPING.name: DatasetTableSchema.CATEGORY_MAPPING.dtype,
        },
    )


class TestT4Segmentation3DTask(unittest.TestCase):
    """Resolution of the raw semantic mask into the class indices of the taxonomy."""

    def setUp(self) -> None:
        """Write a semantic mask holding one raw category index per point."""
        self.root = Path(tempfile.mkdtemp())
        mask_path = self.root / MASK_RELATIVE_PATH
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        np.array([0, 1, 2, 3], dtype=np.uint8).tofile(mask_path)

    def _build_task(
        self, mask_path: str | None = "/absolute/" + MASK_RELATIVE_PATH, **categories: int
    ) -> T4Segmentation3DTask:
        """Build the task over a one row record table."""
        return T4Segmentation3DTask(
            database_root_path=str(self.root),
            dataset_records_dataframe=build_records_dataframe(mask_path, categories),
            taxonomy=TAXONOMY,
        )

    def test_resolves_every_category_through_the_taxonomy(self) -> None:
        task = self._build_task(car=0, police_car=1, tree=2, unpainted=3)

        sample = task.get_data_sample(0)

        assert sample.segmentation3d_gt_sample is not None
        self.assertEqual(sample.segmentation3d_gt_sample.gt_semantic_mask.tolist(), [0, 0, 1, -1])
        self.assertEqual(sample.segmentation3d_gt_sample.ignore_index, -1)

    def test_rejects_a_raw_index_the_record_does_not_name(self) -> None:
        task = self._build_task(car=0, police_car=1, tree=2)

        with self.assertRaisesRegex(ValueError, "does not name"):
            task.get_data_sample(0)

    def test_a_frame_without_a_semantic_mask_is_rejected(self) -> None:
        task = self._build_task(mask_path=None, car=0)

        with self.assertRaisesRegex(ValueError, "carries no semantic mask"):
            task.get_data_sample(0)

    def test_keeps_only_the_columns_it_reads(self) -> None:
        task = self._build_task(car=0, police_car=1, tree=2, unpainted=3)

        assert task.dataset_records_dataframe is not None
        self.assertEqual(
            task.dataset_records_dataframe.columns,
            [DatasetTableSchema.LIDAR_FRAMES.name, DatasetTableSchema.CATEGORY_MAPPING.name],
        )
