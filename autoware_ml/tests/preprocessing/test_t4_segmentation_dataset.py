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

"""Unit tests for the T4 segmentation dataset."""

import pickle

import numpy as np

from autoware_ml.datamodule.t4dataset.segmentation3d import T4Segmentation3DDataset
from autoware_ml.transforms.base import TransformsCompose
from autoware_ml.transforms.point_cloud.loading import LoadPointsFromFile
from autoware_ml.transforms.segmentation3d.loading import LoadSegAnnotations3D


class TestT4Segmentation3DDataset:
    """Tests for T4 source slicing and label mapping."""

    def test_source_specific_loading(self, tmp_path) -> None:
        """The dataset metadata and loaders should reconstruct one source correctly."""
        points = np.array(
            [
                [10.0, 0.0, 0.0, 0.1, 0.0],
                [11.0, 0.0, 0.0, 0.2, 0.0],
                [20.0, 0.0, 0.0, 0.3, 0.0],
                [21.0, 0.0, 0.0, 0.4, 0.0],
            ],
            dtype=np.float32,
        )
        labels = np.array([1, 2, 2, 1], dtype=np.uint8)

        lidar_relpath = "sample/points.bin"
        mask_relpath = "sample/labels.bin"
        point_path = tmp_path / lidar_relpath
        mask_path = tmp_path / mask_relpath
        point_path.parent.mkdir(parents=True)
        points.tofile(point_path)
        labels.tofile(mask_path)

        sample = {
            "token": "sample-token",
            "lidar_points": {"lidar_path": lidar_relpath, "num_pts_feats": 5},
            "pts_semantic_mask_path": mask_relpath,
            "pts_semantic_mask_categories": {"car": 1, "noise": 2},
            "lidar_sources": {
                "LIDAR_A": {
                    "sensor_token": "a",
                    "translation": [10.0, 0.0, 0.0],
                    "rotation": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                },
                "LIDAR_B": {
                    "sensor_token": "b",
                    "translation": [20.0, 0.0, 0.0],
                    "rotation": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                },
            },
            "lidar_sources_info": {
                "sources": [
                    {"sensor_token": "a", "idx_begin": 0, "length": 2},
                    {"sensor_token": "b", "idx_begin": 2, "length": 2},
                ]
            },
        }

        ann_file = tmp_path / "infos.pkl"
        with open(ann_file, "wb") as file:
            pickle.dump({"data_list": [sample]}, file)

        dataset = T4Segmentation3DDataset(
            data_root=str(tmp_path),
            ann_file=str(ann_file),
            lidar_sources=["LIDAR_B"],
            dataset_transforms=TransformsCompose(
                pipeline=[
                    LoadPointsFromFile(load_dim=5, use_dim=4),
                    LoadSegAnnotations3D(
                        class_mapping={"car": 0, "noise": 1},
                        ignore_index=-1,
                    ),
                ]
            ),
        )

        loaded = dataset[0]

        np.testing.assert_allclose(
            loaded["points"][:, :3], np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        )
        np.testing.assert_array_equal(loaded["pts_semantic_mask"], np.array([1, 0], dtype=np.int64))
        assert loaded["idx_begin"] == 2
        assert loaded["length"] == 2
