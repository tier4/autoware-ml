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

"""Unit tests for detection3d datasets and collation."""

from __future__ import annotations

import numpy as np
import pickle
import pytest

from autoware_ml.datamodule.nuscenes.detection3d import NuscenesDetection3DDataModule
from autoware_ml.datamodule.samplers import DistributedWeightedRandomSampler
from autoware_ml.datamodule.t4dataset.detection3d import (
    FrameSamplingConfig,
    T4Detection3DDataModule,
    T4Detection3DDataset,
    compute_frame_sampling_weights,
)
from autoware_ml.transforms.boxes3d.loading import LoadAnnotations3D


class TestT4Detection3DDataset:
    def test_annotation_loader_maps_and_filters_instances(self) -> None:
        dataset = object.__new__(T4Detection3DDataset)
        dataset.class_names = ["car", "truck", "pedestrian"]
        dataset.name_mapping = {
            "vehicle.car": "car",
            "pedestrian.adult": "pedestrian",
            "trailer": "truck",
        }

        sample = {
            "instances": [
                {
                    "bbox_3d_isvalid": True,
                    "num_lidar_pts": 12,
                    "gt_nusc_name": "vehicle.car",
                    "bbox_3d": [1.0, 2.0, 0.5, 4.0, 1.8, 1.5, 0.2],
                    "velocity": [0.1, 0.0],
                },
                {
                    "bbox_3d_isvalid": False,
                    "num_lidar_pts": 0,
                    "gt_nusc_name": "pedestrian.adult",
                    "bbox_3d": [3.0, 1.0, 0.5, 0.6, 0.6, 1.7, 0.0],
                    "velocity": [0.0, 0.0],
                },
            ]
        }

        output = LoadAnnotations3D(name_mapping=dataset.name_mapping)(
            {
                "instances": sample["instances"],
                "class_names": dataset.class_names,
                "name_mapping": dataset.name_mapping,
            }
        )
        boxes = output["gt_boxes"]
        labels = output["gt_labels"]
        names = output["gt_names"]
        num_points = output["gt_num_points"]

        assert boxes.shape == (1, 9)
        assert labels.tolist() == [0]
        assert names.tolist() == ["car"]
        assert num_points.tolist() == [12]
        assert np.allclose(boxes[0, -2:], np.array([0.1, 0.0], dtype=np.float32))

    def test_get_data_info_exposes_metadata_for_loader_pipeline(self, tmp_path) -> None:
        lidar_path = tmp_path / "sample.bin"
        np.arange(10, dtype=np.float32).tofile(lidar_path)

        dataset = object.__new__(T4Detection3DDataset)
        dataset.data_infos = [
            {
                "token": "sample",
                "lidar_path": str(lidar_path),
                "lidar_points": {"num_pts_feats": 5},
                "instances": [],
                "sweeps": [],
            }
        ]
        dataset.data_root = str(tmp_path)
        dataset.class_names = []
        dataset.name_mapping = {}

        sample = dataset.get_data_info(0)

        assert sample["lidar_path"] == str(lidar_path)
        assert sample["num_pts_feats"] == 5
        assert sample["instances"] == []
        assert sample["sweeps"] == []

    def test_frame_sampling_weights_emphasize_rare_categories(self, tmp_path) -> None:
        ann_file = tmp_path / "infos.pkl"
        data = {
            "data_list": [
                *[
                    {
                        "token": f"car_only_{index}",
                        "lidar_path": f"a_{index}.bin",
                        "lidar_points": {"num_pts_feats": 5},
                        "instances": [
                            {
                                "bbox_3d_isvalid": True,
                                "gt_nusc_name": "car",
                                "bbox_3d": [0.0, 0.0, 0.0, 4.0, 1.8, 1.5, 0.0],
                                "num_lidar_pts": 10,
                            }
                        ],
                    }
                    for index in range(5)
                ],
                {
                    "token": "car_and_ped",
                    "lidar_path": "b.bin",
                    "lidar_points": {"num_pts_feats": 5},
                    "instances": [
                        {
                            "bbox_3d_isvalid": True,
                            "gt_nusc_name": "car",
                            "bbox_3d": [0.0, 0.0, 0.0, 4.0, 1.8, 1.5, 0.0],
                            "num_lidar_pts": 10,
                        },
                        {
                            "bbox_3d_isvalid": True,
                            "gt_nusc_name": "pedestrian",
                            "bbox_3d": [1.0, 1.0, 0.0, 0.6, 0.6, 1.2, 0.0],
                            "num_lidar_pts": 10,
                        },
                    ],
                },
            ],
        }
        with open(ann_file, "wb") as file:
            pickle.dump(data, file)

        dataset = T4Detection3DDataset(
            data_root=str(tmp_path),
            ann_file=str(ann_file),
            class_names=["car", "pedestrian"],
            name_mapping={},
            frame_sampling=FrameSamplingConfig(
                repeat_sampling_factor=1.0,
                object_bev_range=[-10.0, -10.0, 10.0, 10.0],
                low_pedestrian_height_threshold=1.5,
                low_pedestrian_bev_range=[-5.0, -5.0, 5.0, 5.0],
            ),
        )

        assert dataset.frame_weights[5] > dataset.frame_weights[0]
        assert np.isclose(dataset.frame_weights[5], 42**0.25)

    def test_frame_sampling_weights_exclude_filtered_attributes(self) -> None:
        data_infos = [
            {
                "instances": [
                    {
                        "gt_nusc_name": "car",
                        "bbox_3d": [0.0, 0.0, 0.0, 4.0, 1.8, 1.5, 0.0],
                        "num_lidar_pts": 10,
                    }
                ]
            },
            {
                "instances": [
                    {
                        "gt_nusc_name": "car",
                        "bbox_3d": [0.0, 0.0, 0.0, 4.0, 1.8, 1.5, 0.0],
                        "num_lidar_pts": 10,
                    },
                    {
                        "gt_nusc_name": "motorcycle",
                        "gt_attrs": ["vehicle_state.parked"],
                        "bbox_3d": [1.0, 1.0, 0.0, 1.8, 0.8, 1.2, 0.0],
                        "num_lidar_pts": 10,
                    },
                ]
            },
        ]
        frame_sampling = FrameSamplingConfig(
            repeat_sampling_factor=1.0,
            object_bev_range=[-10.0, -10.0, 10.0, 10.0],
            low_pedestrian_height_threshold=1.5,
            low_pedestrian_bev_range=[-5.0, -5.0, 5.0, 5.0],
        )

        weights = compute_frame_sampling_weights(
            data_infos,
            class_names=["car", "bicycle"],
            name_mapping={"car": "car", "motorcycle": "bicycle"},
            frame_sampling=frame_sampling,
            filter_attributes=[["motorcycle", "vehicle_state.parked"]],
        )

        assert weights == [1.0, 1.0]

    def test_frame_sampling_weights_exclude_physically_invalid_boxes(self) -> None:
        data_infos = [
            {
                "instances": [
                    {
                        "gt_nusc_name": "car",
                        "bbox_3d": [0.0, 0.0, 0.0, 4.0, 1.8, 1.5, 0.0],
                        "num_lidar_pts": 10,
                    }
                ]
            },
            {
                "instances": [
                    {
                        "gt_nusc_name": "pedestrian",
                        "bbox_3d": [1.0, 1.0, 0.0, 0.6, -0.6, 1.2, 0.0],  # negative dim
                        "num_lidar_pts": 10,
                    },
                    {
                        "gt_nusc_name": "pedestrian",
                        "bbox_3d": [2.0, 2.0, 0.0, 0.6, 0.6, 1.7, 0.0],
                        "velocity": [200.0, 200.0],  # impossible speed
                        "num_lidar_pts": 10,
                    },
                ]
            },
        ]
        frame_sampling = FrameSamplingConfig(
            repeat_sampling_factor=1.0,
            object_bev_range=[-10.0, -10.0, 10.0, 10.0],
            low_pedestrian_height_threshold=1.5,
            low_pedestrian_bev_range=[-5.0, -5.0, 5.0, 5.0],
        )

        weights = compute_frame_sampling_weights(
            data_infos,
            class_names=["car", "pedestrian"],
            name_mapping={"car": "car", "pedestrian": "pedestrian"},
            frame_sampling=frame_sampling,
        )

        # The second frame carries only physically invalid pedestrians, so it
        # contributes no category evidence and keeps the neutral weight.
        assert weights[1] == 1.0

    def test_dataset_frame_sampling_uses_annotation_filter_policy(self, tmp_path) -> None:
        ann_file = tmp_path / "infos.pkl"
        data = {
            "data_list": [
                {
                    "token": "car_only",
                    "lidar_path": "a.bin",
                    "lidar_points": {"num_pts_feats": 5},
                    "instances": [
                        {
                            "gt_nusc_name": "car",
                            "bbox_3d": [0.0, 0.0, 0.0, 4.0, 1.8, 1.5, 0.0],
                            "num_lidar_pts": 10,
                        }
                    ],
                },
                {
                    "token": "car_and_filtered_bicycle",
                    "lidar_path": "b.bin",
                    "lidar_points": {"num_pts_feats": 5},
                    "instances": [
                        {
                            "gt_nusc_name": "car",
                            "bbox_3d": [0.0, 0.0, 0.0, 4.0, 1.8, 1.5, 0.0],
                            "num_lidar_pts": 10,
                        },
                        {
                            "gt_nusc_name": "motorcycle",
                            "gt_attrs": ["vehicle_state.parked"],
                            "bbox_3d": [1.0, 1.0, 0.0, 1.8, 0.8, 1.2, 0.0],
                            "num_lidar_pts": 10,
                        },
                    ],
                },
            ],
        }
        with open(ann_file, "wb") as file:
            pickle.dump(data, file)

        dataset = T4Detection3DDataset(
            data_root=str(tmp_path),
            ann_file=str(ann_file),
            class_names=["car", "bicycle"],
            name_mapping={"car": "car", "motorcycle": "bicycle"},
            filter_attributes=[["motorcycle", "vehicle_state.parked"]],
            frame_sampling=FrameSamplingConfig(
                repeat_sampling_factor=1.0,
                object_bev_range=[-10.0, -10.0, 10.0, 10.0],
                low_pedestrian_height_threshold=1.5,
                low_pedestrian_bev_range=[-5.0, -5.0, 5.0, 5.0],
            ),
        )

        assert dataset.frame_weights == [1.0, 1.0]

    def test_train_dataloader_uses_distributed_weighted_sampler(self, tmp_path) -> None:
        data_root = tmp_path / "t4dataset"
        info_root = data_root / "info" / "detection3d"
        info_root.mkdir(parents=True)
        ann_file = info_root / "train.pkl"
        data = {
            "data_list": [
                {
                    "token": "sample",
                    "lidar_path": "sample.bin",
                    "lidar_points": {"num_pts_feats": 5},
                    "instances": [
                        {
                            "bbox_3d_isvalid": True,
                            "gt_nusc_name": "car",
                            "bbox_3d": [0.0, 0.0, 0.0, 4.0, 1.8, 1.5, 0.0],
                            "num_lidar_pts": 10,
                        }
                    ],
                }
            ]
        }
        with open(ann_file, "wb") as file:
            pickle.dump(data, file)

        datamodule = T4Detection3DDataModule(
            data_root=str(data_root),
            train_ann_file=str(ann_file),
            val_ann_file=str(ann_file),
            test_ann_file=str(ann_file),
            class_names=["car"],
            name_mapping={},
            train_frame_sampling={
                "repeat_sampling_factor": 0.30,
                "object_bev_range": [-10.0, -10.0, 10.0, 10.0],
                "low_pedestrian_height_threshold": 1.5,
                "low_pedestrian_bev_range": [-5.0, -5.0, 5.0, 5.0],
            },
        )
        datamodule.setup("fit")

        dataloader = datamodule.train_dataloader()

        assert isinstance(dataloader.sampler, DistributedWeightedRandomSampler)


class TestNuscenesDetection3DDataModule:
    def test_accepts_ptv3_common_detection_kwargs(self, tmp_path) -> None:
        ann_file = tmp_path / "nuscenes_infos_train.pkl"
        sample = {
            "token": "sample",
            "lidar_points": {"lidar_path": "sample.bin", "num_pts_feats": 5},
            "instances": [],
            "sweeps": [],
        }
        with open(ann_file, "wb") as file:
            pickle.dump({"data_list": [sample], "metainfo": {"classes": ["car"]}}, file)

        datamodule = NuscenesDetection3DDataModule(
            data_root=str(tmp_path),
            train_ann_file=ann_file.name,
            val_ann_file=ann_file.name,
            test_ann_file=ann_file.name,
            class_names=["car"],
            name_mapping=None,
            train_frame_sampling=None,
        )
        datamodule.setup("fit")

        train_sample = datamodule.train_dataset.get_data_info(0)

        assert train_sample["class_names"] == ["car"]
        assert train_sample["name_mapping"] is None
        assert train_sample["label_to_category"] == {0: "car"}

    def test_rejects_train_frame_sampling(self, tmp_path) -> None:
        with pytest.raises(ValueError, match="train_frame_sampling"):
            NuscenesDetection3DDataModule(
                data_root=str(tmp_path),
                train_ann_file="train.pkl",
                val_ann_file="val.pkl",
                test_ann_file="test.pkl",
                class_names=["car"],
                train_frame_sampling={"repeat_sampling_factor": 1.0},
            )
