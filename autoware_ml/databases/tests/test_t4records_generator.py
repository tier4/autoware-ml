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

"""Tests for the lidar sweep window of the T4 records generator."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from pyquaternion import Quaternion
from t4_devkit.schema import SchemaName

from autoware_ml.databases.scenarios import DatasetParams, ScenarioData
from autoware_ml.databases.schemas.lidar_frames import LidarFrameDataModel
from autoware_ml.databases.t4dataset import t4records_generator
from autoware_ml.databases.t4dataset.t4records_generator import T4RecordsGenerator
from autoware_ml.databases.taxonomy import DatabaseTaxonomy
from autoware_ml.types.sensor import LidarChannel

_DATASET_NAME = "db_test_v1"
_SCENARIO_ID = "scene"
_SCENARIO_VERSION = "0"
_SCENE_DIR = f"/data/{_DATASET_NAME}/{_SCENARIO_ID}/{_SCENARIO_VERSION}"


@dataclass(frozen=True)
class _SampleData:
    """Lidar sample data record with the fields the sweep walk reads."""

    token: str
    prev: str
    next: str
    calibrated_sensor_token: str = "sensor"
    ego_pose_token: str = "pose"
    is_key_frame: bool = True
    timestamp: int = 0


@dataclass(frozen=True)
class _Pose:
    """Calibrated sensor or ego pose record at the origin."""

    token: str
    rotation: Quaternion = Quaternion()
    translation: np.ndarray = field(default_factory=lambda: np.zeros(3))


class _ChainDevkit:
    """Devkit stub holding one chain of lidar frames named frame0, frame1 and so on."""

    def __init__(self, num_frames: int) -> None:
        tokens = [f"frame{index}" for index in range(num_frames)]
        sample_data = {
            token: _SampleData(
                token=token,
                prev=tokens[index - 1] if index > 0 else "",
                next=tokens[index + 1] if index < num_frames - 1 else "",
            )
            for index, token in enumerate(tokens)
        }
        self.lidarseg = []
        self._tables = {
            SchemaName.SAMPLE_DATA: sample_data,
            SchemaName.CALIBRATED_SENSOR: {"sensor": _Pose(token="sensor")},
            SchemaName.EGO_POSE: {"pose": _Pose(token="pose")},
        }

    def get(self, schema: SchemaName, token: str) -> _SampleData | _Pose:
        return self._tables[schema][token]

    def get_sample_data_path(self, sample_data_token: str) -> str:
        return f"{_SCENE_DIR}/data/LIDAR_TOP/{sample_data_token}.bin"


@pytest.fixture
def generator(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> T4RecordsGenerator:
    (tmp_path / _DATASET_NAME / _SCENARIO_ID / _SCENARIO_VERSION / "annotation").mkdir(parents=True)
    devkit = _ChainDevkit(num_frames=4)
    monkeypatch.setattr(t4records_generator, "T4Devkit", lambda data_root, verbose: devkit)
    monkeypatch.setattr(t4records_generator, "load_table", lambda path, schema: [])
    dataset_params = DatasetParams(dataset_name=_DATASET_NAME, sample_steps=1)
    return T4RecordsGenerator(
        database_root_path=str(tmp_path),
        scenario_data=ScenarioData(
            dataset_params=dataset_params,
            scenario_id=_SCENARIO_ID,
            scenario_version=_SCENARIO_VERSION,
        ),
        lidar_channel=LidarChannel.LIDAR_TOP,
        lidar_pointcloud_num_features=5,
        box_annotation_dir="annotation",
        taxonomy=MagicMock(spec=DatabaseTaxonomy),
        box3d_pipelines=[],
    )


def test_the_sweep_window_stops_at_the_scene_start(generator: T4RecordsGenerator) -> None:
    sample_frame = LidarFrameDataModel(
        lidar_frame_id="frame1",
        lidar_keyframe=True,
        lidar_sensor_id="sensor",
        lidar_sensor_channel_name=LidarChannel.LIDAR_TOP.value,
        lidar_timestamp_seconds=0.0,
        lidar_pointcloud_path=generator.t4_devkit_dataset.get_sample_data_path("frame1"),
        lidar_pointcloud_source_path=None,
        lidar_pointcloud_num_features=5,
        lidar_sensor_to_ego_pose_matrix=np.eye(4),
        lidar_frame_ego_pose_to_global_matrix=np.eye(4),
        lidar_sensor_to_lidar_sweep_matrix=np.eye(4),
        lidar_pointcloud_semantic_mask_path=None,
    )

    frames = generator._extract_lidar_window(sample_frame)

    # The sample comes first, then the one past frame before the scene start, then the two
    # future frames, each side nearest first
    assert [frame.lidar_frame_id for frame in frames] == ["frame1", "frame0", "frame2", "frame3"]
