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
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from pyquaternion import Quaternion
from t4_devkit.schema import SchemaName

from autoware_ml.databases.scenarios import DatasetParams, ScenarioData
from autoware_ml.databases.schemas.lidar_frames import LidarFrameDataModel
from autoware_ml.databases.t4dataset import t4records_generator
from autoware_ml.databases.t4dataset.t4records_generator import T4RecordsGenerator
from autoware_ml.types.sensor import LidarChannel

_DATASET_NAME = "db_test_v1"
_SCENARIO_ID = "scene"
_SCENARIO_VERSION = "0"
_SCENE_DIR = f"{_DATASET_NAME}/{_SCENARIO_ID}/{_SCENARIO_VERSION}"


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

    def __init__(self, root: Path, num_frames: int) -> None:
        self.root = root
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
        """Resolve a frame under the scene root, as the devkit does."""
        return str(self.root / _SCENE_DIR / "data" / "LIDAR_TOP" / f"{sample_data_token}.bin")


@pytest.fixture
def generator(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> T4RecordsGenerator:
    (tmp_path / _DATASET_NAME / _SCENARIO_ID / _SCENARIO_VERSION / "annotation").mkdir(parents=True)
    devkit = _ChainDevkit(tmp_path, num_frames=4)
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
        box3d_pipelines=[],
    )


def test_the_sweep_window_stops_at_the_scene_start(generator: T4RecordsGenerator) -> None:
    sample_frame = LidarFrameDataModel(
        lidar_frame_id="frame1",
        lidar_keyframe=True,
        lidar_sensor_id="sensor",
        lidar_sensor_channel_name=LidarChannel.LIDAR_TOP.value,
        lidar_timestamp_seconds=0.0,
        lidar_pointcloud_relative_path=generator.relative_to_database_root(
            generator.t4_devkit_dataset.get_sample_data_path("frame1")
        ),
        lidar_pointcloud_source_relative_path=None,
        lidar_pointcloud_num_features=5,
        lidar_sensor_to_ego_pose_matrix=np.eye(4),
        lidar_frame_ego_pose_to_global_matrix=np.eye(4),
        lidar_sensor_to_lidar_sweep_matrix=np.eye(4),
        lidar_pointcloud_semantic_mask_relative_path=None,
    )

    # The public extract_t4_sample_record() needs a full devkit scene (boxes, images, sources).
    # The sweep window is pure over the sample_data chain and is tested on its own here.
    frames = generator._extract_lidar_window(sample_frame)

    # The sample comes first, then the one past frame before the scene start, then the two
    # future frames, each side nearest first
    assert [frame.lidar_frame_id for frame in frames] == ["frame1", "frame0", "frame2", "frame3"]
    # Every sweep is stored relative to the database root, starting at the scene directory
    assert [frame.lidar_pointcloud_relative_path for frame in frames] == [
        f"{_SCENE_DIR}/data/LIDAR_TOP/{token}.bin"
        for token in ("frame1", "frame0", "frame2", "frame3")
    ]


@pytest.mark.parametrize(
    ("require_mask", "mask_indices", "sample_steps", "expected_indices"),
    [
        (True, [0, 10, 20], 1, [0, 10, 20]),
        (True, list(range(21)), 1, list(range(21))),
        (True, [1, 11, 20], 2, [1, 20]),
        (False, [0, 10, 20], 1, list(range(21))),
        (False, [], 2, list(range(0, 21, 2))),
    ],
)
def test_mask_selection_preserves_annotation_rates_and_original_indices(
    generator: T4RecordsGenerator,
    monkeypatch: pytest.MonkeyPatch,
    require_mask: bool,
    mask_indices: list[int],
    sample_steps: int,
    expected_indices: list[int],
) -> None:
    """The opt-in selects masks before sampling despite incorrect keyframe flags."""
    params = DatasetParams(
        dataset_name=_DATASET_NAME,
        sample_steps=sample_steps,
        require_semantic_mask=require_mask,
    )
    generator.scenario_data = generator.scenario_data.model_copy(update={"dataset_params": params})
    generator.sample_steps = sample_steps
    devkit = _ChainDevkit(generator.database_root_path, num_frames=21)
    devkit.sample = [
        SimpleNamespace(data={LidarChannel.LIDAR_TOP: f"frame{index}"}) for index in range(21)
    ]
    generator.t4_devkit_dataset = devkit
    generator.lidarseg_by_sample_data = {f"frame{index}": Mock() for index in mask_indices}
    extract = Mock(
        side_effect=lambda sample, index: SimpleNamespace(to_dataset_record=lambda: index)
    )
    monkeypatch.setattr(generator, "extract_t4_sample_record", extract)

    assert generator.generate_dataset_records() == expected_indices
    assert extract.call_count == len(expected_indices)


def test_a_mask_required_scene_without_masks_raises(generator: T4RecordsGenerator) -> None:
    """An opted-in scene with no semantic annotations is an error."""
    params = DatasetParams(dataset_name=_DATASET_NAME, sample_steps=1, require_semantic_mask=True)
    generator.scenario_data = generator.scenario_data.model_copy(update={"dataset_params": params})
    generator.t4_devkit_dataset.sample = [SimpleNamespace(data={LidarChannel.LIDAR_TOP: "frame0"})]

    with pytest.raises(ValueError, match="has no semantic masks"):
        generator.generate_dataset_records()
