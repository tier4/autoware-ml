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

"""Tests for the records of the frames between the key frames of a scene."""

from __future__ import annotations

import tempfile
import unittest
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
from pyquaternion import Quaternion
from t4_devkit.schema import SchemaName

from autoware_ml.databases.scenarios import DatasetParams, ScenarioData
from autoware_ml.databases.t4dataset import t4records_generator
from autoware_ml.databases.t4dataset.t4records_generator import T4RecordsGenerator
from autoware_ml.types.sensor import LidarChannel

_DATASET_NAME = "db_test_v1"
_SCENARIO_ID = "scene"
_SCENARIO_VERSION = "0"
_SCENE_DIR = f"{_DATASET_NAME}/{_SCENARIO_ID}/{_SCENARIO_VERSION}"
# Lidar frames at 10 Hz, the key frames at 0 and 1 s
_LIDAR_TIMES = [index * 100_000 for index in range(12)]
_KEY_FRAMES = {0, 10}
# A camera at 10 Hz, shifted by 30 ms against the lidar
_CAMERA_TIMES = [index * 100_000 + 30_000 for index in range(12)]


@dataclass(frozen=True)
class _SampleData:
    """Sample data record with the fields the generator reads."""

    token: str
    prev: str
    next: str
    timestamp: int
    is_key_frame: bool
    sample_token: str = ""
    calibrated_sensor_token: str = "sensor"
    ego_pose_token: str = "pose"
    info_filename: str | None = None


@dataclass(frozen=True)
class _Sample:
    """Sample record keyed on its lidar and camera frames."""

    token: str
    scene_token: str
    data: dict[str, str]


@dataclass(frozen=True)
class _Scene:
    """Scene record with its name."""

    token: str
    name: str


@dataclass(frozen=True)
class _Pose:
    """Calibrated sensor or ego pose record at the origin."""

    token: str
    rotation: Quaternion = Quaternion()
    translation: np.ndarray = field(default_factory=lambda: np.zeros(3))


def _chain(name: str, times: list[int], key_frames: set[int]) -> dict[str, _SampleData]:
    """One sensor chain named ``name0``, ``name1`` and so on."""
    tokens = [f"{name}{index}" for index in range(len(times))]
    return {
        token: _SampleData(
            token=token,
            prev=tokens[index - 1] if index > 0 else "",
            next=tokens[index + 1] if index < len(tokens) - 1 else "",
            timestamp=times[index],
            is_key_frame=index in key_frames,
            sample_token=f"sample{index}" if index in key_frames else "",
        )
        for index, token in enumerate(tokens)
    }


class _Devkit:
    """Devkit stub with a lidar and a camera chain and two key frame samples."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.lidarseg = []
        # The first listed sample is the second key frame, the walk still starts at the scene start
        self.sample = [
            _Sample(f"sample{index}", "scene", {"LIDAR_TOP": f"lidar{index}", "CAM": f"cam{index}"})
            for index in sorted(_KEY_FRAMES, reverse=True)
        ]
        self._tables = {
            SchemaName.SAMPLE_DATA: {
                **_chain("lidar", _LIDAR_TIMES, _KEY_FRAMES),
                **_chain("cam", _CAMERA_TIMES, _KEY_FRAMES),
            },
            SchemaName.SAMPLE: {sample.token: sample for sample in self.sample},
            SchemaName.SCENE: {"scene": _Scene(token="scene", name="scene_name")},
            SchemaName.CALIBRATED_SENSOR: {"sensor": _Pose(token="sensor")},
            SchemaName.EGO_POSE: {"pose": _Pose(token="pose")},
        }

    def get(self, schema: SchemaName, token: str) -> object:
        return self._tables[schema][token]

    def get_sample_data_path(self, sample_data_token: str) -> str:
        """Resolve a frame under the scene root, as the devkit does."""
        return str(self.root / _SCENE_DIR / "data" / f"{sample_data_token}.bin")

    def get_sample_data(self, sample_data_token: str, **_: object) -> tuple[str, list, None]:
        """Path of a frame without boxes, as the devkit returns it."""
        return self.get_sample_data_path(sample_data_token), [], None


class TestNonKeyFrameRecords(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        tmp_path = Path(tmp.name)
        (tmp_path / _SCENE_DIR / "annotation").mkdir(parents=True)
        devkit = _Devkit(tmp_path)
        self.enterContext(
            mock.patch.object(t4records_generator, "T4Devkit", lambda data_root, verbose: devkit)
        )
        self.enterContext(
            mock.patch.object(t4records_generator, "load_table", lambda path, schema: [])
        )
        # The record holds what the generator hands it, its own validation is tested elsewhere
        self.enterContext(
            mock.patch.object(
                t4records_generator, "T4SampleRecord", lambda **fields: SimpleNamespace(**fields)
            )
        )
        self.generator = T4RecordsGenerator(
            database_root_path=str(tmp_path),
            scenario_data=ScenarioData(
                dataset_params=DatasetParams(
                    dataset_name=_DATASET_NAME, sample_steps=1, non_key_frames=True
                ),
                scenario_id=_SCENARIO_ID,
                scenario_version=_SCENARIO_VERSION,
            ),
            lidar_channel=LidarChannel.LIDAR_TOP,
            lidar_pointcloud_num_features=5,
            box_annotation_dir="annotation",
            box3d_pipelines=[],
        )
        # The parts a full devkit scene serves, which these records only pass through
        self.enterContext(mock.patch.object(self.generator, "_extract_lidar_sources", lambda: []))
        self.enterContext(
            mock.patch.object(self.generator, "_extract_category_mapping", lambda: None)
        )

    def test_every_lidar_frame_of_the_scene_is_listed_in_time_order(self) -> None:
        """
        Input: a scene of 12 lidar frames whose first listed sample is its second key frame.
        Expected: the walk starts at the first frame of the scene and ends at its last one.
        Check: the tokens of every frame in order.
        """
        self.assertEqual(
            self.generator._lidar_frame_tokens(), [f"lidar{index}" for index in range(12)]
        )

    def test_a_non_key_frame_carries_no_boxes_and_no_camera_frames(self) -> None:
        """
        Input: the lidar frame at 0.4 s, between the two key frames of a scene with a camera.
        Expected: a record of that frame flagged as no key frame, with the lidar window around it and
          no box or camera frame.
        Check: the record fields.
        """
        lidar_frame = self.generator.t4_devkit_dataset.get(SchemaName.SAMPLE_DATA, "lidar4")

        record = self.generator.extract_t4_frame_record("lidar4", 4)

        self.assertEqual(record.frame_basic_metadata.sample_id, "lidar4")
        self.assertEqual(record.frame_basic_metadata.sample_index, 4)
        self.assertAlmostEqual(record.frame_basic_metadata.timestamp_seconds, 0.4)
        self.assertEqual(record.lidar_frame_data_models[0].lidar_frame_id, lidar_frame.token)
        self.assertFalse(record.lidar_frame_data_models[0].lidar_keyframe)
        self.assertEqual(record.boxes_3d_data_model, [])
        self.assertEqual(record.image_frame_data_models, [])

    def test_a_sample_on_a_lidar_frame_between_key_frames_is_refused(self) -> None:
        """
        Input: a sample on the lidar frame at 0.4 s, which is no key frame, and the key frame sample.
        Expected: a sample is a key frame, a record of it on another lidar frame would lose its labels.
        Check: ValueError naming the lidar frame, the key frame sample reads its lidar frame.
        """
        sample = _Sample("sample4", "scene", {"LIDAR_TOP": "lidar4"})

        with self.assertRaisesRegex(ValueError, "lidar4"):
            self.generator._extract_lidar_frame(sample)
        lidar_frame, _ = self.generator._extract_lidar_frame(
            self.generator.t4_devkit_dataset.sample[0]
        )
        self.assertTrue(lidar_frame.lidar_keyframe)

    def test_a_key_frame_keeps_its_sample_record(self) -> None:
        """
        Input: the lidar frame of the second key frame.
        Expected: it is read as its sample, with the index of the frame in the scene.
        Check: the sample and the index handed to the sample record extraction.
        """
        calls = []
        with mock.patch.object(
            self.generator,
            "extract_t4_sample_record",
            lambda sample, index: calls.append((sample, index)),
        ):
            self.generator.extract_t4_frame_record("lidar10", 10)

        self.assertEqual([(sample.token, index) for sample, index in calls], [("sample10", 10)])


class TestNonKeyFrameParams(unittest.TestCase):
    def test_the_non_key_frames_need_every_frame(self) -> None:
        """
        Input: a dataset serving its non-key frames with a sampling step of 2.
        Expected: the parameters are refused, the frames between key frames need every frame.
        Check: ValueError.
        """
        with self.assertRaisesRegex(ValueError, "sample_steps 1"):
            DatasetParams(dataset_name=_DATASET_NAME, sample_steps=2, non_key_frames=True)

    def test_the_setting_enters_the_cache_hash(self) -> None:
        """
        Input: the parameters with and without the non-key frames.
        Expected: the hash representation names the setting either way.
        Check: the representations.
        """
        plain = DatasetParams(dataset_name=_DATASET_NAME, sample_steps=1)
        streamed = DatasetParams(dataset_name=_DATASET_NAME, sample_steps=1, non_key_frames=True)

        self.assertEqual(
            plain.hash_repr,
            f"DatasetParams(dataset_name={_DATASET_NAME!r}, sample_steps=1, non_key_frames=False)",
        )
        self.assertTrue(streamed.hash_repr.endswith(", non_key_frames=True)"))
