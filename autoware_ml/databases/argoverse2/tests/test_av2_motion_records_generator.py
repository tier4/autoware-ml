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

"""Tests for the Argoverse 2 motion forecasting records generator."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("av2")

from autoware_ml.databases.argoverse2.av2_motion_records_generator import (  # noqa: E402
    OBSERVED_STEPS,
    AV2MotionRecordsGenerator,
)
from autoware_ml.databases.argoverse2.tests.av2_fixtures import (  # noqa: E402
    CITY,
    FOCAL_TRACK_ID,
    write_scenario,
)
from autoware_ml.databases.scenarios import DatasetParams, ScenarioData  # noqa: E402
from autoware_ml.types.dataset import DatasetFormat  # noqa: E402


def _scenario_data(scenario_id: str, split: str = "train") -> ScenarioData:
    return ScenarioData(
        dataset_params=DatasetParams(dataset_name=split, sample_steps=1),
        scenario_id=scenario_id,
        scenario_version=split,
    )


def test_one_record_per_scenario_with_relative_paths(tmp_path: Path) -> None:
    root = tmp_path / "av2"
    write_scenario(root / "train", "abc")

    records = AV2MotionRecordsGenerator(root, _scenario_data("abc")).generate_dataset_records()

    assert len(records) == 1
    record = records[0]
    assert record.scenario_id == "abc"
    assert record.sample_id == "abc"
    assert record.scenario_name == "abc"
    assert record.sample_index == 0
    assert record.location == CITY
    assert record.vehicle_type is None
    assert record.lidar_frames == []
    assert record.lidar_sources is None
    assert record.image_frames is None
    assert record.category_mapping is None
    assert record.boxes_3d is None
    # The current step is the last observed one, at 1.0 s + 49 * 0.1 s
    assert record.timestamp_seconds == pytest.approx(1.0 + 0.1 * (OBSERVED_STEPS - 1))

    prediction = record.prediction
    assert prediction is not None
    assert prediction.tracks_relative_path == "train/abc/scenario_abc.parquet"
    assert prediction.map_relative_path == "train/abc/log_map_archive_abc.json"
    assert prediction.dataset_format is DatasetFormat.AV2
    assert prediction.scenario_key is None
    assert prediction.focal_track_id == FOCAL_TRACK_ID
    assert prediction.first_step_index == 0
    assert prediction.current_step_index == OBSERVED_STEPS - 1
    assert prediction.num_steps == 110
    assert prediction.step_seconds == pytest.approx(0.1)
    assert prediction.num_tracks == 3


def test_a_test_split_scenario_ends_at_the_current_step(tmp_path: Path) -> None:
    root = tmp_path / "av2"
    write_scenario(root / "test", "t00", num_steps=OBSERVED_STEPS)

    (record,) = AV2MotionRecordsGenerator(
        root, _scenario_data("t00", split="test")
    ).generate_dataset_records()

    assert record.prediction.num_steps == OBSERVED_STEPS
    assert record.prediction.current_step_index == OBSERVED_STEPS - 1
    assert record.prediction.tracks_relative_path == "test/t00/scenario_t00.parquet"


def test_a_window_shorter_than_the_observed_steps_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "av2"
    write_scenario(root / "train", "short", num_steps=OBSERVED_STEPS - 1)

    with pytest.raises(ValueError, match="fewer than the 50 observed steps"):
        AV2MotionRecordsGenerator(root, _scenario_data("short")).generate_dataset_records()


def test_missing_files_are_rejected(tmp_path: Path) -> None:
    root = tmp_path / "av2"
    generator = AV2MotionRecordsGenerator(root, _scenario_data("none"))
    with pytest.raises(ValueError, match="missing its file .*scenario_none.parquet"):
        generator.generate_dataset_records()

    scenario_path = write_scenario(root / "train", "nomap")
    (scenario_path / "log_map_archive_nomap.json").unlink()
    with pytest.raises(ValueError, match="missing its file .*log_map_archive_nomap.json"):
        AV2MotionRecordsGenerator(root, _scenario_data("nomap")).generate_dataset_records()


def test_a_tracks_file_of_another_scenario_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "av2"
    scenario_path = write_scenario(root / "train", "abc")
    moved = root / "train" / "xyz"
    scenario_path.rename(moved)
    (moved / "scenario_abc.parquet").rename(moved / "scenario_xyz.parquet")
    (moved / "log_map_archive_abc.json").rename(moved / "log_map_archive_xyz.json")

    with pytest.raises(ValueError, match="belongs to scenario 'abc', not to 'xyz'"):
        AV2MotionRecordsGenerator(root, _scenario_data("xyz")).generate_dataset_records()
