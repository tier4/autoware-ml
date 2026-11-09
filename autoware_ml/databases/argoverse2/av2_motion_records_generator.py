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

"""Records generator of the Argoverse 2 Motion Forecasting dataset.

One scenario is one record. The record stores the paths of the tracks file and the map file
relative to the database root and the shape of the time window; the tracks themselves are read
by the loading transform at sample time.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

from autoware_ml.databases.scenarios import ScenarioData
from autoware_ml.databases.schemas.dataset_schemas import DatasetRecord
from autoware_ml.databases.schemas.motion import MotionDataModel
from autoware_ml.databases.schemas.utils import relative_to_database_root
from autoware_ml.types.dataset import DatasetFormat

logger = logging.getLogger(__name__)

# Number of observed steps at the start of every scenario window. The current step of a record
# is the last observed one; the steps after it are the future to forecast. The test split ships
# only the observed steps, so its windows end at the current step.
OBSERVED_STEPS = 50


class AV2MotionRecordsGenerator:
    """Builds the single record of one Argoverse 2 Motion Forecasting scenario."""

    def __init__(self, database_root_path: str | Path, scenario_data: ScenarioData) -> None:
        """
        Initialize the generator.

        Args:
          database_root_path: Root directory of the dataset, holding the split directories.
          scenario_data: Scenario to build the record of, whose dataset name is its split.
        """

        self.database_root_path = Path(database_root_path)
        self.scenario_data = scenario_data

    @property
    def scenario_path(self) -> Path:
        """Directory of the scenario, ``<root>/<split>/<scenario_id>``."""
        return (
            self.database_root_path
            / self.scenario_data.dataset_params.dataset_name
            / self.scenario_data.scenario_id
        )

    @property
    def tracks_path(self) -> Path:
        """Path of the ``scenario_<id>.parquet`` tracks file."""
        return self.scenario_path / f"scenario_{self.scenario_data.scenario_id}.parquet"

    @property
    def map_path(self) -> Path:
        """Path of the ``log_map_archive_<id>.json`` static map file."""
        return self.scenario_path / f"log_map_archive_{self.scenario_data.scenario_id}.json"

    def generate_dataset_records(self) -> Sequence[DatasetRecord]:
        """
        Build the record of the scenario.

        Returns:
          Sequence[DatasetRecord]: Exactly one record, the scenario.

        Raises:
          ValueError: If a file of the scenario is missing, the file names another scenario, or
            its window is shorter than the observed steps.
        """

        # The reader is the only module importing the av2 devkit, so it is imported here and
        # a database can be constructed and hashed without it
        from autoware_ml.databases.argoverse2.av2_motion_reader import read_scenario_metadata

        scenario_id = self.scenario_data.scenario_id
        for path in (self.tracks_path, self.map_path):
            if not path.is_file():
                raise ValueError(f"Scenario {scenario_id} is missing its file {path}.")

        metadata = read_scenario_metadata(self.tracks_path)
        if metadata.scenario_id != scenario_id:
            raise ValueError(
                f"The tracks file {self.tracks_path} belongs to scenario "
                f"{metadata.scenario_id!r}, not to {scenario_id!r}."
            )
        current_step_index = OBSERVED_STEPS - 1
        if metadata.num_steps <= current_step_index:
            raise ValueError(
                f"Scenario {scenario_id} has {metadata.num_steps} steps, fewer than the "
                f"{OBSERVED_STEPS} observed steps."
            )

        motion = MotionDataModel(
            dataset_format=DatasetFormat.AV2,
            tracks_relative_path=relative_to_database_root(
                self.tracks_path, self.database_root_path
            ),
            map_relative_path=relative_to_database_root(self.map_path, self.database_root_path),
            scenario_key=None,
            focal_track_id=metadata.focal_track_id,
            first_step_index=0,
            current_step_index=current_step_index,
            num_steps=metadata.num_steps,
            step_seconds=metadata.step_seconds,
            num_tracks=metadata.num_tracks,
        )
        record = DatasetRecord(
            scenario_id=scenario_id,
            sample_id=scenario_id,
            sample_index=0,
            timestamp_seconds=float(metadata.timestamps_seconds[current_step_index]),
            location=metadata.city,
            vehicle_type=None,
            scenario_name=scenario_id,
            traffic_cone_barrier_bbox_status=None,
            lidar_frames=[],
            lidar_sources=None,
            image_frames=None,
            category_mapping=None,
            boxes_3d=None,
            motion=motion,
        )
        return [record]
