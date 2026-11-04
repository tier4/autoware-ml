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

"""Prediction column of the record table.

A motion forecasting record stores only lightweight references to the source of the agent
tracks and the vector map of one scenario, plus the shape of its time window. The tracks
themselves are loaded at sample time, the way point clouds are, so a record table of hundreds
of thousands of scenarios stays small. The dataset format of the record fixes where the source
lives: a scenario dataset such as Argoverse 2 ships one tracks file and one map file per
scenario, a recording dataset such as T4Dataset keeps the tracks in the boxes of its frame
records and resolves the map of the scene through its map provider.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import polars as pl
from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from autoware_ml.databases.schemas.base_schemas import (
    BaseFieldSchema,
    DatasetTableColumn,
    DataModelInterface,
)
from autoware_ml.databases.schemas.utils import check_relative_path
from autoware_ml.types.dataset import DatasetFormat


@dataclass(frozen=True)
class PredictionDatasetSchema(BaseFieldSchema):
    """
    Dataclass to define the polars schema of the prediction struct column.
    """

    dataset_format = DatasetTableColumn("dataset_format", pl.String)
    tracks_relative_path = DatasetTableColumn("tracks_relative_path", pl.String)
    map_relative_path = DatasetTableColumn("map_relative_path", pl.String)
    scenario_key = DatasetTableColumn("scenario_key", pl.String)
    focal_track_id = DatasetTableColumn("focal_track_id", pl.String)
    first_step_index = DatasetTableColumn("first_step_index", pl.Int32)
    current_step_index = DatasetTableColumn("current_step_index", pl.Int32)
    num_steps = DatasetTableColumn("num_steps", pl.Int32)
    step_seconds = DatasetTableColumn("step_seconds", pl.Float64)
    num_tracks = DatasetTableColumn("num_tracks", pl.Int32)


class PredictionDataModel(BaseModel, DataModelInterface):
    """
    Prediction data model that can be shared by multiple datasets. It names the source of the
    agent tracks and the vector map of one scenario and describes its time window.

    Attributes:
      dataset_format: Dataset family of the record, which fixes the reader of its source and
        which of the fields below are set.
      tracks_relative_path: Path of the agent tracks file, relative to the database root. None
        for a recording dataset, whose tracks are the boxes of its frame records.
      map_relative_path: Path of the vector map file, relative to the database root. None for a
        recording dataset, whose map is resolved by the map provider of the dataset.
      scenario_key: Key of the scenario inside a container file that holds several scenarios.
        None when every scenario has its own files.
      focal_track_id: Identifier of the focal agent of the scenario. None when the scenario
        has no single focal agent.
      first_step_index: Index of the first step of the window inside the source, so one source
        can serve several windows. 0 when the source is the window.
      current_step_index: Index of the current step inside the window. Steps up to and
        including it are observed, steps after it are the future.
      num_steps: Number of steps in the window.
      step_seconds: Period between two steps in seconds.
      num_tracks: Number of agent tracks in the scenario.
    """

    model_config = ConfigDict(frozen=True, strict=True, arbitrary_types_allowed=True)

    dataset_format: DatasetFormat
    tracks_relative_path: str | None
    map_relative_path: str | None
    scenario_key: str | None
    focal_track_id: str | None
    first_step_index: int
    current_step_index: int
    num_steps: int
    step_seconds: float
    num_tracks: int

    @field_validator("tracks_relative_path")
    @classmethod
    def _check_tracks_relative_path(cls, path: str | None) -> str | None:
        """
        Reject a tracks path stored with a root, so a record never binds to one machine.

        Args:
          path: Stored tracks path, None for a recording dataset.

        Returns:
          str | None: The path, unchanged.
        """

        return None if path is None else check_relative_path(path, "tracks_relative_path")

    @field_validator("map_relative_path")
    @classmethod
    def _check_map_relative_path(cls, path: str | None) -> str | None:
        """
        Reject a map path stored with a root, so a record never binds to one machine.

        Args:
          path: Stored map path, None for a recording dataset.

        Returns:
          str | None: The path, unchanged.
        """

        return None if path is None else check_relative_path(path, "map_relative_path")

    @model_validator(mode="after")
    def _check_window_and_source(self) -> PredictionDataModel:
        """
        Check that the time window is well formed and the source fields fit the format.

        Returns:
          PredictionDataModel: The model, unchanged.
        """

        if self.first_step_index < 0:
            raise ValueError(f"first_step_index must not be negative, got {self.first_step_index}.")
        if self.num_steps <= 0:
            raise ValueError(f"num_steps must be positive, got {self.num_steps}.")
        if not 0 <= self.current_step_index < self.num_steps:
            raise ValueError(
                f"current_step_index must lie in [0, {self.num_steps}), got "
                f"{self.current_step_index}."
            )
        if self.step_seconds <= 0.0:
            raise ValueError(f"step_seconds must be positive, got {self.step_seconds}.")
        if self.num_tracks < 0:
            raise ValueError(f"num_tracks must not be negative, got {self.num_tracks}.")
        self._check_source_fields()
        return self

    def _check_source_fields(self) -> None:
        """
        Check that the source fields a format requires are set and the others are None.

        Raises:
          ValueError: If a field the format requires is None or a field it has no use for is set.
        """

        if self.dataset_format is DatasetFormat.AV2:
            required = ("tracks_relative_path", "map_relative_path", "focal_track_id")
            unused = ("scenario_key",)
        else:
            required = ()
            unused = ("tracks_relative_path", "map_relative_path", "scenario_key")
        for name in required:
            if getattr(self, name) is None:
                raise ValueError(f"{name} is required for the {self.dataset_format.value} format.")
        for name in unused:
            if getattr(self, name) is not None:
                raise ValueError(
                    f"{name} must be None for the {self.dataset_format.value} format, got "
                    f"{getattr(self, name)!r}."
                )

    def to_dictionary(self) -> Mapping[str, Any]:
        """
        Convert the prediction data model to a dictionary.

        Returns:
          Mapping[str, Any]: Dictionary representation of the prediction data model.
        """

        return {
            PredictionDatasetSchema.dataset_format.name: self.dataset_format.value,
            PredictionDatasetSchema.tracks_relative_path.name: self.tracks_relative_path,
            PredictionDatasetSchema.map_relative_path.name: self.map_relative_path,
            PredictionDatasetSchema.scenario_key.name: self.scenario_key,
            PredictionDatasetSchema.focal_track_id.name: self.focal_track_id,
            PredictionDatasetSchema.first_step_index.name: self.first_step_index,
            PredictionDatasetSchema.current_step_index.name: self.current_step_index,
            PredictionDatasetSchema.num_steps.name: self.num_steps,
            PredictionDatasetSchema.step_seconds.name: self.step_seconds,
            PredictionDatasetSchema.num_tracks.name: self.num_tracks,
        }

    @classmethod
    def load_from_dictionary(cls, data_model: Mapping[str, Any]) -> PredictionDataModel:
        """
        Load the prediction data model and decode it to the corresponding PredictionDataModel
        from a dictionary, which is deserialized from a Polars dataframe.

        Args:
          data_model: Dictionary representation of the prediction data model, which is
            deserialized from a Polars dataframe.

        Returns:
          PredictionDataModel: PredictionDataModel object.
        """

        return cls(
            dataset_format=DatasetFormat(data_model[PredictionDatasetSchema.dataset_format.name]),
            tracks_relative_path=data_model[PredictionDatasetSchema.tracks_relative_path.name],
            map_relative_path=data_model[PredictionDatasetSchema.map_relative_path.name],
            scenario_key=data_model[PredictionDatasetSchema.scenario_key.name],
            focal_track_id=data_model[PredictionDatasetSchema.focal_track_id.name],
            first_step_index=data_model[PredictionDatasetSchema.first_step_index.name],
            current_step_index=data_model[PredictionDatasetSchema.current_step_index.name],
            num_steps=data_model[PredictionDatasetSchema.num_steps.name],
            step_seconds=data_model[PredictionDatasetSchema.step_seconds.name],
            num_tracks=data_model[PredictionDatasetSchema.num_tracks.name],
        )
