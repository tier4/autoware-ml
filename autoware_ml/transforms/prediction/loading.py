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

"""Loading transforms of the prediction task.

A record of the prediction task names the source of its scenario; these transforms read it
into the dataset agnostic samples, the way LoadPointsFromFile reads the point cloud a record
names. The reader of every dataset format lives under ``databases/`` and is imported when a
sample of that format is loaded, so a format's devkit is only needed by the users of the format.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from autoware_ml.databases.taxonomy import LabelTaxonomy
from autoware_ml.dataclasses.batch.prediction import (
    AgentTracksSample,
    MapPolylinesSample,
    PredictionGTSample,
    PredictionSourceSample,
)
from autoware_ml.dataclasses.batch.sample_batch import ModelGTSample
from autoware_ml.transforms.base import BaseTransform
from autoware_ml.types.dataset import DatasetFormat
from autoware_ml.types.motion import PolylineType


class LoadAgentTracks(BaseTransform):
    """Load the agent tracks of the scenario a sample names into its prediction ground truth."""

    _required_keys = ["prediction_source_sample"]

    def __init__(self, taxonomy: LabelTaxonomy) -> None:
        """
        Initialize the loader.

        Args:
          taxonomy: Taxonomy the agent types are resolved with, the prediction taxonomy of the
            experiment over the vocabulary of the database.
        """

        if not isinstance(taxonomy, LabelTaxonomy):
            raise TypeError(f"taxonomy must be a LabelTaxonomy, got {type(taxonomy).__name__}.")
        super().__init__(probability=None)
        self.taxonomy = taxonomy

    def transform(self, model_gt_sample: ModelGTSample) -> ModelGTSample:
        """
        Read the tracks of the sample.

        Args:
          model_gt_sample: Sample naming the source of its scenario.

        Returns:
          ModelGTSample: The sample with its agent tracks set, keeping a map loaded before.
        """

        source = model_gt_sample.prediction_source_sample

        agents = self._read(source)
        previous = model_gt_sample.prediction_gt_sample
        prediction_gt_sample = PredictionGTSample(
            agents=agents,
            map_polylines=previous.map_polylines if previous is not None else None,
        )
        return model_gt_sample._replace(prediction_gt_sample=prediction_gt_sample)

    def _read(self, source: PredictionSourceSample) -> AgentTracksSample:
        """
        Read the tracks of a scenario with the reader of its dataset format.

        Args:
          source: Source of the scenario.

        Returns:
          AgentTracksSample: The agent tracks of the scenario.
        """

        if source.dataset_format is DatasetFormat.AV2:
            from autoware_ml.databases.argoverse2.av2_motion_reader import (
                read_agent_tracks,
            )

            return read_agent_tracks(
                Path(source.tracks_path), self.taxonomy, source.current_step_index
            )
        raise ValueError(f"No tracks reader for the {source.dataset_format.value} format.")

    def __repr__(self) -> str:
        """Return a string representation of the transform."""
        return f"{self.__class__.__name__}(taxonomy={self.taxonomy})"


def _polyline_type(value: PolylineType | int | str) -> PolylineType:
    """
    Coerce a configured polyline type, given as the enum, its value or its name.

    Args:
      value: The configured type.

    Returns:
      PolylineType: The type.
    """

    if isinstance(value, str):
        try:
            return PolylineType[value.upper()]
        except KeyError as error:
            raise ValueError(
                f"Unknown polyline type {value!r}, expected one of "
                f"{[member.name for member in PolylineType]}."
            ) from error
    return PolylineType(value)


class LoadVectorMap(BaseTransform):
    """Load the vector map of the scenario a sample names into its prediction ground truth."""

    _required_keys = ["prediction_source_sample", "prediction_gt_sample"]

    def __init__(
        self,
        num_points_per_polyline: int = 20,
        polyline_types: Sequence[PolylineType | int | str] = tuple(PolylineType),
    ) -> None:
        """
        Initialize the loader.

        Args:
          num_points_per_polyline: Number of points every polyline is resampled to.
          polyline_types: Kinds of polylines to keep, as the enum, its values or its names.
            Every kind by default.
        """

        if num_points_per_polyline < 2:
            raise ValueError(
                f"num_points_per_polyline must be at least 2, got {num_points_per_polyline}."
            )
        super().__init__(probability=None)
        self.num_points_per_polyline = num_points_per_polyline
        self.polyline_types = tuple(_polyline_type(value) for value in polyline_types)

    def transform(self, model_gt_sample: ModelGTSample) -> ModelGTSample:
        """
        Read the map of the sample.

        Args:
          model_gt_sample: Sample naming the source of its scenario, with its tracks loaded.

        Returns:
          ModelGTSample: The sample with its map polylines set.

        Raises:
          ValueError: If the scenario has no map.
        """

        source = model_gt_sample.prediction_source_sample
        if source.map_path is None:
            raise ValueError(
                f"{self.__class__.__name__}: the scenario of {source.tracks_path} has no map."
            )
        map_polylines = self._read(source.dataset_format, source.map_path)
        prediction_gt_sample = model_gt_sample.prediction_gt_sample._replace(
            map_polylines=map_polylines
        )
        return model_gt_sample._replace(prediction_gt_sample=prediction_gt_sample)

    def _read(self, dataset_format: DatasetFormat, map_path: str) -> MapPolylinesSample:
        """
        Read a map file with the reader of its dataset format.

        Args:
          dataset_format: Dataset format of the scenario.
          map_path: Path of the file.

        Returns:
          MapPolylinesSample: The polylines of the map.
        """

        if dataset_format is DatasetFormat.AV2:
            from autoware_ml.databases.argoverse2.av2_motion_reader import (
                read_vector_map,
            )

            return read_vector_map(
                Path(map_path),
                num_points_per_polyline=self.num_points_per_polyline,
                polyline_types=self.polyline_types,
            )
        raise ValueError(f"No map reader for the {dataset_format.value} format.")

    def __repr__(self) -> str:
        """Return a string representation of the transform."""
        return (
            f"{self.__class__.__name__}(num_points_per_polyline={self.num_points_per_polyline}, "
            f"polyline_types={[member.name for member in self.polyline_types]})"
        )
