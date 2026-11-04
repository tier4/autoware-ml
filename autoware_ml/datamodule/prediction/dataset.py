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

"""Dataset of the prediction task."""

from __future__ import annotations

import logging

import polars as pl
import torch

from autoware_ml.databases.schemas.dataset_schemas import DatasetTableSchema
from autoware_ml.databases.schemas.prediction import (
    PredictionDataModel,
    PredictionDatasetSchema,
)
from autoware_ml.dataclasses.batch.frame_meta import FrameMetaSample
from autoware_ml.dataclasses.batch.prediction import PredictionSourceSample
from autoware_ml.dataclasses.batch.sample_batch import ModelGTSample
from autoware_ml.datamodule.base_dataset import BaseDataset
from autoware_ml.transforms.base import TransformsCompose

logger = logging.getLogger(__name__)


class PredictionDataset(BaseDataset):
    """
    Dataset serving the scenarios of a prediction database, one sample per record.

    A record names the source of its scenario, its files relative to the database root; the
    dataset joins them onto its own root into a PredictionSourceSample, and the loading
    transforms of the pipeline read the agent tracks and the vector map from it. The samples
    carry no sensor data and no box or mask annotations.
    """

    _SERVED_COLUMNS = (
        DatasetTableSchema.SCENARIO_ID.name,
        DatasetTableSchema.TIMESTAMP_SECONDS.name,
        DatasetTableSchema.PREDICTION.name,
    )

    def __init__(
        self,
        database_root_path: str,
        *,
        transforms: TransformsCompose | None,
        dataset_records_dataframe: pl.DataFrame | None = None,
    ) -> None:
        """
        Initialize the dataset.

        The records are given here only when the dataset serves one corpus directly, without a
        datamodule. The datamodule binds the declared dataset to the records of its source.

        Args:
          database_root_path: Root directory of the database the dataset reads.
          transforms: Transforms applied to every sample, starting with the loading transforms
            of the prediction task.
          dataset_records_dataframe: Records of the corpus. None until bound to records.
        """

        super().__init__(
            database_root_path=database_root_path,
            dataset_records_dataframe=None,
            transforms=transforms,
            det3d_supervised=False,
            seg3d_supervised=False,
        )
        if dataset_records_dataframe is not None:
            self._serve_records(dataset_records_dataframe)
        logger.info(f"Initialized {self.__class__.__name__} with transforms: {self.transforms}")

    def _serve_records(self, dataset_records_dataframe: pl.DataFrame) -> None:
        """
        Keep the columns of the records the samples are built from.

        Args:
          dataset_records_dataframe: Records of the corpus.

        Raises:
          ValueError: If a record carries no prediction column, so the table was not built by
            a prediction database.
        """

        records = dataset_records_dataframe.select(self._SERVED_COLUMNS)
        prediction = pl.col(DatasetTableSchema.PREDICTION.name)
        # A null struct may be read back as a struct of null fields, and the tracks path is
        # never null in a prediction record
        absent = (
            prediction.is_null()
            | prediction.struct.field(PredictionDatasetSchema.tracks_relative_path.name).is_null()
        )
        num_absent = records.select(absent.sum()).item()
        if num_absent:
            raise ValueError(
                f"{num_absent} of {len(records)} records carry no prediction scenario, the "
                "record table was not built by a prediction database."
            )
        self.dataset_records_dataframe = records

    def get_data_sample(self, index: int) -> ModelGTSample:
        """
        Build the sample of one scenario before the transform pipeline.

        Args:
          index: Index of the record.

        Returns:
          ModelGTSample: The sample naming the source of its scenario, with every sensor and
            annotation field absent.
        """

        row = self.dataset_records_dataframe.row(index, named=True)
        scenario_id = row[DatasetTableSchema.SCENARIO_ID.name]
        timestamp_seconds = row[DatasetTableSchema.TIMESTAMP_SECONDS.name]
        prediction = PredictionDataModel.load_from_dictionary(
            row[DatasetTableSchema.PREDICTION.name]
        )
        source = PredictionSourceSample(
            dataset_format=prediction.dataset_format,
            tracks_path=(
                str(self.database_root_path / prediction.tracks_relative_path)
                if prediction.tracks_relative_path is not None
                else None
            ),
            map_path=(
                str(self.database_root_path / prediction.map_relative_path)
                if prediction.map_relative_path is not None
                else None
            ),
            scenario_key=prediction.scenario_key,
            focal_track_id=prediction.focal_track_id,
            first_step_index=prediction.first_step_index,
            current_step_index=prediction.current_step_index,
            num_steps=prediction.num_steps,
            step_seconds=prediction.step_seconds,
            num_tracks=prediction.num_tracks,
        )
        # The tracks are served in the frame of their source, so the frame of the sample is
        # the global frame itself
        frame_meta = FrameMetaSample(
            ego2global=torch.eye(4, dtype=torch.float32),
            scene_token=scenario_id,
            timestamp=timestamp_seconds,
            lanelet_map_path=None,
        )
        return ModelGTSample(
            lidar_point_cloud_samples=None,
            image_samples=None,
            point_cloud_data=None,
            camera_image_data=None,
            detection3d_gt_bboxes_3d=None,
            segmentation3d_gt_sample=None,
            frame_meta=frame_meta,
            prediction_source_sample=source,
        )
