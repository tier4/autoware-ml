from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Any

import numpy as np
import numpy.typing as npt
import polars as pl
from pydantic import BaseModel, ConfigDict

from autoware_ml.databases.schemas.base_schemas import (
    BaseFieldSchema,
    DatasetTableColumn,
    DataModelInterface,
)


@dataclass(frozen=True)
class LidarSourceDatasetSchema(BaseFieldSchema):
    """
    Dataclass to define polars schema for columns related to lidar pointcloud.
    """

    channel_name = DatasetTableColumn("channel_name", pl.String)
    sensor_token = DatasetTableColumn("sensor_token", pl.String)
    translation = DatasetTableColumn("translation", pl.Array(pl.Float32, shape=(3,)))
    rotation = DatasetTableColumn("rotation", pl.Array(pl.Float32, shape=(4,)))


class LidarSourceDataModel(BaseModel, DataModelInterface):
    """
    Lidar source data model that can be shared by multiple datasets.

    Attributes:
      channel_name: Lidar source channel name.
      sensor_token: Lidar source sensor token.
      translation: Lidar source translation (3, ).
      rotation: Lidar source rotation (4, ).
    """

    model_config = ConfigDict(frozen=True, strict=True, arbitrary_types_allowed=True)

    channel_name: str
    sensor_token: str
    translation: npt.NDArray[np.float64]
    rotation: npt.NDArray[np.float64]

    @property
    def translation_fp32(self) -> npt.NDArray[np.float32]:
        """
        Convert the lidar source translations to float32.

        Returns:
          npt.NDArray[np.float32]: Lidar source translation.
        """

        return self.translation.astype(np.float32)

    @property
    def rotation_fp32(self) -> npt.NDArray[np.float32]:
        """
        Convert the lidar source rotations to float32.

        Returns:
          npt.NDArray[np.float32]: Lidar source rotation.
        """

        return self.rotation.astype(np.float32)

    def to_dictionary(self) -> Mapping[str, Any]:
        """
        Convert the lidar source data model to a dictionary.

        Returns:
          Mapping[str, Any]: Dictionary representation of the lidar source data model.
        """

        return {
            LidarSourceDatasetSchema.channel_name.name: self.channel_name,
            LidarSourceDatasetSchema.sensor_token.name: self.sensor_token,
            LidarSourceDatasetSchema.translation.name: self.translation_fp32,
            LidarSourceDatasetSchema.rotation.name: self.rotation_fp32,
        }

    @classmethod
    def load_from_dictionary(cls, data_model: Mapping[str, Any]) -> LidarSourceDataModel:
        """
        Load the lidar source data model and decode it to the corresponding LidarSourceDataModel
        from a dictionary, which is deserialized from a Polars dataframe.

        Args:
          data_model: Dictionary representation of the lidar source data model, which is
          deserialized from a Polars dataframe.

        Returns:
          LidarSourceDataModel: LidarSourceDataModel object.
        """

        return cls(
            channel_name=data_model[LidarSourceDatasetSchema.channel_name.name],
            sensor_token=data_model[LidarSourceDatasetSchema.sensor_token.name],
            translation=data_model[LidarSourceDatasetSchema.translation.name],
            rotation=data_model[LidarSourceDatasetSchema.rotation.name],
        )
