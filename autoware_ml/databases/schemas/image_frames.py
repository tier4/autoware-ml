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
class ImageFrameDatasetSchema(BaseFieldSchema):
    """
    Dataclass to define polars schema for columns related to image frames.
    """

    image_frame_id = DatasetTableColumn("image_frame_id", pl.String)
    image_sensor_id = DatasetTableColumn("image_sensor_id", pl.String)
    image_sensor_channel_name = DatasetTableColumn("image_sensor_channel_name", pl.String)
    image_timestamp_seconds = DatasetTableColumn("image_timestamp_seconds", pl.Float64)
    image_path = DatasetTableColumn("image_path", pl.String)
    image_height = DatasetTableColumn("image_height", pl.Int32)
    image_width = DatasetTableColumn("image_width", pl.Int32)
    cam2img = DatasetTableColumn("cam2img", pl.Array(pl.Float32, shape=(3, 3)))
    image_sensor_to_ego_pose_matrix = DatasetTableColumn(
        "image_sensor_to_ego_pose_matrix", pl.Array(pl.Float32, shape=(4, 4))
    )
    image_frame_ego_pose_to_global_matrix = DatasetTableColumn(
        "image_frame_ego_pose_to_global_matrix", pl.Array(pl.Float32, shape=(4, 4))
    )
    lidar2cam = DatasetTableColumn("lidar2cam", pl.Array(pl.Float32, shape=(4, 4)))
    lidar2img = DatasetTableColumn("lidar2img", pl.Array(pl.Float32, shape=(4, 4)))


class ImageFrameDataModel(BaseModel, DataModelInterface):
    """
    Image frame data model that can be shared by multiple datasets. It saves the metadata of a
    single camera keyframe.

    Attributes:
      image_frame_id: Image frame ID (sample data token).
      image_sensor_id: Image sensor ID (calibrated sensor token).
      image_sensor_channel_name: Image sensor channel name, e.g. CAM_FRONT.
      image_timestamp_seconds: Image timestamp in seconds.
      image_path: Image path.
      image_height: Image height in pixels. Set to None if it's not available.
      image_width: Image width in pixels. Set to None if it's not available.
      cam2img: Camera intrinsic matrix (3, 3).
      image_sensor_to_ego_pose_matrix: Transformation matrix from the image sensor of this frame
        to the ego pose of this image frame.
      image_frame_ego_pose_to_global_matrix: Transformation matrix from the ego pose of this
        image frame to the global frame.
      lidar2cam: Transformation matrix from LiDAR frame to camera frame (4, 4).
      lidar2img: Projection matrix from LiDAR frame to image plane (4, 4). Set to None if unavailable.
    """

    model_config = ConfigDict(frozen=True, strict=True, arbitrary_types_allowed=True)

    image_frame_id: str
    image_sensor_id: str
    image_sensor_channel_name: str
    image_timestamp_seconds: float
    image_path: str
    image_height: int | None
    image_width: int | None
    cam2img: npt.NDArray[np.float64]  # (3, 3)
    image_sensor_to_ego_pose_matrix: npt.NDArray[np.float64]  # (4, 4)
    image_frame_ego_pose_to_global_matrix: npt.NDArray[np.float64]  # (4, 4)
    lidar2cam: npt.NDArray[np.float64]  # (4, 4)
    lidar2img: npt.NDArray[np.float64] | None  # (4, 4) or None

    @property
    def cam2img_fp32(self) -> npt.NDArray[np.float32]:
        """
        Convert the camera intrinsic matrix to float32.

        Returns:
          npt.NDArray[np.float32]: Camera intrinsic matrix.
        """

        return self.cam2img.astype(np.float32)

    @property
    def image_sensor_to_ego_pose_matrix_fp32(self) -> npt.NDArray[np.float32]:
        """
        Convert the image sensor to ego pose matrix to float32.

        Returns:
          npt.NDArray[np.float32]: Image sensor to ego pose matrix.
        """

        return self.image_sensor_to_ego_pose_matrix.astype(np.float32)

    @property
    def image_frame_ego_pose_to_global_matrix_fp32(self) -> npt.NDArray[np.float32]:
        """
        Convert the image frame ego pose to global matrix to float32.

        Returns:
          npt.NDArray[np.float32]: Image frame ego pose to global matrix.
        """

        return self.image_frame_ego_pose_to_global_matrix.astype(np.float32)

    @property
    def lidar2cam_fp32(self) -> npt.NDArray[np.float32]:
        """
        Convert the lidar2cam matrix to float32.

        Returns:
          npt.NDArray[np.float32]: Lidar to camera transformation matrix.
        """

        return self.lidar2cam.astype(np.float32)

    @property
    def lidar2img_fp32(self) -> npt.NDArray[np.float32] | None:
        """
        Convert the lidar2img projection matrix to float32 if available.

        Returns:
          npt.NDArray[np.float32] | None: Lidar to image projection matrix.
        """

        if self.lidar2img is None:
            return None
        return self.lidar2img.astype(np.float32)

    def to_dictionary(self) -> Mapping[str, Any]:
        """
        Convert the image frame data model to a dictionary.

        Returns:
          Mapping[str, Any]: Dictionary representation of the image frame data model.
        """

        return {
            ImageFrameDatasetSchema.image_frame_id.name: self.image_frame_id,
            ImageFrameDatasetSchema.image_sensor_id.name: self.image_sensor_id,
            ImageFrameDatasetSchema.image_sensor_channel_name.name: self.image_sensor_channel_name,
            ImageFrameDatasetSchema.image_timestamp_seconds.name: self.image_timestamp_seconds,
            ImageFrameDatasetSchema.image_path.name: self.image_path,
            ImageFrameDatasetSchema.image_height.name: self.image_height,
            ImageFrameDatasetSchema.image_width.name: self.image_width,
            ImageFrameDatasetSchema.cam2img.name: self.cam2img_fp32,
            ImageFrameDatasetSchema.image_sensor_to_ego_pose_matrix.name: self.image_sensor_to_ego_pose_matrix_fp32,
            ImageFrameDatasetSchema.image_frame_ego_pose_to_global_matrix.name: self.image_frame_ego_pose_to_global_matrix_fp32,
            ImageFrameDatasetSchema.lidar2cam.name: self.lidar2cam_fp32,
            ImageFrameDatasetSchema.lidar2img.name: self.lidar2img_fp32,
        }

    @classmethod
    def load_from_dictionary(cls, data_model: Mapping[str, Any]) -> ImageFrameDataModel:
        """
        Load the image frame data model and decode it to the corresponding ImageFrameDataModel
        from a dictionary, which is deserialized from a Polars dataframe.

        Args:
          data_model: Dictionary representation of the image frame data model, which is
          deserialized from a Polars dataframe.

        Returns:
          ImageFrameDataModel: ImageFrameDataModel object.
        """

        raw_lidar2img = data_model.get(ImageFrameDatasetSchema.lidar2img.name)

        return cls(
            image_frame_id=data_model[ImageFrameDatasetSchema.image_frame_id.name],
            image_sensor_id=data_model[ImageFrameDatasetSchema.image_sensor_id.name],
            image_sensor_channel_name=data_model[
                ImageFrameDatasetSchema.image_sensor_channel_name.name
            ],
            image_timestamp_seconds=data_model[
                ImageFrameDatasetSchema.image_timestamp_seconds.name
            ],
            image_path=data_model[ImageFrameDatasetSchema.image_path.name],
            image_height=data_model[ImageFrameDatasetSchema.image_height.name],
            image_width=data_model[ImageFrameDatasetSchema.image_width.name],
            cam2img=np.asarray(
                data_model[ImageFrameDatasetSchema.cam2img.name], dtype=np.float64
            ),
            image_sensor_to_ego_pose_matrix=np.asarray(
                data_model[ImageFrameDatasetSchema.image_sensor_to_ego_pose_matrix.name],
                dtype=np.float64,
            ),
            image_frame_ego_pose_to_global_matrix=np.asarray(
                data_model[ImageFrameDatasetSchema.image_frame_ego_pose_to_global_matrix.name],
                dtype=np.float64,
            ),
            lidar2cam=np.asarray(
                data_model[ImageFrameDatasetSchema.lidar2cam.name],
                dtype=np.float64,
            ),
            lidar2img=(
                np.asarray(raw_lidar2img, dtype=np.float64)
                if raw_lidar2img is not None
                else None
            ),
        )
