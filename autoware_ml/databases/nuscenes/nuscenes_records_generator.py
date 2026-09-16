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

import logging
from pathlib import Path
from typing import Any, Mapping, Sequence, Tuple

import numpy as np
import numpy.typing as npt
from PIL import Image
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import Box
from pyquaternion import Quaternion

from autoware_ml.types.sensor import LidarChannel, Modality
from autoware_ml.types.spatial import CoordinateSystem
from autoware_ml.databases.box3d_pipelines.box3d_pipeline import Box3DPipeline
from autoware_ml.databases.schemas.frame_basic_metadata import FrameBasicMetadata
from autoware_ml.databases.schemas.dataset_schemas import DatasetRecord
from autoware_ml.databases.schemas.lidar_frames import LidarFrameDataModel
from autoware_ml.databases.schemas.lidar_sources import LidarSourceDataModel
from autoware_ml.databases.schemas.image_frames import ImageFrameDataModel
from autoware_ml.databases.schemas.category_mapping import CategoryMappingDataModel
from autoware_ml.databases.schemas.box3d_schemas import Box3DDataModel
from autoware_ml.databases.scenarios import ScenarioData
from autoware_ml.databases.t4dataset.t4sample_records import T4SampleRecord
from autoware_ml.utils.dataset import convert_quaternion_to_matrix

logger = logging.getLogger(__name__)


class NuScenesRecordsGenerator:
    """
    RecordsGenerator for NuScenesDataset. 

    It reuses T4SampleRecord as the intermediate per-sample container since the unified dataset
    row model (DatasetRecord) is shared across all dataset families.
    """

    def __init__(
        self,
        database_root_path: str,
        scenario_data: ScenarioData,
        max_sweeps: int,
        sample_steps: int,
        lidar_pointcloud_num_features: int,
        ignore_label_index: int,
        box3d_pipelines: Sequence[Box3DPipeline],
    ) -> None:
        """
        Initialize NuScenesRecordsGenerator.

        Args:
          database_root_path: Root path where the NuScenes version directories are stored.
          scenario_data: Scenario data, one NuScenes scene.
          max_sweeps: Max number of lidar sweeps to include, only for 3D, set to 0
            if skipping lidar sweep concatenation.
          sample_steps: Number of frames/samples to skip between each sample, set to 1
            if not skipping any samples/frames.
          lidar_pointcloud_num_features: Number of features of the lidar pointcloud.
          ignore_label_index: Label index to use for ignored labels in the box3d annotations.
          box3d_pipelines: List of box3d pipelines to process the box3d annotations.
        """

        self.database_root_path = Path(database_root_path)
        self.scenario_data = scenario_data
        self.max_sweeps = max_sweeps
        self.sample_steps = sample_steps
        self.lidar_pointcloud_num_features = lidar_pointcloud_num_features
        self.ignore_label_index = ignore_label_index
        self.box3d_pipelines = box3d_pipelines

        assert sample_steps > 0, "Sample steps must be greater than 0."
        assert max_sweeps >= 0, "Max sweeps must be greater than or equal to 0."

        self.nusc = self._construct_nuscenes_devkit_dataset()
        self.scene_record = self._find_scene_record()
        self.sample_tokens = self._build_ordered_sample_tokens()

    def _construct_nuscenes_devkit_dataset(self) -> NuScenes:
        """
        Construct the nuscenes-devkit NuScenes instance for this scenario's version.

        Returns:
          NuScenes: NuScenes devkit dataset instance.
        """

        return NuScenes(
            version=self.scenario_data.scenario_version,
            dataroot=str(self.database_root_path),
            verbose=False,
        )

    def _find_scene_record(self) -> Mapping[str, Any]:
        """
        Find this scenario's scene record by name.

        Returns:
          Mapping[str, Any]: NuScenes scene record.
        """

        for scene in self.nusc.scene:
            if scene["name"] == self.scenario_data.scenario_id:
                return scene
        raise ValueError(
            f"Scene {self.scenario_data.scenario_id} not found in version "
            f"{self.scenario_data.scenario_version}"
        )

    def _build_ordered_sample_tokens(self) -> Sequence[str]:
        """
        Build the list of sample tokens for this scene in chronological order, by walking the
        `next` pointer starting from `first_sample_token`.

        Returns:
          Sequence[str]: Sample tokens in chronological order.
        """

        tokens = []
        sample_token = self.scene_record["first_sample_token"]
        while sample_token:
            tokens.append(sample_token)
            sample_token = self.nusc.get("sample", sample_token)["next"]
        return tokens

    def generate_dataset_records(self) -> Sequence[DatasetRecord]:
        """
        Generate dataset records for this scene, in chronological order.

        Returns:
          Sequence[DatasetRecord]: Sequence of dataset records.
        """

        records = []
        logger.info(
            f"Generating dataset records for scenario: {self.scenario_data.scenario_id} "
            f"with sample steps: {self.sample_steps} and max sweeps: {self.max_sweeps}"
        )

        for sample_index in range(0, len(self.sample_tokens), self.sample_steps):
            sample_token = self.sample_tokens[sample_index]
            sample = self.nusc.get("sample", sample_token)
            nuscenes_sample_record = self.extract_nuscenes_sample_record(sample, sample_index)

            if nuscenes_sample_record is None:
                logger.info(
                    f"dataset_name: {self.scenario_data.dataset_name}, "
                    f"scenario_id: {self.scenario_data.scenario_id}, "
                    f"sample_index: {sample_index}, "
                    f"No lidar channel found in sample data"
                )
                continue

            records.append(nuscenes_sample_record.to_dataset_record())

        return records

    def _extract_sample_basic_metadata(
        self, sample: Mapping[str, Any], sample_index: int
    ) -> FrameBasicMetadata:
        """
        Extract basic metadata from a NuScenes sample.

        Args:
          sample: NuScenes sample record.
          sample_index: Sample index.

        Returns:
          FrameBasicMetadata: Frame basic metadata of the NuScenes sample.
        """

        return FrameBasicMetadata(
            scenario_id=self.scenario_data.scenario_id,
            sample_id=sample["token"],
            sample_index=sample_index,
            location=self.scenario_data.location,
            vehicle_type=self.scenario_data.vehicle_type,
            timestamp_seconds=sample["timestamp"] / 1e6,
            scenario_name=self.scene_record["name"],
        )

    def _extract_boxes_3d_annotations(
        self,
        sample: Mapping[str, Any],
        boxes_3d: Sequence[Box],
        lidar_sensor_to_ego_pose_matrix: npt.NDArray[np.float64],
        lidar_frame_ego_pose_to_global_matrix: npt.NDArray[np.float64],
    ) -> Sequence[Box3DDataModel]:
        """
        Extract boxes 3D annotations from a NuScenes sample and process them with the pipeline.

        Args:
          sample: NuScenes sample record.
          boxes_3d: Sequence of nuscenes-devkit Box objects from the sample, in sensor coordinates
            (as returned by `NuScenes.get_sample_data`).
          lidar_sensor_to_ego_pose_matrix: Transformation matrix (4, 4) from the lidar sensor to
            the ego pose, used to rotate `NuScenes.box_velocity()`'s global-frame velocity vector
            into the sensor frame (translation is irrelevant for a velocity vector).
          lidar_frame_ego_pose_to_global_matrix: Transformation matrix (4, 4) from the ego pose to
            the global frame, used for the same purpose.

        Returns:
          Sequence[Box3DDataModel]: Sequence of Box3DDataModel, which is the data model for the
            3D bounding boxes.
        """

        if not len(boxes_3d):
            return []

        # NuScenes.box_velocity() returns velocity in the **global** frame, unlike box.center
        # from get_sample_data() which is already in the sensor frame. 
        global_to_ego_rotation = lidar_frame_ego_pose_to_global_matrix[:3, :3].T
        ego_to_sensor_rotation = lidar_sensor_to_ego_pose_matrix[:3, :3].T
        global_to_sensor_rotation = ego_to_sensor_rotation @ global_to_ego_rotation

        boxes_3d_data_model = []
        for box3d in boxes_3d:
            sample_annotation_record = self.nusc.get("sample_annotation", box3d.token)
            velocity = self.nusc.box_velocity(box3d.token)
            velocity = np.nan_to_num(velocity, nan=0.0)
            velocity = global_to_sensor_rotation @ velocity

            # Convert the box3d to the Box3DFieldIndex format.
            box3d_params = np.asarray(
                [
                    box3d.center[0],
                    box3d.center[1],
                    box3d.center[2],
                    box3d.wlh[1],
                    box3d.wlh[0],
                    box3d.wlh[2],
                    box3d.orientation.yaw_pitch_roll[0],
                    velocity[0],
                    velocity[1],
                    velocity[2],
                ],
                dtype=np.float64,
            )
            box3d_valid = sample_annotation_record["num_lidar_pts"] > 0

            box_3d_attributes = set()
            for attribute_token in sample_annotation_record["attribute_tokens"]:
                attribute_record = self.nusc.get("attribute", attribute_token)
                box_3d_attributes.add(attribute_record["name"])

            boxes_3d_data_model.append(
                Box3DDataModel(
                    box3d_params=box3d_params,
                    box3d_instance_id=sample_annotation_record["instance_token"],
                    box3d_dataset_label_name=box3d.name,
                    box3d_label_name=box3d.name,
                    # Initially, set all label indices to the ignore label index
                    box3d_label_index=self.ignore_label_index,
                    box3d_num_lidar_points=sample_annotation_record["num_lidar_pts"],
                    box3d_num_radar_points=sample_annotation_record["num_radar_pts"],
                    box3d_valid=box3d_valid,
                    box3d_attributes=box_3d_attributes,
                    box3d_coordinate=CoordinateSystem.LIDAR_COMMON.name,
                )
            )

        for box3d_pipeline in self.box3d_pipelines:
            boxes_3d_data_model = box3d_pipeline(boxes_3d_data_model)

        return boxes_3d_data_model

    def _extract_lidar_pointcloud_semantic_mask_path(
        self, calibrated_lidar_sample_data_token: str
    ) -> str | None:
        """
        Extract the lidarseg semantic mask path for a lidar sample_data token, if lidarseg
        annotations are available for this NuScenes version.

        Args:
          calibrated_lidar_sample_data_token: Sample data token of the lidar frame.

        Returns:
          str | None: Absolute path to the lidarseg mask file, or None if not available.
        """

        if "lidarseg" not in self.nusc.table_names:
            return None

        lidarseg_record = self.nusc.get("lidarseg", calibrated_lidar_sample_data_token)
        return str(Path(self.nusc.dataroot) / lidarseg_record["filename"])

    def _extract_lidar_frame(
        self, sample: Mapping[str, Any], lidar_channel_name: str
    ) -> Tuple[LidarFrameDataModel, Sequence[Box]]:
        """
        Extract lidar frame records from a NuScenes sample.

        Args:
          sample: NuScenes sample record.
          lidar_channel_name: Lidar channel name.

        Returns:
          Tuple of:
            LidarFrameDataModel: Lidar records of the NuScenes sample.
            Sequence[Box]: Sequence of Box3D annotations in the lidar frame, in the sensor
              coordinate.
        """

        calibrated_lidar_sample_data_token = sample["data"][lidar_channel_name]
        sd_record = self.nusc.get("sample_data", calibrated_lidar_sample_data_token)
        cs_record = self.nusc.get("calibrated_sensor", sd_record["calibrated_sensor_token"])
        lidar_sensor_to_ego_matrix = convert_quaternion_to_matrix(
            rotation_quaternion=Quaternion(cs_record["rotation"]),
            translation=np.asarray(cs_record["translation"], dtype=np.float64),
            convert_to_float32=False,
        )

        lidar_path, box3d, _ = self.nusc.get_sample_data(calibrated_lidar_sample_data_token)

        ego_pose_record = self.nusc.get("ego_pose", sd_record["ego_pose_token"])
        lidar_frame_ego_pose_to_global_matrix = convert_quaternion_to_matrix(
            rotation_quaternion=Quaternion(ego_pose_record["rotation"]),
            translation=np.asarray(ego_pose_record["translation"], dtype=np.float64),
            convert_to_float32=False,
        )

        lidar_pointcloud_semantic_mask_path = self._extract_lidar_pointcloud_semantic_mask_path(
            calibrated_lidar_sample_data_token=calibrated_lidar_sample_data_token,
        )

        lidar_frame_data_model = LidarFrameDataModel(
            lidar_frame_id=calibrated_lidar_sample_data_token,
            lidar_keyframe=sd_record["is_key_frame"],
            lidar_sensor_id=cs_record["token"],
            lidar_sensor_channel_name=lidar_channel_name,
            lidar_timestamp_seconds=sd_record["timestamp"] / 1e6,
            lidar_pointcloud_path=lidar_path,
            lidar_pointcloud_source_path=None,
            lidar_pointcloud_num_features=self.lidar_pointcloud_num_features,
            lidar_sensor_to_ego_pose_matrix=lidar_sensor_to_ego_matrix,
            lidar_frame_ego_pose_to_global_matrix=lidar_frame_ego_pose_to_global_matrix,
            lidar_sensor_to_lidar_sweep_matrix=np.eye(4),
            lidar_pointcloud_semantic_mask_path=lidar_pointcloud_semantic_mask_path,
        )
        return lidar_frame_data_model, box3d

    def _compute_sensor_transformation_matrices(
        self,
        sensor_sample_data_record: Mapping[str, Any],
        selected_sensor_to_ego_pose_matrix: npt.NDArray[np.float64],
        selected_sensor_frame_ego_pose_to_global_matrix: npt.NDArray[np.float64],
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """
        Compute transformation matrices for a sensor.

        Args:
            sensor_sample_data_record: Sample data record of the sensor.
            selected_sensor_to_ego_pose_matrix: Transformation matrix from the selected
              sensor to its' the ego pose.
            selected_sensor_frame_ego_pose_to_global_matrix: Transformation matrix from the selected
              sensor frame ego pose to the global frame.

        Returns:
            Tuple of transformation matrices:
              1. Sensor frame ego pose to global matrix (4x4)
              2. Selected sensor to sensor transformation matrix (4x4)
        """

        sensor_calibrated_sensor_record = self.nusc.get(
            "calibrated_sensor", sensor_sample_data_record["calibrated_sensor_token"]
        )
        sensor_ego_pose_record = self.nusc.get(
            "ego_pose", sensor_sample_data_record["ego_pose_token"]
        )

        sensor_frame_ego_pose_to_global_matrix = convert_quaternion_to_matrix(
            rotation_quaternion=Quaternion(sensor_ego_pose_record["rotation"]),
            translation=np.asarray(sensor_ego_pose_record["translation"], dtype=np.float64),
            convert_to_float32=False,
        )

        sensor_to_ego_pose_matrix = convert_quaternion_to_matrix(
            rotation_quaternion=Quaternion(sensor_calibrated_sensor_record["rotation"]),
            translation=np.asarray(sensor_calibrated_sensor_record["translation"], dtype=np.float64),
            convert_to_float32=False,
        )

        # Compute the transformation matrix of sensor to the selected sensor coordinate
        # Sensor -> sensor frame ego pose -> global -> selected sensor frame ego pose -> selected sensor
        sensor_to_selected_sensor_matrix = (
            np.linalg.inv(selected_sensor_to_ego_pose_matrix)
            @ np.linalg.inv(selected_sensor_frame_ego_pose_to_global_matrix)
            @ sensor_frame_ego_pose_to_global_matrix
            @ sensor_to_ego_pose_matrix
        )
        return sensor_frame_ego_pose_to_global_matrix, sensor_to_selected_sensor_matrix

    def _extract_lidar_sweeps(
        self, lidar_frame_data_model: LidarFrameDataModel
    ) -> Sequence[LidarFrameDataModel]:
        """
        Extract multi-sweep lidar metadata from a NuScenes sample.

        Args:
            lidar_frame_data_model: Lidar frame data model of the key frame to walk sweeps back
              from.

        Returns:
            Sequence[LidarFrameDataModel]: Lidar sweep metadata corresponding to the current
              lidar frame.
        """

        current_lidar_sample_data_token = lidar_frame_data_model.lidar_frame_id

        lidar_frame_data_models = []
        current_sample_data_record = self.nusc.get("sample_data", current_lidar_sample_data_token)

        for _ in range(self.max_sweeps):
            if not current_sample_data_record["prev"]:
                break

            current_sample_data_record = self.nusc.get(
                "sample_data", current_sample_data_record["prev"]
            )
            current_cs_record = self.nusc.get(
                "calibrated_sensor", current_sample_data_record["calibrated_sensor_token"]
            )
            current_lidar_sensor_to_ego_matrix = convert_quaternion_to_matrix(
                rotation_quaternion=Quaternion(current_cs_record["rotation"]),
                translation=np.asarray(current_cs_record["translation"], dtype=np.float64),
                convert_to_float32=False,
            )

            lidar_sweep_transformations = self._compute_sensor_transformation_matrices(
                sensor_sample_data_record=current_sample_data_record,
                selected_sensor_to_ego_pose_matrix=lidar_frame_data_model.lidar_sensor_to_ego_pose_matrix,
                selected_sensor_frame_ego_pose_to_global_matrix=lidar_frame_data_model.lidar_frame_ego_pose_to_global_matrix,
            )
            lidar_sweep_frame_ego_pose_to_global_matrix, lidar_sweep_to_lidar_sensor_matrix = (
                lidar_sweep_transformations
            )

            lidar_sensor_to_lidar_sweep_matrix = np.linalg.inv(lidar_sweep_to_lidar_sensor_matrix)

            lidar_sweep_pointcloud_path = self.nusc.get_sample_data_path(
                current_sample_data_record["token"]
            )

            lidar_frame_data_models.append(
                LidarFrameDataModel(
                    lidar_frame_id=current_sample_data_record["token"],
                    lidar_keyframe=current_sample_data_record["is_key_frame"],
                    lidar_sensor_id=current_cs_record["token"],
                    lidar_sensor_channel_name=lidar_frame_data_model.lidar_sensor_channel_name,
                    lidar_timestamp_seconds=current_sample_data_record["timestamp"] / 1e6,
                    lidar_pointcloud_path=lidar_sweep_pointcloud_path,
                    lidar_pointcloud_source_path=None,
                    lidar_pointcloud_num_features=self.lidar_pointcloud_num_features,
                    lidar_sensor_to_ego_pose_matrix=current_lidar_sensor_to_ego_matrix,
                    lidar_frame_ego_pose_to_global_matrix=lidar_sweep_frame_ego_pose_to_global_matrix,
                    lidar_sensor_to_lidar_sweep_matrix=lidar_sensor_to_lidar_sweep_matrix,
                    lidar_pointcloud_semantic_mask_path=None,
                )
            )
        return lidar_frame_data_models

    def _extract_lidar_sources(self) -> Sequence[LidarSourceDataModel]:
        """
        Extract lidar sources metadata for this NuScenes version.

        Returns:
          Sequence[LidarSourceDataModel]: Lidar sources metadata.
        """

        if not len(self.nusc.calibrated_sensor):
            return []

        lidar_source_channel_names = []
        lidar_source_data_models = []
        for calibrated_sensor_record in self.nusc.calibrated_sensor:
            try:
                sensor_record = self.nusc.get("sensor", calibrated_sensor_record["sensor_token"])
            except KeyError:
                continue

            if sensor_record["modality"] != Modality.LIDAR:
                continue

            if sensor_record["channel"] not in lidar_source_channel_names:
                lidar_source_channel_names.append(sensor_record["channel"])
                lidar_source_data_models.append(
                    LidarSourceDataModel(
                        channel_name=sensor_record["channel"],
                        sensor_token=sensor_record["token"],
                        translation=np.asarray(
                            calibrated_sensor_record["translation"], dtype=np.float64
                        ),
                        rotation=Quaternion(calibrated_sensor_record["rotation"]).rotation_matrix,
                    )
                )

        return lidar_source_data_models

    def _extract_camera_channel_names(self, sample: Mapping[str, Any]) -> Sequence[str]:
        """
        Extract camera channel names present in a NuScenes sample.

        Args:
          sample: NuScenes sample record.

        Returns:
          Sequence[str]: Sequence of camera channel names in the sample.
        """

        camera_channel_names = []
        for channel_name, sample_data_token in sample["data"].items():
            sd_record = self.nusc.get("sample_data", sample_data_token)
            cs_record = self.nusc.get("calibrated_sensor", sd_record["calibrated_sensor_token"])
            sensor_record = self.nusc.get("sensor", cs_record["sensor_token"])
            if sensor_record["modality"] == Modality.CAMERA:
                camera_channel_names.append(channel_name)

        return camera_channel_names

    def _extract_image_frame(
        self,
        sample: Mapping[str, Any],
        camera_channel_name: str,
        lidar_sensor_to_ego_pose_matrix: npt.NDArray[np.float64],
        lidar_frame_ego_pose_to_global_matrix: npt.NDArray[np.float64],
    ) -> ImageFrameDataModel:
        """
        Extract image frame from a NuScenes sample.

        Args:
          sample: NuScenes sample record.
          camera_channel_name: Camera channel name.
          lidar_sensor_to_ego_pose_matrix: Transformation matrix from LiDAR sensor to ego pose.
          lidar_frame_ego_pose_to_global_matrix: Transformation matrix from LiDAR ego pose to global.

        Returns:
          ImageFrameDataModel: Image frame data model of the NuScenes sample.
        """

        calibrated_camera_sample_data_token = sample["data"][camera_channel_name]
        sd_record = self.nusc.get("sample_data", calibrated_camera_sample_data_token)
        cs_record = self.nusc.get("calibrated_sensor", sd_record["calibrated_sensor_token"])
        image_sensor_to_ego_matrix = convert_quaternion_to_matrix(
            rotation_quaternion=Quaternion(cs_record["rotation"]),
            translation=np.asarray(cs_record["translation"], dtype=np.float64),
            convert_to_float32=False,
        )

        ego_pose_record = self.nusc.get("ego_pose", sd_record["ego_pose_token"])
        image_frame_ego_pose_to_global_matrix = convert_quaternion_to_matrix(
            rotation_quaternion=Quaternion(ego_pose_record["rotation"]),
            translation=np.asarray(ego_pose_record["translation"], dtype=np.float64),
            convert_to_float32=False,
        )

        image_path = self.nusc.get_sample_data_path(calibrated_camera_sample_data_token)

        image_height = sd_record.get("height")
        image_width = sd_record.get("width")
        if not image_height or not image_width:
            with Image.open(image_path) as image:
                image_width, image_height = image.size

        cam2img = np.asarray(cs_record["camera_intrinsic"], dtype=np.float64)

        cam2global = image_frame_ego_pose_to_global_matrix @ image_sensor_to_ego_matrix
        global2cam = np.linalg.inv(cam2global)
        lidar2global = lidar_frame_ego_pose_to_global_matrix @ lidar_sensor_to_ego_pose_matrix
        lidar2cam = global2cam @ lidar2global

        cam2img_4x4 = np.eye(4, dtype=np.float64)
        cam2img_4x4[:3, :3] = cam2img
        lidar2img = cam2img_4x4 @ lidar2cam

        return ImageFrameDataModel(
            image_frame_id=calibrated_camera_sample_data_token,
            image_keyframe=sd_record["is_key_frame"],
            image_sensor_id=cs_record["token"],
            image_sensor_channel_name=camera_channel_name,
            image_timestamp_seconds=sd_record["timestamp"] / 1e6,
            image_path=image_path,
            image_height=image_height,
            image_width=image_width,
            cam2img=cam2img,
            image_sensor_to_ego_pose_matrix=image_sensor_to_ego_matrix,
            image_frame_ego_pose_to_global_matrix=image_frame_ego_pose_to_global_matrix,
            lidar2cam=lidar2cam,
            lidar2img=lidar2img,
        )

    def _extract_image_channel_frames(
        self, sample: Mapping[str, Any], lidar_frame_data_model: LidarFrameDataModel
    ) -> Sequence[ImageFrameDataModel]:
        """
        Extract the current-frame image metadata for all camera channels of a NuScenes sample.

        Args:
          sample: NuScenes sample record.
          lidar_frame_data_model: Lidar frame data model of the current sample, used to compute
            lidar2cam / lidar2img projections for each camera channel.

        Returns:
          Sequence[ImageFrameDataModel]: Image frame data models of all camera channels present
            in the current sample.
        """

        camera_channel_names = self._extract_camera_channel_names(sample=sample)

        return [
            self._extract_image_frame(
                sample=sample,
                camera_channel_name=camera_channel_name,
                lidar_sensor_to_ego_pose_matrix=lidar_frame_data_model.lidar_sensor_to_ego_pose_matrix,
                lidar_frame_ego_pose_to_global_matrix=lidar_frame_data_model.lidar_frame_ego_pose_to_global_matrix,
            )
            for camera_channel_name in camera_channel_names
        ]

    def _extract_image_channel_sweeps(
        self, sample: Mapping[str, Any], lidar_channel_name: str
    ) -> Sequence[Sequence[ImageFrameDataModel]]:
        """
        Extract multi-sweep image metadata (past camera keyframes, all channels) from a NuScenes
        sample, for sequence-based models. It walks the sample's `prev` chain (keyframe-to-keyframe).

        Args:
          sample: NuScenes sample record to walk backwards from.
          lidar_channel_name: Lidar channel name.

        Returns:
          Sequence[Sequence[ImageFrameDataModel]]: Image sweep data models, ordered from most
            recent to oldest. Each item is the list of image frame data models across all camera
            channels present at that sweep offset.
        """

        image_channel_sweep_data_models = []
        current_sample = sample

        for _ in range(self.max_sweeps):
            if not current_sample["prev"]:
                break

            current_sample = self.nusc.get("sample", current_sample["prev"])
            if lidar_channel_name not in current_sample["data"]:
                break

            current_lidar_sd_record = self.nusc.get(
                "sample_data", current_sample["data"][lidar_channel_name]
            )
            current_lidar_cs_record = self.nusc.get(
                "calibrated_sensor", current_lidar_sd_record["calibrated_sensor_token"]
            )
            current_lidar_sensor_to_ego_pose_matrix = convert_quaternion_to_matrix(
                rotation_quaternion=Quaternion(current_lidar_cs_record["rotation"]),
                translation=np.asarray(current_lidar_cs_record["translation"], dtype=np.float64),
                convert_to_float32=False,
            )
            current_lidar_ego_pose_record = self.nusc.get(
                "ego_pose", current_lidar_sd_record["ego_pose_token"]
            )
            current_lidar_frame_ego_pose_to_global_matrix = convert_quaternion_to_matrix(
                rotation_quaternion=Quaternion(current_lidar_ego_pose_record["rotation"]),
                translation=np.asarray(
                    current_lidar_ego_pose_record["translation"], dtype=np.float64
                ),
                convert_to_float32=False,
            )

            camera_channel_names = self._extract_camera_channel_names(sample=current_sample)

            image_channel_sweep_data_models.append(
                [
                    self._extract_image_frame(
                        sample=current_sample,
                        camera_channel_name=camera_channel_name,
                        lidar_sensor_to_ego_pose_matrix=current_lidar_sensor_to_ego_pose_matrix,
                        lidar_frame_ego_pose_to_global_matrix=current_lidar_frame_ego_pose_to_global_matrix,
                    )
                    for camera_channel_name in camera_channel_names
                ]
            )
        return image_channel_sweep_data_models

    def _extract_category_mapping(self) -> CategoryMappingDataModel:
        """
        Extract category metadata for this NuScenes version.

        Returns:
          CategoryMappingDataModel: Category metadata.
        """

        category_records = self.nusc.category
        if not len(category_records):
            return CategoryMappingDataModel(category_names=[], category_indices=[])

        category_names = []
        category_indices = []
        for category_record in category_records:
            category_names.append(category_record["name"])
            category_indices.append(category_record["index"])

        return CategoryMappingDataModel(
            category_names=category_names,
            category_indices=category_indices,
        )

    def extract_nuscenes_sample_record(
        self, sample: Mapping[str, Any], sample_index: int
    ) -> T4SampleRecord | None:
        """
        Extract a T4SampleRecord (unified intermediate sample container) from a NuScenes sample.

        Args:
          sample: NuScenes sample record.
          sample_index: Sample index.

        Returns:
          T4SampleRecord | None: T4SampleRecord, or None if no supported lidar channel was found.
        """

        if LidarChannel.LIDAR_TOP in sample["data"]:
            lidar_channel_name = LidarChannel.LIDAR_TOP
        elif LidarChannel.LIDAR_CONCAT in sample["data"]:
            lidar_channel_name = LidarChannel.LIDAR_CONCAT
        else:
            return None

        frame_basic_metadata = self._extract_sample_basic_metadata(
            sample=sample, sample_index=sample_index
        )

        lidar_frame_data_model, box3d = self._extract_lidar_frame(
            sample=sample, lidar_channel_name=lidar_channel_name
        )

        boxes_3d_data_model = self._extract_boxes_3d_annotations(
            sample=sample,
            boxes_3d=box3d,
            lidar_sensor_to_ego_pose_matrix=lidar_frame_data_model.lidar_sensor_to_ego_pose_matrix,
            lidar_frame_ego_pose_to_global_matrix=lidar_frame_data_model.lidar_frame_ego_pose_to_global_matrix,
        )

        lidar_sweep_data_models = self._extract_lidar_sweeps(
            lidar_frame_data_model=lidar_frame_data_model
        )

        lidar_frame_data_models = [lidar_frame_data_model] + lidar_sweep_data_models

        lidar_source_data_models = self._extract_lidar_sources()

        image_channel_frame_data_models = self._extract_image_channel_frames(
            sample=sample, lidar_frame_data_model=lidar_frame_data_model
        )
        image_channel_sweep_data_models = self._extract_image_channel_sweeps(
            sample=sample, lidar_channel_name=lidar_channel_name
        )
        image_frame_data_models = [
            image_channel_frame_data_models
        ] + image_channel_sweep_data_models

        category_mapping_data_model = self._extract_category_mapping()

        return T4SampleRecord(
            frame_basic_metadata=frame_basic_metadata,
            lidar_frame_data_models=lidar_frame_data_models,
            lidar_source_data_models=lidar_source_data_models,
            image_frame_data_models=image_frame_data_models,
            category_mapping_data_model=category_mapping_data_model,
            boxes_3d_data_model=boxes_3d_data_model,
        )
