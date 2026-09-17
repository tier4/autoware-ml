import json
import logging
from types import MappingProxyType
from typing import Sequence

from jaxtyping import Float32
import polars as pl
import numpy as np
import torch

from autoware_ml.databases.schemas.image_frames import (
    ImageFrameDataModel,
    ImageFrameDatasetSchema,
)
from autoware_ml.databases.schemas.lidar_frames import (
    LidarFrameDataModel,
    LidarFrameDatasetSchema,
    relative_to_database_root,
)
from autoware_ml.databases.schemas.dataset_schemas import DatasetTableSchema
from autoware_ml.databases.schemas.lidar_sources import (
    LidarSourceDataModel,
    LidarSourceDatasetSchema,
)
from autoware_ml.dataclasses.batch.sample_batch import ModelGTSample
from autoware_ml.dataclasses.geometry.images import ImageSample
from autoware_ml.dataclasses.geometry.point_clouds import LiDARPointCloudSample, LidarSourceView
from autoware_ml.datamodule.base_dataset import (
    BaseDataset,
)
from autoware_ml.datamodule.base_dataset_task import BaseDatasetTask
from autoware_ml.transforms.base import TransformsCompose
from autoware_ml.types.tasks import TaskType


logger = logging.getLogger(__name__)


class T4Dataset(BaseDataset):
    """Dataset serving every task from the record table of a T4 database."""

    def __init__(
        self,
        database_root_path: str,
        max_num_3d_gt_bboxes: int,
        dataset_records_dataframe: pl.DataFrame,
        transforms: TransformsCompose | None,
        dataset_tasks: MappingProxyType[TaskType | str, BaseDatasetTask],
        lidar_intensity_scale: float,
        lidar_sources: Sequence[str] | None = None,
        camera_names: Sequence[str] | None = None,
        camera_sources: Sequence[str] | None = None,
    ) -> None:
        """
        Initialize the T4Dataset class.

        Args:
          database_root_path: Root directory of the database.
          max_num_3d_gt_bboxes: Maximum number of 3D ground truth bounding boxes in the dataset.
            This is allowed to be 0 if the dataset does not contain any 3D ground truth
            bounding boxes or it does not need to run 3D detection tasks.
          dataset_records_dataframe: Records of the corpus.
          transforms: Global transforms to be applied to the dataset records.
          dataset_tasks: Every task dataset that is part of the multi-task dataset, mapped by
            task type.
          lidar_intensity_scale: Intensity of the strongest return in the point clouds of the
            database, every loaded intensity is divided by it.
          lidar_sources: Channel names of the lidars to serve one by one out of the merged
            cloud of every record, each one in its own sensor frame. None serves the merged
            cloud. Every record must then carry its lidar sources and the index file of its
            merged cloud.
          camera_names: Channel names of the cameras to serve, in the order the model takes
            them. The records missing one of them at the sample time are left out. None
            serves every camera of a record in the order the record lists them.
          camera_sources: Channel names of the cameras to serve one by one, each camera of a
            record as its own sample. The records missing one of them at the sample time are
            left out. It replaces camera_names and cannot be combined with lidar_sources.
        """
        if lidar_sources is not None:
            if len(lidar_sources) == 0 or len(set(lidar_sources)) != len(lidar_sources):
                raise ValueError(
                    f"lidar_sources must name distinct lidars, got {list(lidar_sources)}."
                )
            if TaskType.DETECTION3D in {TaskType(key) for key in dataset_tasks}:
                raise ValueError(
                    "Boxes are annotated in the frame of the merged cloud, so 3D detection "
                    "cannot run on single lidar sources."
                )
        if camera_sources is not None and camera_names is not None:
            raise ValueError("Set camera_names or camera_sources, not both.")
        if camera_sources is not None and lidar_sources is not None:
            raise ValueError("lidar_sources and camera_sources cannot be combined.")
        served_cameras = camera_names if camera_sources is None else camera_sources
        if served_cameras is not None:
            if len(served_cameras) == 0 or len(set(served_cameras)) != len(served_cameras):
                raise ValueError(
                    f"The served cameras must be distinct, got {list(served_cameras)}."
                )
            dataset_records_dataframe = select_records_with_cameras(
                dataset_records_dataframe, served_cameras
            )
        super().__init__(
            database_root_path=database_root_path,
            max_num_3d_gt_bboxes=max_num_3d_gt_bboxes,
            dataset_records_dataframe=dataset_records_dataframe,
            transforms=transforms,
        )
        self.lidar_intensity_scale = lidar_intensity_scale
        self.lidar_sources = None if lidar_sources is None else tuple(lidar_sources)
        self.camera_names = None if camera_names is None else tuple(camera_names)
        self.camera_sources = None if camera_sources is None else tuple(camera_sources)

        # Convert the dataset_tasks to TaskType: BaseDatasetTask mapping if the keys are strings
        self.dataset_tasks: MappingProxyType[TaskType, BaseDatasetTask] = MappingProxyType(
            {
                TaskType(key) if isinstance(key, str) else key: value
                for key, value in dataset_tasks.items()
            }
        )
        logger.info(
            f"Initialized T4Dataset with {len(self.dataset_tasks)} "
            f"task datasets: {list(self.dataset_tasks.keys())} "
            f"transforms: {self.transforms} and max_num_3d_gt_bboxes: {self.max_num_3d_gt_bboxes}"
        )

    @property
    def samples_per_record(self) -> int:
        """Number of samples every record serves, one per lidar or camera source."""
        if self.lidar_sources is not None:
            return len(self.lidar_sources)
        if self.camera_sources is not None:
            return len(self.camera_sources)
        return 1

    def __len__(self) -> int:
        """Return the number of samples, one per record and served source.

        Returns:
          int: Number of samples.
        """
        return super().__len__() * self.samples_per_record

    def get_data_sample(self, index: int) -> ModelGTSample:
        """
        Read the sample behind one index, with the annotations of every task.

        Args:
          index: Index of the sample to be processed. With lidar or camera sources every
            record serves one sample per source, in the order the sources are declared.

        Returns:
          ModelGTSample: The sample before the transform pipeline.
        """
        record_index = self.record_index(index)
        source_index = index % self.samples_per_record
        source_view = (
            None
            if self.lidar_sources is None
            else self.get_lidar_source_view(record_index, self.lidar_sources[source_index])
        )

        data_samples = {}
        for task_type, dataset_task in self.dataset_tasks.items():
            data_samples[task_type] = dataset_task.get_data_sample(record_index)

        # Retrieve general data row for the given index from the dataset records dataframe
        lidar_pointcloud_samples = self.get_lidar_pointcloud_data_samples(record_index, source_view)

        # Retrieve the detection3d_gt_bboxes_3d and segmentation3d_gt_sample from the data_samples dictionary
        detection3d_gt_sample: ModelGTSample | None = data_samples.get(TaskType.DETECTION3D, None)
        if detection3d_gt_sample is not None:
            detection3d_gt_bboxes_3d = detection3d_gt_sample.detection3d_gt_bboxes_3d
        else:
            detection3d_gt_bboxes_3d = None

        segmentation3d_multi_task_gt_sample: ModelGTSample | None = data_samples.get(
            TaskType.SEGMENTATION3D, None
        )
        if segmentation3d_multi_task_gt_sample is not None:
            segmentation3d_gt_sample = segmentation3d_multi_task_gt_sample.segmentation3d_gt_sample
        else:
            segmentation3d_gt_sample = None
        # The mask of a merged cloud labels the points of every source in file order
        if segmentation3d_gt_sample is not None and source_view is not None:
            end = source_view.point_index_begin + source_view.num_points
            segmentation3d_gt_sample = segmentation3d_gt_sample._replace(
                gt_semantic_mask=segmentation3d_gt_sample.gt_semantic_mask[
                    source_view.point_index_begin : end
                ]
            )

        # The camera calibrations refer to the merged cloud, so a single source serves no image
        return ModelGTSample(
            lidar_point_cloud_samples=lidar_pointcloud_samples,
            image_samples=(
                self.get_image_data_samples(record_index, source_index)
                if source_view is None
                else None
            ),
            point_cloud_data=None,  # point cloud data will be populated in the transform pipeline
            camera_image_data=None,
            detection3d_gt_bboxes_3d=detection3d_gt_bboxes_3d,
            segmentation3d_gt_sample=segmentation3d_gt_sample,
        )

    def get_lidar_source_view(self, idx: int, channel_name: str) -> LidarSourceView:
        """
        Locate one lidar source inside the merged cloud of a record.

        The record lists the calibration of every lidar source, and the index file of its
        merged cloud names the points each source contributes.

        Args:
          idx: Index of the record.
          channel_name: Channel name of the lidar source.

        Returns:
          LidarSourceView: Points and mounting of the source.
        """
        lidar_sources = self.dataset_records_dataframe.item(
            idx, DatasetTableSchema.LIDAR_SOURCES.name
        )
        if lidar_sources is None:
            raise ValueError(f"Record {idx} lists no lidar sources to serve {channel_name} from.")
        by_channel = {
            source[LidarSourceDatasetSchema.channel_name.name]: source for source in lidar_sources
        }
        if channel_name not in by_channel:
            raise ValueError(
                f"Record {idx} has no lidar source {channel_name}, it has {sorted(by_channel)}."
            )
        lidar_source = LidarSourceDataModel.load_from_dictionary(by_channel[channel_name])

        lidar_frame = LidarFrameDataModel.load_from_dictionary(
            self.dataset_records_dataframe.item(idx, DatasetTableSchema.LIDAR_FRAMES.name)[0]
        )
        source_index_path = lidar_frame.lidar_pointcloud_source_relative_path
        if source_index_path is None:
            raise ValueError(
                f"The lidar frame {lidar_frame.lidar_frame_id} of record {idx} has no index file "
                "naming the points of each lidar source."
            )
        with open(self.database_root_path / source_index_path) as index_file:
            source_index = {
                entry["sensor_token"]: entry for entry in json.load(index_file)["sources"]
            }
        if lidar_source.sensor_token not in source_index:
            raise ValueError(
                f"The index file of lidar frame {lidar_frame.lidar_frame_id} has no points of "
                f"lidar source {channel_name}."
            )
        entry = source_index[lidar_source.sensor_token]

        sensor_to_frame = np.eye(4, dtype=np.float32)
        sensor_to_frame[:3, :3] = lidar_source.rotation_fp32
        sensor_to_frame[:3, 3] = lidar_source.translation_fp32
        return LidarSourceView(
            point_index_begin=int(entry["idx_begin"]),
            num_points=int(entry["length"]),
            sensor_to_frame_matrix=torch.from_numpy(sensor_to_frame),
        )

    def get_image_data_samples(
        self, idx: int, source_index: int = 0
    ) -> Sequence[ImageSample] | None:
        """
        Retrieve the image of every served camera at the sample time of the record.

        Args:
          idx: Index of the specific record to be processed.
          source_index: Camera source to serve when the cameras are served one by one, unused
            otherwise.

        Returns:
          Sequence[ImageSample] | None: One sample per served camera, None when the record
            holds no camera frame at the sample time.
        """
        image_frames = self.dataset_records_dataframe.item(
            idx, DatasetTableSchema.IMAGE_FRAMES.name
        )
        # The frames of every camera at the sample time lead the record, the camera sweeps
        # follow them. On a corpus where every lidar frame is a record, a camera can have no
        # frame at the lidar timestamp and is then missing from the first list.
        sample_frames = image_frames[0] if len(image_frames) > 0 else []
        frames_by_camera = {
            frame[ImageFrameDatasetSchema.image_sensor_channel_name.name]: frame
            for frame in sample_frames
        }
        if self.camera_sources is not None:
            camera_names = (self.camera_sources[source_index],)
        elif self.camera_names is not None:
            camera_names = self.camera_names
        else:
            camera_names = tuple(frames_by_camera)
        if len(camera_names) == 0:
            return None

        image_samples = []
        for camera_name in camera_names:
            image_frame = ImageFrameDataModel.load_from_dictionary(frames_by_camera[camera_name])
            if image_frame.lidar2cam_fp32 is None or image_frame.lidar2img_fp32 is None:
                raise ValueError(
                    f"The image frame {image_frame.image_frame_id} of record {idx} carries no "
                    "lidar to camera calibration, so it cannot be projected."
                )
            image_samples.append(
                ImageSample(
                    image_path=self._update_frame_path(image_frame.image_path),
                    camera_name=image_frame.image_sensor_channel_name,
                    timestamp=image_frame.image_timestamp_seconds,
                    camera_intrinsic=torch.tensor(image_frame.cam2img_fp32, dtype=torch.float32),
                    lidar2cam=torch.tensor(image_frame.lidar2cam_fp32, dtype=torch.float32),
                    lidar2image=torch.tensor(image_frame.lidar2img_fp32, dtype=torch.float32),
                    distortion_model=image_frame.image_distortion_model,
                    distortion_coefficients=torch.tensor(
                        image_frame.image_distortion_coefficients, dtype=torch.float32
                    ),
                )
            )
        return image_samples

    def _update_frame_path(self, frame_path: str) -> str:
        """
        Resolve a stored frame path against the database root.

        Args:
          frame_path: Frame path as the record table stores it.

        Returns:
          str: Path of the frame under the database root.
        """
        return str(self.database_root_path / relative_to_database_root(frame_path, "frame path"))

    def get_lidar_pointcloud_data_samples(
        self, idx: int, source_view: LidarSourceView | None
    ) -> Sequence[LiDARPointCloudSample]:
        """
        Retrieve the lidar point cloud data row for the given index.

        Args:
          idx: Index of the specific record to be processed.
          source_view: Lidar source to serve out of the current frame, None for the merged
            cloud. A source is served from the current frame alone, since the stored sweeps
            merge every source.
        """
        lidar_pointcloud_metadata_samples = self.dataset_records_dataframe.item(
            idx, DatasetTableSchema.LIDAR_FRAMES.name
        )
        if source_view is not None:
            lidar_pointcloud_metadata_samples = lidar_pointcloud_metadata_samples[:1]
        lidar_pointcloud_samples = []
        for lidar_pointcloud_metadata in lidar_pointcloud_metadata_samples:
            lidar_sensor_to_ego_pose_matrix: Float32[np.ndarray, "4 4"] = lidar_pointcloud_metadata[
                LidarFrameDatasetSchema.lidar_sensor_to_ego_pose_matrix.name
            ]
            lidar_to_ego_pose_to_global_matrix: Float32[np.ndarray, "4 4"] = (
                lidar_pointcloud_metadata[
                    LidarFrameDatasetSchema.lidar_frame_ego_pose_to_global_matrix.name
                ]
            )
            lidar_sensor_to_lidar_sweep_matrix: Float32[np.ndarray, "4 4"] = (
                lidar_pointcloud_metadata[
                    LidarFrameDatasetSchema.lidar_sensor_to_lidar_sweep_matrix.name
                ]
            )
            lidar_pointcloud_path = self._update_frame_path(
                lidar_pointcloud_metadata[LidarFrameDatasetSchema.lidar_pointcloud_path.name]
            )
            if source_view is not None:
                lidar_sensor_to_ego_pose_matrix = (
                    lidar_sensor_to_ego_pose_matrix @ source_view.sensor_to_frame_matrix.numpy()
                )

            lidar_pointcloud_samples.append(
                LiDARPointCloudSample(
                    point_cloud_path=lidar_pointcloud_path,
                    timestamp=lidar_pointcloud_metadata[
                        LidarFrameDatasetSchema.lidar_timestamp_seconds.name
                    ],
                    intensity_scale=self.lidar_intensity_scale,
                    sensor_to_ego_pose_matrix=torch.tensor(
                        lidar_sensor_to_ego_pose_matrix,
                        dtype=torch.float32,
                    ),
                    lidar_to_ego_pose_to_global_matrix=torch.tensor(
                        lidar_to_ego_pose_to_global_matrix,
                        dtype=torch.float32,
                    ),
                    lidar_sensor_to_lidar_sweep_matrix=torch.tensor(
                        lidar_sensor_to_lidar_sweep_matrix,
                        dtype=torch.float32,
                    ),
                    source_view=source_view,
                )
            )
        return lidar_pointcloud_samples

    def assign_dataset_records(self, dataset_records_dataframe: pl.DataFrame) -> None:
        """
        Recursively assign the dataset records dataframe to each task dataset as well and
        perform their pre_filtering .

        Args:
            dataset_records_dataframe: Polars DataFrame of dataset records.
        """
        self.dataset_records_dataframe = dataset_records_dataframe
        for dataset_task in self.dataset_tasks.values():
            filtered_dataset_records_dataframe = dataset_task.pre_filter_dataset_records(
                dataset_records_dataframe
            )
            dataset_task.dataset_records_dataframe = filtered_dataset_records_dataframe


def select_records_with_cameras(
    dataset_records_dataframe: pl.DataFrame, camera_names: Sequence[str]
) -> pl.DataFrame:
    """
    Keep the records that hold a frame of every given camera at the sample time.

    Args:
      dataset_records_dataframe: Records of a corpus.
      camera_names: Channel names of the cameras every kept record must hold.

    Returns:
      pl.DataFrame: The records holding every camera.
    """
    sample_camera_names = (
        pl.col(DatasetTableSchema.IMAGE_FRAMES.name)
        .list.first()
        .list.eval(
            pl.element().struct.field(ImageFrameDatasetSchema.image_sensor_channel_name.name)
        )
    )
    selected_records = dataset_records_dataframe.filter(
        sample_camera_names.list.set_intersection(list(camera_names)).list.len()
        == len(camera_names)
    )
    if selected_records.height == 0:
        raise ValueError(f"No record holds a frame of every camera of {list(camera_names)}.")
    logger.info(
        f"Kept {selected_records.height} of {dataset_records_dataframe.height} records holding "
        f"every camera of {list(camera_names)}."
    )
    return selected_records
