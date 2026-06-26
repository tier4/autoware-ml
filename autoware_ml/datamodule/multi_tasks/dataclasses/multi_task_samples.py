from typing import Sequence, NamedTuple

import numpy.typing as npt
import numpy as np

from autoware_ml.datamodule.multi_tasks.dataclasses.detection3d import (
    Detection3DGTSample,
    Detection3DGTBatch,
)
from autoware_ml.datamodule.multi_tasks.dataclasses.segmentation3d import Segmentation3DGTSample


class LiDARPointCloudSample(NamedTuple):
    """
    Named tuple to represent a single row of LiDAR point cloud data,
    which contains the dataset record for the LiDAR point cloud task.
    """

    point_cloud_path: str
    timestamp_seconds: float
    # Transformation matrix from LiDAR sensor frame to ego pose of this LiDAR sensor frame
    sensor_to_ego_pose_matrix: npt.NDArray[np.float32]  # (4, 4)
    # Transformation matrix from ego pose of this LiDAR sensor frame to global frame
    lidar_to_ego_pose_to_global_matrix: npt.NDArray[np.float32]  # (4, 4)
    # Transformation matrix from the main lidar sensor to other lidar sweeps at this frame
    lidar_sensor_to_lidar_sweep_matrix: npt.NDArray[np.float32]  # (4, 4)


class MultiTaskGTSample(NamedTuple):
    """
    Named tuple to represent a single row/sample of multi-task data when inputting to the
    multi-task model.
    """

    # Can be multi-sweep LiDAR point cloud data, which is a list of LiDAR point cloud data rows for each sweep.
    lidar_point_cloud_samples: Sequence[LiDARPointCloudSample] | None

    # (number of point clouds, number of features for each point), can be None
    # if it doesn't need to be loaded
    point_cloud_features: npt.NDArray[np.float32] | None

    detection3d_gt_sample: Detection3DGTSample | None
    segmentation3d_gt_sample: Segmentation3DGTSample | None


class MultiTaskGTBatch(NamedTuple):
    """
    Named tuple to represent a batch of multi-task data after collating from sequence of
    MultiTaskGTSample when inputting to the multi-task model.
    """

    # 3D branch
    point_cloud_features: npt.NDArray[np.float32] | None  # (B*P, pointcloud feature dimension)
    # Batch indices for each point cloud feature, shape (B*P, ), where B is the batch size and
    # P is the number of points in the point cloud.
    point_cloud_features_batch_indices: npt.NDArray[np.int32] | None  # (B*P, )

    detection3d_gt_batch: Detection3DGTBatch | None

    # TODO (Kok Seang): 3D segmentation
