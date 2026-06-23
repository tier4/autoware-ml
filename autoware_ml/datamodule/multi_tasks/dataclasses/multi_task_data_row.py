from typing import Sequence, NamedTuple

import numpy.typing as npt
import numpy as np

from autoware_ml.datamodule.multi_tasks.dataclasses.detection3d import Detection3DDataRow
from autoware_ml.datamodule.multi_tasks.dataclasses.segmentation3d import Segmentation3DDataRow


class LiDARPointCloudDataRow(NamedTuple):
    """
    Named tuple to represent a single row of LiDAR point cloud data,
    which contains the dataset record for the LiDAR point cloud task.
    """

    point_cloud_path: str
    # Transformation matrix from LiDAR sensor frame to ego pose of this LiDAR sensor frame
    sensor_to_ego_pose_matrix: npt.ArrayLike[np.float32]  # (4, 4)
    # Transformation matrix from ego pose of this LiDAR sensor frame to global frame
    lidar_to_ego_pose_to_global_matrix: npt.ArrayLike[np.float32]  # (4, 4)
    # Transformation matrix from the main lidar sensor to other lidar sweeps at this frame
    lidar_sensor_to_lidar_sweep_matrix: npt.ArrayLike[np.float32]  # (4, 4)


class MultiTaskDataRow(NamedTuple):
    """
    Named tuple to represent a single row of multi-task data,
    which contains the dataset records for each task.
    """

    # Can be multi-sweep LiDAR point cloud data, which is a list of LiDAR point cloud data rows for each sweep.
    lidar_point_cloud_data_row: Sequence[LiDARPointCloudDataRow] | None

    # (number of point clouds, number of features for each point), can be None
    # if it doesn't need to be loaded
    point_cloud_features: npt.ArrayLike[np.float32] | None

    detection3d_data_row: Detection3DDataRow | None
    segmentation3d_data_row: Segmentation3DDataRow | None
