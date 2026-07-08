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


"""
Point-cloud loading transforms to support MultiTaskGTSample.
The code is modified from mmdetection3d.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch

from autoware_ml.geometry.points.base_points import BasePoints
from autoware_ml.geometry.points.lidar_points import LiDARPoints
from autoware_ml.transforms.multi_task.base import MultiTaskBaseTransform
from autoware_ml.datamodule.multi_task.dataclasses.multi_task_samples import (
    MultiTaskGTSample,
    LiDARPointCloudSample,
)
from autoware_ml.types.geometry import PointFeatureName, PointFieldIndex


class LoadPointsFromFile(MultiTaskBaseTransform):
    """Load point clouds from a lidar file path stored in sample metadata."""

    _required_keys = ["lidar_point_cloud_samples"]

    def __init__(
        self,
        load_dim: int = 5,
        use_dim: Sequence[int] | int = (0, 1, 2, 3),
        bev_remove_radius: float = 0.0,
    ) -> None:
        """Initialize the point-cloud loader.

        Args:
            load_dim: Number of features stored per point in the source file.
            use_dim: Selected feature dimensions preserved in the loaded tensor.
            bev_remove_radius: Radius (x and y) within which points will be removed (e.g., to remove ego vehicle
                points). Set to 0.0 to disable point removal.
        """
        super().__init__(probability=None)
        self.load_dim = load_dim
        self.use_dim = use_dim
        self.bev_remove_radius = bev_remove_radius

    def remove_close(self, points_data: BasePoints) -> BasePoints:
        """Remove point too close within a certain radius from origin.

        Args:
            points_data (BasePoints): Sweep points.

        Returns:
            BasePoints: Points after removing.
        """
        if self.bev_remove_radius <= 0:
            return points_data
        x_filt = torch.abs(points_data.points[:, PointFieldIndex.X]) < self.bev_remove_radius
        y_filt = torch.abs(points_data.points[:, PointFieldIndex.Y]) < self.bev_remove_radius
        not_close = ~(x_filt & y_filt)
        points_data.remove_points(not_close)
        return points_data

    def load_points_from_samples(
        self, index: int, lidar_point_cloud_samples: Sequence[LiDARPointCloudSample]
    ) -> BasePoints:
        """Load point cloud data from a binary file.

        Args:
            file_path: Path to the binary file containing point cloud data.
        """
        if index >= len(lidar_point_cloud_samples):
            raise IndexError(
                f"Index {index} is out of bounds for lidar_point_cloud_samples with length {len(lidar_point_cloud_samples)}."
            )

        current_lidar_point_path = lidar_point_cloud_samples[index].point_cloud_path
        points_np = np.fromfile(current_lidar_point_path, dtype=np.float32).reshape(
            -1, self.load_dim
        )

        if isinstance(self.use_dim, int):
            use_dims = list(range(self.use_dim))
        else:
            use_dims = list(self.use_dim)

        points_np = points_np[:, use_dims]
        point_feature_names = [PointFeatureName(PointFieldIndex(i).name.lower()) for i in use_dims]
        timestamp_seconds = lidar_point_cloud_samples[index].timestamp_seconds
        return LiDARPoints.from_numpy(
            points_np=points_np,
            point_feature_names=point_feature_names,
            timestamp_seconds=timestamp_seconds,
        )

    def transform(self, multi_task_gt_sample: MultiTaskGTSample) -> MultiTaskGTSample:
        """Load point data from the current sample at the current sweep.

        Args:
            multi_task_gt_sample: MultiTaskGTSample instance containing `lidar_point_cloud_samples`.

        Returns:
            Updated MultiTaskGTSample instance with a loaded `point_cloud_features` array.
        """
        # Load the first index of the point cloud file, and reshape it to (N, load_dim)
        if not multi_task_gt_sample.lidar_point_cloud_samples:
            raise ValueError("No lidar point cloud samples found in the MultiTaskGTSample.")

        # Always select 0 for the point cloud at the current frame.
        lidar_points = self.load_points_from_samples(
            0, multi_task_gt_sample.lidar_point_cloud_samples
        )

        return MultiTaskGTSample(
            lidar_point_cloud_samples=multi_task_gt_sample.lidar_point_cloud_samples,
            detection3d_gt_bboxes_3d=multi_task_gt_sample.detection3d_gt_bboxes_3d,
            point_cloud_data=lidar_points,
            segmentation3d_gt_sample=multi_task_gt_sample.segmentation3d_gt_sample,
        )


class LoadMultiSweepPointsFromFile(LoadPointsFromFile):
    """Load multi-sweep point clouds from lidar file paths stored in sample metadata."""

    _required_keys = ["lidar_point_cloud_samples"]

    def __init__(
        self,
        sweeps_num: int,
        test_mode: bool,
        use_timestamp_difference: bool = True,
        load_dim: int = 5,
        use_dim: Sequence[int] | int = (0, 1, 2, 3),
        bev_remove_radius: float = 1.0,
    ) -> None:
        """Initialize the multi-sweep point-cloud loader.

        Args:
            sweeps_num: Number of sweeps to concatenate for each sample. If the number of available
              sweeps is less than sweeps_num, it will take the maximum available sweeps.
            test_mode: Whether the loader is in test mode. If True, it will always load the
              first sweeps_num sweeps. If False, it will randomly select sweeps_num sweeps.
            use_timestamp_difference: Whether to add a timestamp difference feature to each point.
              If True, it will add a feature representing the time difference between the main lidar
              frame and the sweep frame for each point.
            load_dim: Number of features stored per point in the source file.
            use_dim: Selected feature dimensions preserved in the loaded tensor.
            bev_remove_radius: Radius (x and y) within which points will be removed (e.g., to remove ego vehicle
                points). Set to 0.0 to disable point removal.
        """
        super().__init__(load_dim=load_dim, use_dim=use_dim)
        self.sweeps_num = sweeps_num
        self.test_mode = test_mode
        self.bev_remove_radius = bev_remove_radius
        self.use_timestamp_difference = use_timestamp_difference

    def transform(self, multi_task_gt_sample: MultiTaskGTSample) -> MultiTaskGTSample:
        """Load multi-sweep point data from the current sample.

        Args:
            multi_task_gt_sample: MultiTaskGTSample instance containing `lidar_point_cloud_samples`.

        Returns:
            Updated MultiTaskGTSample instance with a loaded `point_cloud_features` array.
        """
        if not multi_task_gt_sample.lidar_point_cloud_samples:
            raise ValueError("No lidar point cloud samples found in the MultiTaskGTSample.")

        if multi_task_gt_sample.point_cloud_data is None:
            raise ValueError("Point cloud data is not available in the MultiTaskGTSample.")

        current_frame_point_cloud_data = multi_task_gt_sample.point_cloud_data
        available_sweeps_nums = min(
            len(multi_task_gt_sample.lidar_point_cloud_samples) - 1, self.sweeps_num
        )
        if self.test_mode:
            sweep_indices = list(range(1, available_sweeps_nums + 1))
        else:
            sweep_indices = torch.arange(1, len(multi_task_gt_sample.lidar_point_cloud_samples))
            sweep_indices = torch.randperm(len(sweep_indices))[:available_sweeps_nums].tolist()

        # Create timestamp_feature for each point
        main_lidar_frame_timestamp = current_frame_point_cloud_data.timestamp_seconds

        # Add timestamp difference feature to the current frame pointcloud points
        if self.use_timestamp_difference:
            current_frame_point_cloud_data.add_timestamp_difference(0.0)

        concat_points = [current_frame_point_cloud_data]
        for sweep_idx in sweep_indices:
            sweep_points = self.load_points_from_samples(
                sweep_idx, multi_task_gt_sample.lidar_point_cloud_samples
            )
            # Remove points too close within a certain radius from origin if remove_bev_radius is set
            sweep_points = self.remove_close(sweep_points)

            # Get the lidar sweep point cloud sample data
            sweep_lidar_sample = multi_task_gt_sample.lidar_point_cloud_samples[sweep_idx]
            if self.use_timestamp_difference:
                timestamp_difference = (
                    main_lidar_frame_timestamp - sweep_lidar_sample.timestamp_seconds
                )
                sweep_points.add_timestamp_difference(timestamp_difference)

            # Transform from the last lidar sweep frame to the current lidar frame using the provided transformation matrices
            translation_vector = sweep_lidar_sample.lidar_sensor_to_lidar_sweep_matrix[:3, 3]
            rotation_matrix = sweep_lidar_sample.lidar_sensor_to_lidar_sweep_matrix[:3, :3]

            # Transformation
            # Check https://github.com/open-mmlab/mmdetection3d/issues/3054
            # Subtract first, sensor to lidar
            sweep_points.translate(-translation_vector)
            # Rotate: P @ R (Lidar_to_sweep) equivalent to R^T (sweep_to_lidar) @ P
            sweep_points.rotate(rotation_matrix)
            concat_points.append(sweep_points)

        # Concatenate all points from the current frame and the selected sweeps
        multi_sweep_points = LiDARPoints.concat(concat_points)

        return MultiTaskGTSample(
            lidar_point_cloud_samples=multi_task_gt_sample.lidar_point_cloud_samples,
            detection3d_gt_bboxes_3d=multi_task_gt_sample.detection3d_gt_bboxes_3d,
            point_cloud_data=multi_sweep_points,
            segmentation3d_gt_sample=multi_task_gt_sample.segmentation3d_gt_sample,
        )
