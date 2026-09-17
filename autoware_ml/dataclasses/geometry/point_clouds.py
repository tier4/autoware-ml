from __future__ import annotations

from typing import Sequence, NamedTuple

from jaxtyping import Float32, Int32
import torch
from torch import Tensor

from autoware_ml.geometry.points.base_points import BasePoints


class PointCloudGTBatch(NamedTuple):
    """Named tuple to represent pointcloud features in a batch size with their batch indices."""

    points: Float32[
        Tensor, "batch_size*num_points num_features"
    ]  # (B*P, number of features for each point)
    batch_indices: Int32[Tensor, " batch_size*num_points"]  # (B*P, ), batch indices for each point

    @staticmethod
    def collate_gt_samples(
        point_gt_samples: Sequence[BasePoints],
    ) -> PointCloudGTBatch | None:
        """
        Collate a sequence of points (BasePoints) into a single PointCloudGTBatch.

        Args:
          point_gt_samples: Sequence of points (BasePoints) to be collated.

        Returns:
          PointCloudGTBatch: Collated point cloud GT batch.
        """
        if len(point_gt_samples) == 0:
            return None

        # Concatenate all points from the sequence of point_gt_samples
        points = torch.cat([sample.points for sample in point_gt_samples], dim=0)

        # Convert it to (0, 0, 0, 1, 1, 1, 2, 2, 2, ...) for each point in the batch
        batch_indices = torch.cat(
            [
                torch.full(
                    (point.points.shape[0],), i, dtype=torch.int32, device=point.points.device
                )
                for i, point in enumerate(point_gt_samples)
            ],
            dim=0,
        )

        if points.shape[0] != batch_indices.shape[0]:
            raise ValueError(
                "Mismatch between number of points and batch indices. "
                f"Points shape: {points.shape}, Batch indices shape: {batch_indices.shape}"
            )

        return PointCloudGTBatch(
            points=points,
            batch_indices=batch_indices,
        )

    def to_device(self, device: torch.device) -> PointCloudGTBatch:
        """
        Move the PointCloudGTBatch to the specified device.

        Args:
          device: The target device to move the batch to.

        Returns:
          PointCloudGTBatch: The batch moved to the specified device.
        """
        return PointCloudGTBatch(
            points=self.points.to(device),
            batch_indices=self.batch_indices.to(device),
        )


class LiDARPointCloudSample(NamedTuple):
    """
    Named tuple to represent a single row of LiDAR point cloud data,
    which contains the dataset record for the LiDAR point cloud task.
    """

    point_cloud_path: str
    timestamp: float
    # Transformation matrix from LiDAR sensor frame to ego pose of this LiDAR sensor frame
    sensor_to_ego_pose_matrix: Float32[Tensor, "4 4"]  # (4, 4)
    # Transformation matrix from ego pose of this LiDAR sensor frame to global frame
    lidar_to_ego_pose_to_global_matrix: Float32[Tensor, "4 4"]  # (4, 4)
    # Transformation matrix from the main lidar sensor to other lidar sweeps at this frame
    lidar_sensor_to_lidar_sweep_matrix: Float32[Tensor, "4 4"]  # (4, 4)
