from __future__ import annotations

from typing import Sequence, NamedTuple

from jaxtyping import Float32, Int32, Int64
import torch
from torch import Tensor

from autoware_ml.geometry.points.base_points import BasePoints
from autoware_ml.databases.t4pack.t4pack_frame import T4PackFrame


class PointCloudGTBatch(NamedTuple):
    """Named tuple to represent pointcloud features in a batch size with their batch indices."""

    points: Float32[
        Tensor, "batch_size*num_points num_features"
    ]  # (B*P, number of features for each point)
    # (B*P, ), batch indices for each point
    batch_indices: Int32[Tensor, " batch_size*num_points"]
    # Number of samples collated into this batch. Fixed by the number of collated samples, so it
    # stays correct even when the trailing samples carry zero points and are absent from
    # batch_indices.
    batch_size: int
    # Column of the time lag feature, -1 when the points carry none
    timestamp_difference_dim: int

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

        Raises:
          ValueError: If the samples carry the time lag in different columns.
        """
        if len(point_gt_samples) == 0:
            return None
        timestamp_difference_dims = {sample.timestamp_difference_dim for sample in point_gt_samples}
        if len(timestamp_difference_dims) != 1:
            raise ValueError(
                "All samples of a batch must have the time lag in the same column, got "
                f"{sorted(timestamp_difference_dims)}."
            )

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
            batch_size=len(point_gt_samples),
            timestamp_difference_dim=point_gt_samples[0].timestamp_difference_dim,
        )

    def sample_point_counts(self) -> Int64[Tensor, " batch_size"]:
        """
        Count the points of every sample of the batch.

        Returns:
          Int64[Tensor, " batch_size"]: Number of points of every sample, zero for a sample
            without points.
        """
        return torch.bincount(self.batch_indices.long(), minlength=self.batch_size)

    def offsets(self) -> Int64[Tensor, " batch_size"]:
        """
        Give the end of every sample in the concatenated points.

        Returns:
          Int64[Tensor, " batch_size"]: Cumulative point count of every sample.
        """
        return torch.cumsum(self.sample_point_counts(), dim=0)

    def split_points(self) -> list[Float32[Tensor, "num_points num_features"]]:
        """
        Split the concatenated points into the points of every sample.

        Returns:
          list: Points of every sample of the batch.
        """
        return list(torch.split(self.points, self.sample_point_counts().tolist()))

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
            batch_size=self.batch_size,
            timestamp_difference_dim=self.timestamp_difference_dim,
        )


class LidarSourceView(NamedTuple):
    """
    One lidar source inside a point cloud that merges several lidars.

    Attributes:
      point_index_begin: Index of the first point of the source in the merged cloud.
      num_points: Number of consecutive points the source contributes.
      sensor_to_frame_matrix: Transformation matrix from the source sensor to the frame of the
        merged cloud.
    """

    point_index_begin: int
    num_points: int
    sensor_to_frame_matrix: Float32[Tensor, "4 4"]


class LiDARPointCloudSample(NamedTuple):
    """
    Named tuple to represent a single row of LiDAR point cloud data,
    which contains the dataset record for the LiDAR point cloud task.
    """

    point_cloud_path: str
    timestamp: float
    # Number of float32 features stored per point in the file, declared by the database
    num_features: int
    # Intensity of the strongest return in the file, declared by the database
    intensity_scale: float
    # Transformation matrix from LiDAR sensor frame to ego pose of this LiDAR sensor frame
    sensor_to_ego_pose_matrix: Float32[Tensor, "4 4"]  # (4, 4)
    # Transformation matrix from ego pose of this LiDAR sensor frame to global frame
    lidar_to_ego_pose_to_global_matrix: Float32[Tensor, "4 4"]  # (4, 4)
    # Transformation matrix from the main lidar sensor to other lidar sweeps at this frame
    lidar_sensor_to_lidar_sweep_matrix: Float32[Tensor, "4 4"]  # (4, 4)
    # Location of the frame in the t4pack file of its channel, None when the scene has no pack
    t4pack_frame: T4PackFrame | None = None
    # Lidar source to serve from a merged cloud, None to serve the whole file
    source_view: LidarSourceView | None = None
