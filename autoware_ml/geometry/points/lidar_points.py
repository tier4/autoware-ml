"""
This class provides a class for LiDAR point cloud data structures.

Note that the code is modified from:
https://github.com/open-mmlab/mmdetection3d/blob/main/mmdet3d/structures/points/lidar_points.py
"""

from __future__ import annotations

from typing import Sequence

from jaxtyping import Float32
from torch import Tensor

from autoware_ml.geometry.points.base_points import BasePoints
from autoware_ml.types.geometry import PointFieldIndex, PointFeatureName
from autoware_ml.types.spatial import BEVDirection


class LiDARPoints(BasePoints):
    """
    LiDAR point cloud data structure that extends the BasePoints class to support LiDAR-specific
    functionalities.
    """

    def __init__(
        self,
        points: Float32[Tensor, "num_points num_point_features"],
        point_feature_names: Sequence[PointFeatureName],
        timestamp: float,
    ) -> None:
        """
        Initialize the LiDARPoints instance.

        Args:
            points: A tensor of shape (num_points, num_point_features) representing the point cloud data.
            point_feature_names: A sequence of PointFeatureName representing the names of the features for each point.
            timestamp: A float representing the timestamp of the point cloud data in seconds.
        """
        super().__init__(
            points=points,
            point_feature_names=point_feature_names,
            timestamp=timestamp,
        )

    def flip_bev(self, bev_direction: BEVDirection) -> None:
        """Flip the points along given BEV direction.

        Args:
            bev_direction (BEVDirection): Flip direction (horizontal or vertical).
        """
        if bev_direction == BEVDirection.HORIZONTAL:
            self._points[:, PointFieldIndex.Y.value] = -self._points[:, PointFieldIndex.Y.value]
        elif bev_direction == BEVDirection.VERTICAL:
            self._points[:, PointFieldIndex.X.value] = -self._points[:, PointFieldIndex.X.value]
        else:
            raise ValueError(f"Invalid BEV direction: {bev_direction}")
