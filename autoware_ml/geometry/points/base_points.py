"""
It is a base class for all point cloud data structures, providing common attributes and methods that can be used by derived classes.

Note that the code is modified from:
https://github.com/open-mmlab/mmdetection3d/blob/main/mmdet3d/structures/points/base_points.py
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import copy
from typing import Sequence

from jaxtyping import Float32, Int32, Bool
import numpy.typing as npt
import numpy as np
import torch
from torch import Tensor

from autoware_ml.types.geometry import PointFieldIndex, PointFeatureName
from autoware_ml.types.spatial import BEVDirection


class BasePoints(ABC):
    """
    Abstract base class for point cloud data that defines the common interface
    for every point cloud type.
    Note that it only supports tensor data type for now.
    """

    def __init__(
        self,
        points: Float32[Tensor, "num_points num_point_features"],
        point_feature_names: Sequence[PointFeatureName],
        timestamp_seconds: float,
    ) -> None:
        """
        Initialize the BasePoints instance.

        Args:
            points: A tensor of shape (num_points, num_point_features) representing the point cloud data.
            point_feature_names: A sequence of PointFeatureName representing the names of the features for each point.
            timestamp_seconds: A float representing the timestamp of the point cloud data in seconds.
        """
        self._points = points
        self._point_feature_names = point_feature_names
        self._timestamp_seconds = timestamp_seconds
        # Dimension index for the timestamp difference feature, if it exists. -1 indicates
        # that it does not exist.
        self._timestamp_difference_dim = -1

    @property
    def timestamp_seconds(self) -> float:
        """Timestamp of the point cloud data in seconds."""
        return self._timestamp_seconds

    def __len__(self) -> int:
        """
        Get the total number of points.

        Returns:
            int: The number of points.
        """
        return self.shape[0]

    def __repr__(self) -> str:
        """
        Get the string representation of the 3D bounding boxes.

        Returns:
            str: The string representation of the 3D bounding boxes.
        """
        return self.__class__.__name__ + f"(shape={self.shape})"

    def __eq__(self, other: object) -> bool:
        """
        Check if two BasePoints instances are equal.

        Args:
            other: Another BasePoints instance to compare with.
        """
        if not isinstance(other, BasePoints):
            return NotImplemented

        return (
            torch.equal(self.points, other.points)
            and self.point_feature_names == other.point_feature_names
        )

    @property
    def points(self) -> Float32[Tensor, "num_points num_point_features"]:
        """Points in shape (num_points, num_point_features)."""
        return self._points

    @property
    def coords(self) -> Float32[Tensor, "num_points 3"]:
        """Coordinates of each point in shape (num_points, 3)."""
        return self.points[:, [PointFieldIndex.X, PointFieldIndex.Y, PointFieldIndex.Z]]

    @coords.setter
    def coords(self, coords: Float32[Tensor, "num_points 3"]) -> None:
        """Set the coordinates of each point in shape (num_points, 3)."""
        self.points[:, [PointFieldIndex.X, PointFieldIndex.Y, PointFieldIndex.Z]] = coords

    @property
    def point_feature_names(self) -> Sequence[PointFeatureName]:
        """Names of the features for each point."""
        return self._point_feature_names

    @property
    def shape(self) -> torch.Size:
        """torch.Size(int, int): Shape of points."""
        return self.points.shape

    @property
    def bev_coords(self) -> Float32[Tensor, "num_points 2"]:
        """Coordinates in BEV (x and y) of the points in shape (num_points, 2)."""
        return self.points[:, [PointFieldIndex.X, PointFieldIndex.Y]]

    @property
    def device(self) -> torch.device:
        """torch.device: Device of the points."""
        return self.points.device

    @property
    def timestamp_difference_dim(self) -> int:
        """int: Dimension index for the timestamp difference feature, if it exists. -1 indicates that it does not exist."""
        return self._timestamp_difference_dim

    def add_timestamp_difference(self, timestamp_difference: float) -> None:
        """Add a timestamp difference feature to the points.

        Args:
            timestamp_difference (float): The timestamp difference to be added to each point.
        """
        num_points = self.points.shape[0]
        timestamp_column = torch.full(
            (num_points, 1),
            timestamp_difference,
            dtype=self.points.dtype,
            device=self.points.device,
        )
        self._points = torch.cat((self._points, timestamp_column), dim=1)
        self._point_feature_names = list(self._point_feature_names) + [
            PointFeatureName.TIMESTAMP_DIFFERENCE
        ]
        self._timestamp_difference_dim = self._points.shape[1] - 1

    def remove_points(self, valid_mask: Bool[Tensor, " num_points"]) -> None:
        """Remove points based on a validity mask.

        Args:
            valid_mask (Bool[Tensor, " num_points"]): A binary mask indicating which points to keep.
        """
        self._points = self._points[valid_mask]

    def shuffle(self) -> Int32[Tensor, " num_points"]:
        """Shuffle the points.

        Returns:
            Int32[Tensor, " num_points"]: The shuffled index.
        """
        idx = torch.randperm(self.__len__(), device=self.points.device)
        self._points = self._points[idx]
        return idx

    def rotate(self, rotation_matrix: Float32[Tensor, "3 3"]) -> None:
        """Rotate points with the given rotation matrix or angle.

        Args:
            rotation_matrix (Float32[Tensor, "3 3"]): Rotation matrix.

        """
        self.points[:, [PointFieldIndex.X, PointFieldIndex.Y, PointFieldIndex.Z]] = (
            self.points[:, [PointFieldIndex.X, PointFieldIndex.Y, PointFieldIndex.Z]]
            @ rotation_matrix
        )

    @abstractmethod
    def flip_bev(self, bev_direction: BEVDirection) -> None:
        """Flip the points along given BEV direction.

        Args:
            bev_direction (BEVDirection): Flip direction (horizontal or vertical).
        """
        pass

    def translate(self, trans_vector: Float32[Tensor, "1 3"]) -> None:
        """Translate points with the given translation vector.

        Args:
            trans_vector (Float32[Tensor, "1 3"]): Translation vector of size 1x3.

        """
        self.points[:, [PointFieldIndex.X, PointFieldIndex.Y, PointFieldIndex.Z]] += trans_vector

    def in_range_3d(self, point_range: Float32[Tensor, "6"]) -> Bool[Tensor, " num_points"]:
        """Check whether the points are in the given range.

        Args:
            point_range (Float32[Tensor, "6"]): The range of
                point (x_min, y_min, z_min, x_max, y_max, z_max).

        Note:
            In the original implementation of SECOND, checking whether a box in
            the range checks whether the points are in a convex polygon, we try
            to reduce the burden for simpler cases.

        Returns:
            Bool[Tensor, " num_points"]: A binary vector indicating whether each point is inside the
            reference range.
        """
        in_range_flags = (
            (self.points[:, PointFieldIndex.X] >= point_range[0])
            & (self.points[:, PointFieldIndex.Y] >= point_range[1])
            & (self.points[:, PointFieldIndex.Z] >= point_range[2])
            & (self.points[:, PointFieldIndex.X] <= point_range[3])
            & (self.points[:, PointFieldIndex.Y] <= point_range[4])
            & (self.points[:, PointFieldIndex.Z] <= point_range[5])
        )
        return in_range_flags

    def in_range_bev(self, point_range: Float32[Tensor, "4"]) -> Bool[Tensor, " num_points"]:
        """Check whether the points are in the given BEV range.

        Args:
            point_range (Float32[Tensor, "4"]): The range of
                point (x_min, y_min, x_max, y_max).

        Note:
            In the original implementation of SECOND, checking whether a box in
            the range checks whether the points are in a convex polygon, we try
            to reduce the burden for simpler cases.

        Returns:
            Bool[Tensor, " num_points"]: A binary vector indicating whether each point is inside the
            reference range.
        """
        in_range_flags = (
            (self.points[:, PointFieldIndex.X] >= point_range[0])
            & (self.points[:, PointFieldIndex.Y] >= point_range[1])
            & (self.points[:, PointFieldIndex.X] <= point_range[2])
            & (self.points[:, PointFieldIndex.Y] <= point_range[3])
        )
        return in_range_flags

    def scale(self, scale_factor: float) -> None:
        """Scale the points with horizontal and vertical scaling factors.

        Args:
            scale_factor (float): Scale factor to scale the points.
        """
        self._points[:, [PointFieldIndex.X, PointFieldIndex.Y, PointFieldIndex.Z]] *= scale_factor

    @classmethod
    def concat(cls, points: Sequence[BasePoints]) -> BasePoints:
        """Concatenate a sequence of BasePoints instances into a single BasePoints instance.

        Args:
            points (Sequence[BasePoints]): A sequence of BasePoints instances to be concatenated.

        Returns:
            BasePoints: A new BasePoints instance containing the concatenated points.
        """
        if not points:
            raise ValueError("The points list must not be empty.")

        # Ensure all point_feature_names are the same
        first_point_feature_names = points[0].point_feature_names
        for point in points:
            if point.point_feature_names != first_point_feature_names:
                raise ValueError(
                    "All BasePoints instances must have the same point_feature_names for concatenation."
                )

        concatenated_points = torch.cat([point.points for point in points], dim=0)
        # Always use the timestamp_seconds of the first BasePoints instance for the concatenated result
        return cls(
            concatenated_points,
            first_point_feature_names,
            timestamp_seconds=points[0].timestamp_seconds,
        )

    def to_numpy(self) -> npt.NDArray[np.float32]:
        """Convert the points to a numpy array.

        Returns:
            npt.NDArray[np.float32]: The points as a numpy array.
        """
        return self.points.cpu().numpy()

    def to(self, device: torch.device) -> None:
        """
        Move the 3D bounding boxes to a specified device.

        Args:
            device (torch.device): The device to move the bounding boxes to.
        """
        self._points = self._points.to(device)

    def detach(self) -> BasePoints:
        """
        Detach the 3D bounding boxes from the current computation graph.

        Returns:
            BasePoints: The detached 3D bounding boxes.
        """
        new_points = self.shallow_copy()
        new_points._points = new_points._points.detach()
        return new_points

    def shallow_copy(self) -> BasePoints:
        """
        Create a shallow copy of the 3D bounding boxes.

        Returns:
            BasePoints: A shallow copy of the 3D bounding boxes.
        """
        return copy.copy(self)

    def deep_copy(self) -> BasePoints:
        """
        Create a deep copy of the 3D bounding boxes.

        Returns:
            BasePoints: A deep copy of the 3D bounding boxes.
        """
        return copy.deepcopy(self)

    @classmethod
    def from_numpy(
        cls,
        points_np: npt.NDArray[np.float32],
        point_feature_names: Sequence[PointFeatureName],
        timestamp_seconds: float,
    ) -> BasePoints:
        """Load points from a numpy array.

        Args:
            points_np (npt.NDArray[np.float32]): The points as a numpy array.
            point_feature_names (Sequence[PointFeatureName]): The names of the point features.
            timestamp_seconds (float): The timestamp of the point cloud data in seconds.
        """
        return cls(
            torch.from_numpy(points_np).float(),
            point_feature_names,
            timestamp_seconds=timestamp_seconds,
        )
