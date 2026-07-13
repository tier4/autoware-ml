"""
This code provides the base class for 3D bounding boxes in different coordinate systems.
It defines the interface and common properties for 3D bounding boxes, which can be extended by
specific implementations for different coordinate systems (e.g., camera, LiDAR, etc.).
Note that the code is modified from:
https://github.com/open-mmlab/mmdetection3d/blob/main/mmdet3d/structures/bbox_3d/base_box3d.py
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import copy
from typing import Sequence

from jaxtyping import Float32, Int32, Bool
import torch
from torch import Tensor
import numpy as np
import numpy.typing as npt

from autoware_ml.types.geometry import Box3DFieldIndex, Box3DCenterCoordinateType
from autoware_ml.types.spatial import BEVDirection, RotationAxis


class BaseBBoxes3D(ABC):
    """
    Base class for 3D bounding boxes in different coordinate systems.
    This class defines the interface and common properties for 3D bounding boxes, which can be
    extended by specific implementations for different coordinate systems (e.g., camera, LiDAR, etc.).
    """

    # Default yaw axis is the x-axis.
    # Subclasses should override this attribute if needed.
    YAW_AXIS: RotationAxis = RotationAxis.X

    def __init__(
        self,
        bbox_params: Float32[Tensor, "num_bboxes num_Box3DFieldIndex"],
        bbox_labels: Int32[Tensor, " num_bboxes"],
        bbox_label_names: Sequence[str],
        bbox_num_points: Int32[Tensor, " num_bboxes"],
        bbox_center_coordinate_type: Box3DCenterCoordinateType,
    ) -> None:
        """
        Initialize the base class for 3D bounding boxes. Note that the class in only available
            in Torch.tensor. And It's always assume the height is in gravity center
            (middle of the box).

        Args:
            bbox_params (Float32[Tensor, "num_bboxes num_Box3DFieldIndex"]): The parameters of the 3D bounding boxes.
            bbox_labels (Int32[Tensor, "num_bboxes"]): The labels of the 3D bounding boxes.
            bbox_label_names (Sequence[str]): The label names of the 3D bounding boxes.
            bbox_num_points (Int32[Tensor, "num_bboxes"]): The number of points in each 3D bounding box.
            bbox_center_coordinate_type (Box3DCenterCoordinateType): The center coordinate type of the 3D bounding boxes.
                It only support "gravity_center (center of z is in the middle)" for now.
                We specify this to make sure users are aware of the center coordinate type being used.
        """

        self._bbox_params = bbox_params
        self._bbox_labels = bbox_labels
        self._bbox_label_names = bbox_label_names
        self._bbox_num_points = bbox_num_points
        self._bbox_center_coordinate_type = bbox_center_coordinate_type
        if self._bbox_center_coordinate_type != Box3DCenterCoordinateType.GRAVITY_CENTER:
            raise ValueError(
                f"Only gravity center coordinate type is supported for now, but got {self._bbox_center_coordinate_type}"
            )

        # Verify if bbox_params and bbox_labels are correct
        self._verify_bbox_params()

    def __len__(self) -> int:
        """
        Get the number of 3D bounding boxes.

        Returns:
            int: The number of 3D bounding boxes.
        """
        return self.shape[0]

    def __repr__(self) -> str:
        """
        Get the string representation of the 3D bounding boxes.

        Returns:
            str: The string representation of the 3D bounding boxes.
        """
        return self.__class__.__name__ + f"(shape={self.shape})"

    def __eq__(self, other: BaseBBoxes3D) -> bool:
        """
        Check if two BaseBBoxes3D instances are equal.

        Args:
            other (BaseBBoxes3D): Another instance of BaseBBoxes3D to compare with.

        Returns:
            bool: True if the two instances are equal, False otherwise.
        """
        if not isinstance(other, BaseBBoxes3D):
            return NotImplemented

        return (
            torch.equal(self.bbox_params, other.bbox_params)
            and torch.equal(self.bbox_labels, other.bbox_labels)
            and self.bbox_label_names == other.bbox_label_names
            and self.bbox_center_coordinate_type == other.bbox_center_coordinate_type
        )

    def _verify_bbox_params(self) -> None:
        """
        Verify the shape of the bbox_params tensor.

        Raises:
            ValueError: If the shape of bbox_params is not (N, len(Box3DFieldIndex)).
        """
        if self.bbox_params.ndim != 2 or self.bbox_params.shape[1] != len(Box3DFieldIndex):
            raise ValueError(
                f"bbox_params must have shape (N, {len(Box3DFieldIndex)}), but got {self.bbox_params.shape}"
            )

        if self.bbox_labels.ndim != 1 or self.bbox_labels.shape[0] != self.bbox_params.shape[0]:
            raise ValueError(
                f"bbox_labels must have shape (N,), where N is the number of bounding boxes, "
                f"but got {self.bbox_labels.shape} for {self.bbox_params.shape[0]} bounding boxes"
            )

        if not (self.dims > 0).all():
            raise ValueError("All bounding boxes must have positive dimensions.")

    @property
    def bbox_params(self) -> Float32[Tensor, "num_bboxes num_Box3DFieldIndex"]:
        """
        Get the parameters of the 3D bounding boxes.

        Returns:
            (num_bboxes, num_Box3DFieldIndex): The parameters of the 3D bounding boxes, where
            num_bboxes is the number of bounding boxes and num_Box3DFieldIndex is the number of
            parameters defined in Box3DFieldIndex.
        """
        return self._bbox_params

    @property
    def bbox_label_names(self) -> Sequence[str]:
        """
        Get the label names of the 3D bounding boxes.

        Returns:
            Sequence[str]: The label names of the 3D bounding boxes.
        """
        return self._bbox_label_names

    @property
    def bbox_num_points(self) -> Int32[Tensor, " num_bboxes"]:
        """
        Get the number of points in each 3D bounding box.

        Returns:
            (num_bboxes,): The number of points in each 3D bounding box.
        """
        return self._bbox_num_points

    @property
    def bbox_center_coordinate_type(self) -> Box3DCenterCoordinateType:
        """
        Get the center coordinate type of the 3D bounding boxes.

        Returns:
            Box3DCenterCoordinateType: The center coordinate type of the 3D bounding boxes.
        """
        return self._bbox_center_coordinate_type

    @property
    def shape(self) -> torch.Size:
        """
        Get the shape of the bounding boxes.

        Returns:
            torch.Size(int, int): The shape of the bounding boxes as (N, len(Box3DFieldIndex)),
              where N is the number of bounding boxes, C is the number of channels.
        """
        return self._bbox_params.shape

    @property
    def volume(self) -> Float32[Tensor, " num_bboxes"]:
        """
        Calculate the volume of the 3D bounding boxes.

        Returns:
            (N, ): The volume of the 3D bounding boxes.
        """
        length = self._bbox_params[:, Box3DFieldIndex.LENGTH]
        width = self._bbox_params[:, Box3DFieldIndex.WIDTH]
        height = self._bbox_params[:, Box3DFieldIndex.HEIGHT]
        return length * width * height

    @property
    def area(self) -> Float32[Tensor, " num_bboxes"]:
        """
        Calculate the area of the 3D bounding boxes.

        Returns:
            (N, ): The area of the 3D bounding boxes.
        """
        length = self._bbox_params[:, Box3DFieldIndex.LENGTH]
        width = self._bbox_params[:, Box3DFieldIndex.WIDTH]
        return length * width

    @property
    def center(self) -> Float32[Tensor, "num_bboxes 3"]:
        """
        Get the center coordinates of the 3D bounding boxes.

        Returns:
            (N, 3): The center coordinates of the 3D bounding boxes as (x, y, z).
        """
        return self._bbox_params[:, [Box3DFieldIndex.X, Box3DFieldIndex.Y, Box3DFieldIndex.Z]]

    @property
    def height(self) -> Float32[Tensor, " num_bboxes"]:
        """
        Get the height of the 3D bounding boxes.

        Returns:
            (N, ): The height of the 3D bounding boxes.
        """
        return self._bbox_params[:, Box3DFieldIndex.HEIGHT]

    @property
    def yaw(self) -> Float32[Tensor, " num_bboxes"]:
        """
        Get the yaw angle of the 3D bounding boxes.

        Returns:
            (N, ): The yaw angle of the 3D bounding boxes.
        """
        return self._bbox_params[:, Box3DFieldIndex.YAW]

    @property
    def dims(self) -> Float32[Tensor, "num_bboxes 3"]:
        """
        Get the dimensions (length, width, height) of the 3D bounding boxes.

        Returns:
            (N, 3): The dimensions of the 3D bounding boxes as (length, width, height).
        """
        return self._bbox_params[
            :,
            [
                Box3DFieldIndex.LENGTH,
                Box3DFieldIndex.WIDTH,
                Box3DFieldIndex.HEIGHT,
            ],
        ]

    @property
    def center_z(self) -> Float32[Tensor, " num_bboxes"]:
        """
        Get the center height (z-coordinate) of the 3D bounding boxes.

        Returns:
            (N, ): The center height of the 3D bounding boxes.
        """
        return self._bbox_params[:, Box3DFieldIndex.Z.value]

    @property
    def velocity(self) -> Float32[Tensor, "num_bboxes 3"]:
        """
        Get the velocity components (vx, vy, vz) of the 3D bounding boxes.

        Returns:
            (N, 3): The velocity components of the 3D bounding boxes as (vx, vy, vz).
        """
        return self._bbox_params[
            :,
            [
                Box3DFieldIndex.VELOCITY_X,
                Box3DFieldIndex.VELOCITY_Y,
                Box3DFieldIndex.VELOCITY_Z,
            ],
        ]

    @property
    def bbox_labels(self) -> Int32[Tensor, " num_bboxes"]:
        """
        Get the labels of the 3D bounding boxes.

        Returns:
            (N, ): The labels of the 3D bounding boxes.
        """
        return self._bbox_labels

    @property
    @abstractmethod
    def corners(self) -> Float32[Tensor, "num_bboxes 8 3"]:
        """
        Get the corners of the 3D bounding boxes.

        Returns:
            (num_bboxes, 8, 3): The corners of the 3D bounding boxes as (x, y, z) coordinates.
        """
        raise NotImplementedError("Subclasses must implement the `corners` property.")

    @abstractmethod
    def corners_to_surfaces_3d(self) -> Float32[Tensor, "num_bboxes 6 4 3"]:
        """
        Get the corners of the 3D bounding boxes in the form of surfaces.

        Returns:
            (num_bboxes, 6, 4, 3): The corners of the 3D bounding boxes as surfaces, where
                6 is the number of surface, 4 is the number of corners for each surface,
                and 3 is the (x, y, z) coordinates.
        """
        raise NotImplementedError(
            "Subclasses must implement the `corners_to_surfaces_3d` property."
        )

    @abstractmethod
    def compute_points_in_bboxes(
        self, points: Float32[Tensor, "num_points 3"]
    ) -> Bool[Tensor, "num_bboxes num_points"]:
        """
        Compute whether the given points are inside the 3D bounding boxes.

        Args:
            points (Float32[Tensor, "num_points 3"]): The points to check, with shape (num_points, 3).

        Returns:
            Bool[Tensor, "num_bboxes num_points"]: A boolean tensor indicating whether each point is inside each bounding box.
        """
        raise NotImplementedError(
            "Subclasses must implement the `compute_points_in_bboxes` method."
        )

    @property
    def bev_center(self) -> Float32[Tensor, "num_bboxes 2"]:
        """
        Get the BEV (Bird's Eye View) center coordinates of the 3D bounding boxes.

        Returns:
            (num_bboxes, 2): The BEV center coordinates of the 3D bounding boxes as (x, y).
        """
        return self._bbox_params[:, [Box3DFieldIndex.X, Box3DFieldIndex.Y]]

    @property
    def bev_dims(self) -> Float32[Tensor, "num_bboxes 2"]:
        """
        Get the BEV (Bird's Eye View) dimensions (length, width) of the 3D bounding boxes.

        Returns:
            (num_bboxes, 2): The BEV dimensions of the 3D bounding boxes as (length, width).
        """
        return self._bbox_params[:, [Box3DFieldIndex.LENGTH, Box3DFieldIndex.WIDTH]]

    @abstractmethod
    def rotate(
        self,
        rotation_matrix: Float32[Tensor, "3 3"],
    ) -> None:
        """
        Rotate the 3D bounding boxes globally using a given rotation angle.

        Args:
            rotation_matrix (Float32[Tensor, "3 3"]): The rotation matrix to apply to the bounding boxes.
                It should be a 3x3 matrix representing the rotation in 3D space.
        """
        raise NotImplementedError("Subclasses must implement the `rotate` method.")

    @abstractmethod
    def flip_bev(
        self,
        bev_direction: BEVDirection,
    ) -> None:
        """
        Flip the 3D bounding boxes globally using a given flip direction.

        Args:
            bev_direction (BEVDirection): The flip direction to apply to the bounding boxes.
                It can be 'horizontal' or 'vertical'.
        """
        raise NotImplementedError("Subclasses must implement the `flip_bev` method.")

    def translate(self, translation_vector: Float32[Tensor, "1 3"]) -> None:
        """
        Translate the 3D bounding boxes globally using a given translation vector.

        Args:
            translation_vector (Tensor.float32, (1, 3)): The translation vector to apply to the
                bounding boxes.
        """
        self._bbox_params[:, [Box3DFieldIndex.X, Box3DFieldIndex.Y, Box3DFieldIndex.Z]] += (
            translation_vector
        )

    def in_range_3d(self, bev_range: Float32[Tensor, "6"]) -> Bool[Tensor, " num_bboxes"]:
        """
        Check if the 3D bounding boxes are within a given BEV (Bird's Eye View) range.

        Args:
            bev_range (Float32[Tensor, "6"]): The BEV range to check against, defined as
                [x_min, y_min, z_min, x_max, y_max, z_max].

        Returns:
            Tensor.bool, (num_bboxes,): A boolean tensor to indicate whether
                each 3D bounding box is within the BEV range.
        """
        if bev_range.shape != (6,):
            raise ValueError(
                "BEV range must be a 1D array of shape (6,) representing [x_min, y_min, z_min, x_max, y_max, z_max]."
            )

        in_range_masks = (
            (self.center[:, 0] >= bev_range[0])
            & (self.center[:, 0] <= bev_range[3])
            & (self.center[:, 1] >= bev_range[1])
            & (self.center[:, 1] <= bev_range[4])
            & (self.center[:, 2] >= bev_range[2])
            & (self.center[:, 2] <= bev_range[5])
        )
        return in_range_masks

    def in_range_bev(self, bev_range: Float32[Tensor, "4"]) -> Bool[Tensor, " num_bboxes"]:
        """
        Check if the 3D bounding boxes are within a given BEV (Bird's Eye View) range.

        Args:
            bev_range (Float32[Tensor, "4"]): The BEV range to check against, defined as
                [x_min, y_min, x_max, y_max].

        Returns:
            Tensor.bool, (num_bboxes,): A boolean tensor to indicate whether
                each 3D bounding box is within the BEV range.
        """
        if bev_range.shape != (4,):
            raise ValueError(
                "BEV range must be a 1D array of shape (4,) representing [x_min, y_min, x_max, y_max]."
            )

        in_range_masks = (
            (self.center[:, 0] >= bev_range[0])
            & (self.center[:, 0] <= bev_range[2])
            & (self.center[:, 1] >= bev_range[1])
            & (self.center[:, 1] <= bev_range[3])
        )
        return in_range_masks

    def remove_bboxes(self, valid_masks: Bool[Tensor, " num_bboxes"]) -> None:
        """
        Remove 3D bounding boxes based on a boolean mask.

        Args:
            valid_masks (Bool[Tensor, " num_bboxes"]): A boolean tensor indicating which bounding boxes to keep.
        """
        self._bbox_params = self._bbox_params[valid_masks]
        self._bbox_labels = self._bbox_labels[valid_masks]
        self._bbox_num_points = self._bbox_num_points[valid_masks]
        self._bbox_label_names = [
            name for i, name in enumerate(self._bbox_label_names) if valid_masks[i]
        ]

    def scale(self, scale_factor: float) -> None:
        """
        Horizontally and vertically scale the 3D bounding boxes using a given scale factor.
            Note that the scale is applied to the center coordinates, dimensions,
            and velocity components of the bounding boxes.

        Args:
            scale_factor (float): The scale factor to apply to the bounding boxes.
        """
        self._bbox_params[
            :,
            [
                Box3DFieldIndex.LENGTH,
                Box3DFieldIndex.WIDTH,
                Box3DFieldIndex.HEIGHT,
            ],
        ] *= scale_factor
        self._bbox_params[:, [Box3DFieldIndex.X, Box3DFieldIndex.Y, Box3DFieldIndex.Z]] *= (
            scale_factor
        )
        # TODO (KokSeang): Apply to velocity_z as well.
        self._bbox_params[
            :,
            [
                Box3DFieldIndex.VELOCITY_X,
                Box3DFieldIndex.VELOCITY_Y,
            ],
        ] *= scale_factor

    def limit_yaw(self, offset: float = 0.5, period: float | None = None) -> None:
        """
        Limit the yaw angle of the 3D bounding boxes to a specified range.

        Args:
            offset (float): The offset to apply to the yaw angle. Default is 0.5.
            period (float | None): The period to limit the yaw angle. If None, it will be set to 2 * torch.pi.
        """
        if period is None:
            period = 2 * torch.pi

        bboxes_yaw = self._bbox_params[:, Box3DFieldIndex.YAW]
        self._bbox_params[:, Box3DFieldIndex.YAW] = (
            bboxes_yaw - torch.floor(bboxes_yaw / period + offset) * period
        )

    def to_numpy(self) -> npt.NDArray[np.float32]:
        """
        Convert the 3D bounding boxes to a NumPy array.

        Returns:
            npt.NDArray[np.float32] (N, len(Box3DFieldIndex)): A NumPy array representation of the
              3D bounding boxes.
        """
        return self.bbox_params.cpu().numpy()

    def to(self, device: torch.device) -> None:
        """
        Move the 3D bounding boxes to a specified device.

        Args:
            device (torch.device): The device to move the bounding boxes to.
        """
        self._bbox_params = self._bbox_params.to(device)
        self._bbox_labels = self._bbox_labels.to(device)

    def detach(self) -> BaseBBoxes3D:
        """
        Detach the 3D bounding boxes from the current computation graph.

        Returns:
            BaseBBoxes3D: The detached 3D bounding boxes.
        """
        new_bboxes_3d = self.shallow_copy()
        new_bboxes_3d._bbox_params = new_bboxes_3d._bbox_params.detach()
        new_bboxes_3d._bbox_labels = new_bboxes_3d._bbox_labels.detach()
        return new_bboxes_3d

    def shallow_copy(self) -> BaseBBoxes3D:
        """
        Create a shallow copy of the 3D bounding boxes.

        Returns:
            BaseBBoxes3D: A shallow copy of the 3D bounding boxes.
        """
        return copy.copy(self)

    def deep_copy(self) -> BaseBBoxes3D:
        """
        Create a deep copy of the 3D bounding boxes.

        Returns:
            BaseBBoxes3D: A deep copy of the 3D bounding boxes.
        """
        return copy.deepcopy(self)

    @classmethod
    def from_numpy(
        cls,
        bbox_params: npt.NDArray[np.float32],
        bbox_labels: npt.NDArray[np.int32],
        bbox_label_names: Sequence[str],
        bbox_num_points: npt.NDArray[np.int32],
        bbox_center_coordinate_type: Box3DCenterCoordinateType,
    ) -> BaseBBoxes3D:
        """
        Create a BaseBBoxes3D instance from a NumPy array.

        Args:
            bbox_params (npt.NDArray[np.float32], (num_bboxes, len(Box3DFieldIndex))): A NumPy array
                representation of the 3D bounding boxes.
            bbox_labels (npt.NDArray[np.int32], (num_bboxes,)): A NumPy array representation of the
                labels of the 3D bounding boxes.
            bbox_num_points (npt.NDArray[np.int32], (num_bboxes,)): A NumPy array representation of the
                number of points in each 3D bounding box.
            bbox_center_coordinate_type (Box3DCenterCoordinateType): The center coordinate type of
                the 3D bounding boxes.
        """
        bbox_params_tensor = torch.from_numpy(bbox_params).float()
        bbox_labels_tensor = torch.from_numpy(bbox_labels).int()
        bbox_num_points_tensor = torch.from_numpy(bbox_num_points).int()
        return cls(
            bbox_params=bbox_params_tensor,
            bbox_labels=bbox_labels_tensor,
            bbox_num_points=bbox_num_points_tensor,
            bbox_center_coordinate_type=bbox_center_coordinate_type,
            bbox_label_names=bbox_label_names,
        )
