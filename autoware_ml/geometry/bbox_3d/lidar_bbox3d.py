"""
This class extends the BaseBBoxes3D class to provide additional functionalities for
3D bounding boxes in LiDAR coordinate systems.

Note that the code is modified from:
https://github.com/open-mmlab/mmdetection3d/blob/main/mmdet3d/structures/bbox_3d/lidar_box3d.py
"""

from typing import Sequence
from torch import Tensor
from jaxtyping import Float32, Int32
import torch

from autoware_ml.geometry.bbox_3d.base_bbox3d import BaseBBoxes3D
from autoware_ml.geometry.utils import (
    create_axis_rotation_matrices,
    rotate_points_3d,
    points_in_convex_polygon_3d,
)
from autoware_ml.types.spatial import BEVDirection, RotationAxis
from autoware_ml.types.geometry import Box3DFieldIndex, Box3DCenterCoordinateType


class LidarBBoxes3D(BaseBBoxes3D):
    """
    A class to represent 3D bounding boxes in LiDAR coordinate systems.
    This class extends the BaseBBoxes3D class and provides additional functionalities
    specific to LiDAR-based 3D bounding boxes.

    Coordinates in LiDAR (copy from mmdetection3d):

    .. code-block:: none

                                 up z    x front (yaw=0)
                                    ^   ^
                                    |  /
                                    | /
        (yaw=0.5*pi) left y <------ 0

    """

    YAW_AXIS: RotationAxis = RotationAxis.Z  # yaw axis is the z-axis in LiDAR coordinate system

    def __init__(
        self,
        bbox_params: Float32[Tensor, "num_bboxes num_Box3DFieldIndex"],
        bbox_labels: Int32[Tensor, " num_bboxes"],
        bbox_label_names: Sequence[str],
        bbox_num_points: Int32[Tensor, " num_bboxes"],
        bbox_center_coordinate_type: Box3DCenterCoordinateType,
    ) -> None:
        """
        Initialize the LidarBBoxes3D instance.

        Args:
            bbox_params (Float32[Tensor, "num_bboxes num_Box3DFieldIndex"]): The parameters of the 3D bounding boxes.
            bbox_labels (Int32[Tensor, "num_bboxes"]): The labels of the 3D bounding boxes.
            bbox_label_names (Sequence[str]): The label names of the 3D bounding boxes.
            bbox_num_points (Int32[Tensor, "num_bboxes"]): The number of points in each 3D bounding box.
            bbox_center_coordinate_type (Box3DCenterCoordinateType): The center coordinate type of the 3D bounding boxes.
                It only support "gravity_center (center of z is in the middle)" for now.
                We specify this to make sure users are aware of the center coordinate type being used.
        """
        super().__init__(
            bbox_params=bbox_params,
            bbox_labels=bbox_labels,
            bbox_label_names=bbox_label_names,
            bbox_num_points=bbox_num_points,
            bbox_center_coordinate_type=bbox_center_coordinate_type,
        )

    @property
    def corners(self) -> Tensor:
        """Convert boxes to corners in clockwise order, in the form of (x0y0z0,
        x0y0z1, x0y1z1, x0y1z0, x1y0z0, x1y0z1, x1y1z1, x1y1z0).

        .. code-block:: none

                                           up z
                            front x           ^
                                 /            |
                                /             |
                  (x1, y0, z1) + -----------  + (x1, y1, z1)
                              /|            / |
                             / |           /  |
               (x0, y0, z1) + ----------- +   + (x1, y1, z0)
                            |  /    .     |  /
                            | / origin    | /
            left y <------- + ----------- + (x0, y1, z0)
                (x0, y0, z0)

        Returns:
            Tensor: A tensor with 8 corners of each box in shape (N, 8, 3).
        """
        tensor_device = self.bbox_params.device
        tensor_dtype = self.bbox_params.dtype
        if self.bbox_params.numel() == 0:
            return torch.empty([0, 8, 3], device=tensor_device, dtype=tensor_dtype)

        dims = self.dims
        corners_norm = torch.stack(torch.unravel_index(torch.arange(8), [2] * 3), dim=1).to(
            device=tensor_device, dtype=tensor_dtype
        )

        corners_norm = corners_norm[[0, 1, 3, 2, 4, 5, 7, 6]]
        # use relative origin (0.5, 0.5, 0.5), where the center of the box is at (0.5, 0.5, 0.5)
        # in the normalized box coordinate system
        corners_norm = corners_norm - dims.new_tensor([0.5, 0.5, 0.0])
        corners = dims.view([-1, 1, 3]) * corners_norm.reshape([1, 8, 3])

        # rotate around z axis
        rotation_matrices = create_axis_rotation_matrices(
            self.yaw, axis=self.YAW_AXIS, clockwise=False
        )
        corners = rotate_points_3d(points=corners, rotation_matrices=rotation_matrices)
        corners += self.bbox_params[:, :3].view(-1, 1, 3)
        return corners

    def corners_to_surfaces_3d(self) -> Float32[Tensor, "num_bboxes 6 4 3"]:
        """
        Get the corners of the 3D bounding boxes in the form of surfaces.

        Returns:
            (num_bboxes, 6, 4, 3): The corners of the 3D bounding boxes as surfaces, where
                6 is the number of surface, 4 is the number of corners for each surface,
                and 3 is the (x, y, z) coordinates.
        """
        corners = self.corners
        surfaces = torch.stack(
            [
                corners[:, [0, 1, 2, 3]],  # front surface
                corners[:, [7, 6, 5, 4]],  # back surface
                corners[:, [0, 3, 7, 4]],  # bottom surface
                corners[:, [1, 5, 6, 2]],  # top surface
                corners[:, [0, 4, 5, 1]],  # left surface
                corners[:, [3, 2, 6, 7]],  # right surface
            ],
            dim=1,
        )
        return surfaces

    def compute_points_in_bboxes(
        self, points: Float32[Tensor, "num_points 3"]
    ) -> Float32[Tensor, "num_bboxes num_points"]:
        """
        Compute the number of points inside each 3D bounding box.

        Args:
            points (Float32[Tensor, "num_points 3"]): The point cloud data in shape (N, 3).

        Returns:
            Float32[Tensor, "num_bboxes num_points"]: A tensor indicating the number of points inside each bounding box.
        """
        if self.bbox_params.numel() == 0 or points.numel() == 0:
            return torch.zeros(
                (self.bbox_params.shape[0], points.shape[0]), device=self.bbox_params.device
            )

        # Create a mask for each bounding box to check if points are inside
        in_bboxes_mask = torch.zeros(
            (self.bbox_params.shape[0], points.shape[0]),
            dtype=torch.bool,
            device=self.bbox_params.device,
        )

        # Compute the surfaces of the bounding boxes
        surfaces = self.corners_to_surfaces_3d()
        in_bboxes_mask = points_in_convex_polygon_3d(points, surfaces)
        return in_bboxes_mask

    def rotate(
        self,
        rotation_matrix: Float32[Tensor, "3 3"],
    ) -> None:
        """Rotate boxes with points (optional) with the given angle or rotation
        matrix.

        Args:
            rotation_matrix (Float32[Tensor, "3 3"]): The rotation matrix to apply to the
                bounding boxes.

        Returns:
            tuple or None: When ``points`` is None, the function returns None,
            otherwise it returns the rotated points and the rotation matrix
            ``rot_mat_T``.
        """
        updated_translation = (
            self.bbox_params[:, [Box3DFieldIndex.X, Box3DFieldIndex.Y, Box3DFieldIndex.Z]]
            @ rotation_matrix
        )
        self.bbox_params[:, [Box3DFieldIndex.X, Box3DFieldIndex.Y, Box3DFieldIndex.Z]] = (
            updated_translation
        )

        rot_sin = rotation_matrix[0, 1]
        rot_cos = rotation_matrix[0, 0]
        angle = torch.arctan2(rot_sin, rot_cos)
        self.bbox_params[:, Box3DFieldIndex.YAW] += angle

        # TODO (KokSeang): Apply to velocity_z as well.
        # Velocity
        self.bbox_params[:, [Box3DFieldIndex.VELOCITY_X, Box3DFieldIndex.VELOCITY_Y]] = (
            self.bbox_params[:, [Box3DFieldIndex.VELOCITY_X, Box3DFieldIndex.VELOCITY_Y]]
            @ rotation_matrix[:2, :2]
        )

    def flip_bev(
        self,
        bev_direction: BEVDirection,
    ) -> None:
        """Flip the boxes in BEV along given BEV direction.

        In LIDAR coordinates, it flips the y (horizontal) or x (vertical) axis.

        Args:
            bev_direction (BEVDirection): Direction by which to flip. Can be chosen from
                'horizontal' and 'vertical'. Defaults to 'horizontal'.
            points (Tensor or np.ndarray or :obj:`BasePoints`, optional):
                Points to flip. Defaults to None.

        Returns:
            Tensor or np.ndarray or :obj:`BasePoints` or None: When ``points``
            is None, the function returns None, otherwise it returns the
            flipped points.
        """
        if bev_direction == BEVDirection.HORIZONTAL:
            self.bbox_params[
                :, [Box3DFieldIndex.Y, Box3DFieldIndex.VELOCITY_Y]
            ] = -self.bbox_params[:, [Box3DFieldIndex.Y, Box3DFieldIndex.VELOCITY_Y]]
            self.bbox_params[:, Box3DFieldIndex.YAW] = -self.bbox_params[:, Box3DFieldIndex.YAW]
        elif bev_direction == BEVDirection.VERTICAL:
            self.bbox_params[
                :, [Box3DFieldIndex.X, Box3DFieldIndex.VELOCITY_X]
            ] = -self.bbox_params[:, [Box3DFieldIndex.X, Box3DFieldIndex.VELOCITY_X]]
            self.bbox_params[:, Box3DFieldIndex.YAW] = (
                -self.bbox_params[:, Box3DFieldIndex.YAW] + torch.pi
            )
