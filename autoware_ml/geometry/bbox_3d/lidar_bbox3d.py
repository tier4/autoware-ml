"""
This class extends the BaseBBoxes3D class to provide additional functionalities for
3D bounding boxes in LiDAR coordinate systems.

Note that the code is modified from:
https://github.com/open-mmlab/mmdetection3d/blob/main/mmdet3d/structures/bbox_3d/lidar_box3d.py
"""

from torch import Tensor
from jaxtyping import Float32, Int32
import torch

from autoware_ml.geometry.bbox_3d.base_bbox3d import BaseBBoxes3D
from autoware_ml.geometry.utils import create_axis_rotation_matrices, rotate_points_3d
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
        bbox_center_coordinate_type: Box3DCenterCoordinateType,
    ) -> None:
        """
        Initialize the LidarBBoxes3D instance.

        Args:
            bbox_params: A tensor of shape (num_bboxes, num_Box3DFieldIndex) representing the
                parameters of the 3D bounding boxes.
            bbox_labels: A tensor of shape (num_bboxes,) representing the labels of the 3D bounding boxes.
            bbox_center_coordinate_type: The center coordinate type of the 3D bounding boxes.
        """
        super().__init__(
            bbox_params=bbox_params,
            bbox_labels=bbox_labels,
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
        corners_norm = corners_norm - dims.new_tensor([0.5, 0.5, 0.5])
        corners = dims.view([-1, 1, 3]) * corners_norm.reshape([1, 8, 3])

        # rotate around z axis
        angles = torch.full(corners.shape[:1], self.yaw, device=tensor_device, dtype=tensor_dtype)
        rotation_matrices = create_axis_rotation_matrices(
            angles, axis=self.YAW_AXIS, clockwise=False
        )
        corners = rotate_points_3d(points=corners, rotation_matrices=rotation_matrices)
        corners += self.bbox_params[:, :3].view(-1, 1, 3)
        return corners

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
