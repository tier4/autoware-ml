"""
This class extends the BaseBBoxes3D class to provide additional functionalities for
3D bounding boxes in LiDAR coordinate systems.

Note that the code is modified from:
https://github.com/open-mmlab/mmdetection3d/blob/main/mmdet3d/structures/bbox_3d/lidar_box3d.py
"""

from typing import Sequence
from torch import Tensor
from jaxtyping import Bool, Float32, Int32
import torch

from autoware_ml.geometry.bbox_3d.base_bbox3d import BaseBBoxes3D
from autoware_ml.geometry.utils import (
    create_axis_rotation_matrices,
    rotate_points_3d,
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
        bbox_num_lidar_points: Int32[Tensor, " num_bboxes"],
        bbox_center_coordinate_type: Box3DCenterCoordinateType,
        bbox_attributes: Sequence[Sequence[str]] | None = None,
    ) -> None:
        """
        Initialize the LidarBBoxes3D instance.

        Args:
            bbox_params (Float32[Tensor, "num_bboxes num_Box3DFieldIndex"]): The parameters of the 3D bounding boxes.
            bbox_labels (Int32[Tensor, "num_bboxes"]): The labels of the 3D bounding boxes.
            bbox_label_names (Sequence[str]): The label names of the 3D bounding boxes.
            bbox_num_lidar_points (Int32[Tensor, "num_bboxes"]): The number of LiDAR points in each 3D bounding box.
            bbox_center_coordinate_type (Box3DCenterCoordinateType): The center coordinate type of the 3D bounding boxes.
                It only support "gravity_center (center of z is in the middle)" for now.
                We specify this to make sure users are aware of the center coordinate type being used.
            bbox_attributes (Sequence[Sequence[str]] | None): The attributes of every 3D bounding box, where
                the outer sequence is aligned with the bounding boxes and the inner sequence holds
                the attribute names of the corresponding bounding box. It is optional since not every
                dataset provides attributes. Defaults to None.
        """
        super().__init__(
            bbox_params=bbox_params,
            bbox_labels=bbox_labels,
            bbox_label_names=bbox_label_names,
            bbox_num_lidar_points=bbox_num_lidar_points,
            bbox_center_coordinate_type=bbox_center_coordinate_type,
            bbox_attributes=bbox_attributes,
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
        rotation_matrices = create_axis_rotation_matrices(
            self.yaw, axis=self.YAW_AXIS, clockwise=False
        )
        corners = rotate_points_3d(points=corners, rotation_matrices=rotation_matrices)
        corners += self.bbox_params[:, :3].view(-1, 1, 3)
        return corners

    def compute_points_in_bboxes(
        self, points: Float32[Tensor, "num_points 3"]
    ) -> Bool[Tensor, "num_bboxes num_points"]:
        """
        Mark the points inside every 3D bounding box, the surface included.

        The boxes are visited in a Python loop over slices of the sorted points, which suits
        the CPU callers (the dataloader filters and the database build). A CUDA caller would
        pay a host sync and a few kernel launches per box.

        Args:
            points (Float32[Tensor, "num_points 3"]): The point cloud data in shape (N, 3).

        Returns:
            Bool[Tensor, "num_bboxes num_points"]: A tensor mask indicating which points are
            inside each bounding box.
        """
        in_bboxes_mask = torch.zeros(
            (self.bbox_params.shape[0], points.shape[0]),
            dtype=torch.bool,
            device=self.bbox_params.device,
        )
        centers = self.center
        half_dims = self.dims / 2
        # Only the points within the circumscribed BEV radius of a box are tested: the slice
        # of the points sorted along x, found by binary search, narrowed along y by one
        # comparison. The radius carries a rounding margin so that the candidates hold every
        # point of the box surface.
        bev_radius = torch.linalg.vector_norm(half_dims[:, :2], dim=1) * (1 + 1e-5)
        sorted_x, order = torch.sort(points[:, 0])
        sorted_points = points[order]
        slice_begins = torch.searchsorted(sorted_x, centers[:, 0] - bev_radius)
        slice_ends = torch.searchsorted(sorted_x, centers[:, 0] + bev_radius, right=True)
        cos_yaw = torch.cos(self.yaw)
        sin_yaw = torch.sin(self.yaw)
        for index, (begin, end) in enumerate(zip(slice_begins.tolist(), slice_ends.tolist())):
            slice_points = sorted_points[begin:end]
            candidates = (slice_points[:, 1] - centers[index, 1]).abs() <= bev_radius[index]
            offsets = slice_points[candidates] - centers[index]
            # The BEV offsets rotated by -yaw lie in the box frame, z is unchanged
            local_x = cos_yaw[index] * offsets[:, 0] + sin_yaw[index] * offsets[:, 1]
            local_y = cos_yaw[index] * offsets[:, 1] - sin_yaw[index] * offsets[:, 0]
            inside = (
                (local_x.abs() <= half_dims[index, 0])
                & (local_y.abs() <= half_dims[index, 1])
                & (offsets[:, 2].abs() <= half_dims[index, 2])
            )
            in_bboxes_mask[index, order[begin:end][candidates][inside]] = True
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
