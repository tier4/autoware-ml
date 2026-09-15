"""
This script provides utility functions for geometric operations, particularly for 3D bounding boxes
and related transformations.
The code is modified from:
https://github.com/open-mmlab/mmdetection3d/blob/main/mmdet3d/structures/bbox_3d/utils.py
"""

import numpy as np
from jaxtyping import Float32
from torch import Tensor
import torch

from autoware_ml.types.spatial import RotationAxis


def create_axis_rotation_matrices(
    angles: Float32[Tensor, " num_angles"],
    axis: RotationAxis,
    clockwise: bool = False,
) -> Float32[Tensor, "num_angles 3 3"]:
    """
    Create a 3D rotation matrix for a given angle and axis.

    Args:
        angles (Float32[Tensor, "num_angles"]): The angles of rotation in radians.
        axis (RotationAxis): The axis of rotation (X, Y, or Z).
        clockwise (bool): Whether the rotation is clockwise. Defaults to False.

    Returns:
        Float32[Tensor, "num_angles 3 3"]: A Nx3x3 rotation matrix.
    """
    rot_sin = torch.sin(angles)
    rot_cos = torch.cos(angles)
    ones = torch.ones_like(rot_cos)
    zeros = torch.zeros_like(rot_cos)

    if axis == RotationAxis.X:
        rot_mat_T = torch.stack(
            [
                torch.stack([ones, zeros, zeros]),
                torch.stack([zeros, rot_cos, -rot_sin]),
                torch.stack([zeros, rot_sin, rot_cos]),
            ]
        )
    elif axis == RotationAxis.Y:
        rot_mat_T = torch.stack(
            [
                torch.stack([rot_cos, zeros, rot_sin]),
                torch.stack([zeros, ones, zeros]),
                torch.stack([-rot_sin, zeros, rot_cos]),
            ]
        )
    elif axis == RotationAxis.Z:
        rot_mat_T = torch.stack(
            [
                torch.stack([rot_cos, -rot_sin, zeros]),
                torch.stack([rot_sin, rot_cos, zeros]),
                torch.stack([zeros, zeros, ones]),
            ]
        )
    else:
        raise ValueError(f"Axis should be one of {list(RotationAxis)}, but got {axis}")

    rot_mat_T = rot_mat_T.permute(2, 0, 1).contiguous()  # Change the shape to (num_angles, 3, 3)
    if clockwise:
        rot_mat_T = rot_mat_T.transpose(1, 2)
    return rot_mat_T


def rotate_points_3d(
    points: Float32[Tensor, "batch_size number_of_points 3"],
    rotation_matrices: Float32[Tensor, "batch_size 3 3"],
) -> Float32[Tensor, "batch_size number_of_points 3"]:
    """Rotate points by angles according to axis.

    Args:
        points (Float32[Tensor, "batch_size number_of_points 3"]): Points with shape (N, M, 3).
        rotation_matrices (Float32[Tensor, "batch_size 3 3"]): Rotation matrices to use.

    Returns:
        Float32[Tensor, "batch_size number_of_points 3"]: Rotated points with shape (N, M, 3).
    """
    if points.dim() != 3:
        raise ValueError(
            f"Points should be a 3D tensor with shape (Batch_size, number of points, 3), but got {points.shape}"
        )

    rotation_matrices = rotation_matrices.permute(
        1, 2, 0
    ).contiguous()  # Change shape to (3, 3, batch_size)
    # Apply the rotation matrix as R @ point (points are stored as rows, so contract
    # over the matrix column index j): out[a, i, k] = sum_j R_a[k, j] * point[a, i, j].
    points_new = torch.einsum("aij,kja->aik", points, rotation_matrices)
    return points_new


def surface_equ_3d(
    polygon_surfaces: Float32[Tensor, "num_polygon max_num_surfaces max_num_points_of_surface 3"],
) -> tuple[
    Float32[Tensor, "num_polygon max_num_surfaces 3"],
    Float32[Tensor, "num_polygon max_num_surfaces"],
]:
    """
    Calculate the surface equations [a, b, c], d in ax+by+cz+d=0 for a set of polygon surfaces
        in 3D space.

    The function is modified from:
    https://github.com/open-mmlab/mmdetection3d/blob/main/mmdet3d/structures/ops/box_np_ops.py
    Args:
        polygon_surfaces (Float32[Tensor, "num_polygon max_num_surfaces max_num_points_of_surface 3"]): Polygon surfaces with shape of
            [num_polygon, max_num_surfaces, max_num_points_of_surface, 3].
            All surfaces' normal vector must direct to internal.
            Max_num_points_of_surface must at least 3.

    Returns:
        tuple: normal vector and its direction.
    """
    # return [a, b, c], d in ax+by+cz+d=0
    # polygon_surfaces: [num_polygon, num_surfaces, num_points_of_polygon, 3]
    surface_vec = polygon_surfaces[:, :, :2, :] - polygon_surfaces[:, :, 1:3, :]
    normal_vec = torch.cross(surface_vec[:, :, 0, :], surface_vec[:, :, 1, :], dim=-1)
    d = torch.einsum("aij, aij->ai", normal_vec, polygon_surfaces[:, :, 0, :])
    return normal_vec, -d


def points_in_convex_polygon_3d(
    points: Float32[Tensor, "num_points 3"],
    polygon_surfaces: Float32[Tensor, "num_polygon max_num_surfaces max_num_points_of_surface 3"],
) -> Float32[Tensor, "num_polygon num_points"]:
    """
    Check if points are inside a convex polygon in 3D space and assign their corresponding polygon.

    Args:
        points (Float32[Tensor, "num_points 3"]): Points with shape of [num_points, 3].
        polygon_surfaces (Float32[Tensor, "num_polygon max_num_surfaces max_num_points_of_surface 3"]): Polygon surfaces with shape of
            [num_polygon, max_num_surfaces, max_num_points_of_surface, 3].
            All surfaces' normal vector must direct to internal.
            Max_num_points_of_surface must at least 3.

    Returns:
        Float32[Tensor, "num_polygon num_points"]: A tensor indicating whether each point is inside each polygon.
    """
    normal_vec, d = surface_equ_3d(polygon_surfaces)
    # ax + by + cz + d <= 0 inequality for points inside or on the surfaces
    inner_product = torch.einsum("aij, kj->aik", normal_vec, points) + d.unsqueeze(-1)
    return (inner_product <= 0).all(dim=1).float()


def points_in_rotated_box(coord: np.ndarray, box: np.ndarray) -> np.ndarray:
    """Boolean mask of the points inside one oriented 3D box.

    ``cz`` is the box's gravity center, so the height bound keeps ground from a lower
    level and returns from an object above out of the box.

    Args:
        coord: Point coordinates of shape ``(N, 3+)``, only xyz is read.
        box: Box row ``[cx, cy, cz, dx, dy, dz, yaw, ...]``.

    Returns:
        Boolean mask of shape ``(N,)``, True for a point inside the box.
    """
    center = np.asarray(box[:3], dtype=np.float64)
    half = np.asarray(box[3:6], dtype=np.float64) / 2.0
    yaw = float(box[6])
    offset = np.asarray(coord[:, :3], dtype=np.float64) - center
    cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)
    local_x = offset[:, 0] * cos_yaw + offset[:, 1] * sin_yaw
    local_y = -offset[:, 0] * sin_yaw + offset[:, 1] * cos_yaw
    return (
        (np.abs(local_x) <= half[0])
        & (np.abs(local_y) <= half[1])
        & (np.abs(offset[:, 2]) <= half[2])
    )
