"""
This script provides utility functions for geometric operations, particularly for 3D bounding boxes
and related transformations.
The code is modified from:
https://github.com/open-mmlab/mmdetection3d/blob/main/mmdet3d/structures/bbox_3d/utils.py
"""

from typing import Tuple

from jaxtyping import Float32
from torch import Tensor
import torch

from autoware_ml.types.spatial import RotationAxis


def rotation_3d_in_axis(
    points: Float32[Tensor, "batch_size number_of_points 3"],
    angle: float,
    axis: RotationAxis,
    clockwise: bool = False,
) -> Tuple[Float32[Tensor, "batch_size number_of_points 3"], Float32[Tensor, "3 3"]]:
    """Rotate points by angles according to axis.

    Args:
        points (Float32[Tensor, "batch_size number_of_points 3"]): Points with shape (N, M, 3).
        angle (float): Angle to rotate the points in the selected axis.
        axis (RotationAxis): The axis to be rotated. Defaults to RotationAxis.X.
        return_mat (bool): Whether or not to return the rotation matrix
            (transposed). Defaults to False.
        clockwise (bool): Whether the rotation is clockwise. Defaults to False.

    Returns:
        Tuple[Float32[Tensor, "batch_size number_of_points 3"], Float32[Tensor, "3 3"]]:
            Rotated points with shape (N, M, 3) and rotation matrix with shape (3, 3).
    """
    if points.dim != 3:
        raise ValueError(
            f"Points should be a 3D tensor with shape (Batch_size, number of points, 3), but got {points.shape}"
        )

    angles = torch.full(points.shape[:1], angle, dtype=points.dtype, device=points.device)
    rot_sin = torch.sin(angles)
    rot_cos = torch.cos(angles)
    ones = torch.ones_like(rot_cos)
    zeros = torch.zeros_like(rot_cos)

    if axis == RotationAxis.X:
        rot_mat_T = torch.stack(
            [
                torch.stack([rot_cos, zeros, -rot_sin]),
                torch.stack([zeros, ones, zeros]),
                torch.stack([rot_sin, zeros, rot_cos]),
            ]
        )
    elif axis == RotationAxis.Y:
        rot_mat_T = torch.stack(
            [
                torch.stack([ones, zeros, zeros]),
                torch.stack([zeros, rot_cos, rot_sin]),
                torch.stack([zeros, -rot_sin, rot_cos]),
            ]
        )
    elif axis == RotationAxis.Z:
        rot_mat_T = torch.stack(
            [
                torch.stack([rot_cos, rot_sin, zeros]),
                torch.stack([-rot_sin, rot_cos, zeros]),
                torch.stack([zeros, zeros, ones]),
            ]
        )
    else:
        raise ValueError(f"Axis should be one of {list(RotationAxis)}, but got {axis}")

    if clockwise:
        rot_mat_T = rot_mat_T.transpose(0, 1)

    points_new = torch.einsum("aij,jka->aik", points, rot_mat_T)

    return points_new, rot_mat_T
