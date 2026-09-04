"""Pure box geometry shared by the driving-aware detection metrics.

These helpers operate on boxes in the ego/base_link frame, laid out as
``[cx, cy, cz, dx, dy, dz, yaw, ...]`` (the same 7+ layout the matching code
validates). Everything here is stateless NumPy math so the metric components and
the matching core can share one definition of corners, corner displacement, and
the nearest box surface. Each definition exists once: the single-box functions
are the readable reference form, and the ``*_batch`` / paired forms are their
broadcast equivalents used on the matching hot path.
"""

from __future__ import annotations

from math import pi

import numpy as np


def wrap_angle(angle: float) -> float:
    """Wrap an angle to ``[-pi, pi]``.

    Args:
        angle: Angle in radians.

    Returns:
        The equivalent angle in ``[-pi, pi]``.
    """
    return (float(angle) + pi) % (2.0 * pi) - pi


def bev_corners(box: np.ndarray) -> np.ndarray:
    """Return the four BEV corners of a box as a ``(4, 2)`` array.

    Corners are ordered clockwise starting from the ``(+dx/2, +dy/2)`` corner
    in the box frame, then rotated by ``yaw`` and shifted to the center.

    Args:
        box: Box row ``[cx, cy, cz, dx, dy, dz, yaw, ...]``.

    Returns:
        Corner coordinates ``(4, 2)``.
    """
    center = box[:2].astype(np.float64)
    half = box[3:5].astype(np.float64) / 2.0
    yaw = float(box[6])
    local = np.array(
        [
            [half[0], half[1]],
            [half[0], -half[1]],
            [-half[0], -half[1]],
            [-half[0], half[1]],
        ],
        dtype=np.float64,
    )
    cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)
    rotation = np.array([[cos_yaw, -sin_yaw], [sin_yaw, cos_yaw]], dtype=np.float64)
    return local @ rotation.T + center


def corner_displacement(pred_box: np.ndarray, gt_box: np.ndarray) -> float:
    """Mean BEV corner distance under the best cyclic corner assignment.

    Couples position, size, and yaw into one distance in meters. The cyclic
    ``argmin`` over the four rotations keeps a box's 90-degree parameterization
    ambiguity from being punished as a gross error.

    Args:
        pred_box: Prediction box row ``[cx, cy, cz, dx, dy, dz, yaw, ...]``.
        gt_box: Ground-truth box row of the same layout.

    Returns:
        The mean corner distance in meters.
    """
    pred_corners = bev_corners(pred_box)
    gt_corners = bev_corners(gt_box)
    best = np.inf
    for shift in range(4):
        rolled = np.roll(pred_corners, shift, axis=0)
        mean_distance = float(np.mean(np.linalg.norm(rolled - gt_corners, axis=1)))
        best = min(best, mean_distance)
    return best


def nearest_surface_distance(box: np.ndarray) -> float:
    """Distance from the ego origin to the nearest point of the box's BEV outline.

    The ego origin is the frame origin ``(0, 0)``. Returns ``0.0`` when the origin
    lies inside the footprint (unphysical for a real object, but kept well defined).

    Args:
        box: Box row ``[cx, cy, cz, dx, dy, dz, yaw, ...]``.

    Returns:
        The distance in meters.
    """
    center = box[:2].astype(np.float64)
    half = box[3:5].astype(np.float64) / 2.0
    yaw = float(box[6])
    cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)
    # Express the origin in the box-local frame: rotate (-yaw) about the center.
    offset = -center
    local = np.array(
        [
            offset[0] * cos_yaw + offset[1] * sin_yaw,
            -offset[0] * sin_yaw + offset[1] * cos_yaw,
        ],
        dtype=np.float64,
    )
    clamped = np.clip(local, -half, half)
    return float(np.linalg.norm(local - clamped))


def signed_nearest_surface_error(pred_box: np.ndarray, gt_box: np.ndarray) -> float:
    """Signed near-face error ``d_pred - d_gt`` in meters.

    Positive means the predicted near face sits farther than the truth (ego would
    brake late), negative means nearer (over-caution).

    Args:
        pred_box: Prediction box row ``[cx, cy, cz, dx, dy, dz, yaw, ...]``.
        gt_box: Ground-truth box row of the same layout.

    Returns:
        The signed error in meters.
    """
    return nearest_surface_distance(pred_box) - nearest_surface_distance(gt_box)


def bev_corners_batch(boxes: np.ndarray) -> np.ndarray:
    """:func:`bev_corners` for ``(N, 7+)`` boxes at once, as ``(N, 4, 2)``.

    Args:
        boxes: Box rows ``(N, 7+)``.

    Returns:
        Corner coordinates ``(N, 4, 2)``.
    """
    boxes = np.asarray(boxes, dtype=np.float64)
    half_dx, half_dy = boxes[:, 3] / 2.0, boxes[:, 4] / 2.0
    # Same counter-clockwise corner order as bev_corners.
    local_x = np.stack([half_dx, half_dx, -half_dx, -half_dx], axis=1)
    local_y = np.stack([half_dy, -half_dy, -half_dy, half_dy], axis=1)
    cos_yaw = np.cos(boxes[:, 6])[:, None]
    sin_yaw = np.sin(boxes[:, 6])[:, None]
    corners_x = local_x * cos_yaw - local_y * sin_yaw + boxes[:, 0:1]
    corners_y = local_x * sin_yaw + local_y * cos_yaw + boxes[:, 1:2]
    return np.stack([corners_x, corners_y], axis=2)


def corner_displacements(pred_boxes: np.ndarray, gt_boxes: np.ndarray) -> np.ndarray:
    """:func:`corner_displacement` for paired boxes, ``(N, 7+)`` x ``(N, 7+)`` to ``(N,)``.

    Args:
        pred_boxes: Prediction box rows ``(N, 7+)``.
        gt_boxes: Ground-truth box rows paired row by row with the predictions.

    Returns:
        Mean corner distances ``(N,)`` in meters.
    """
    if pred_boxes.shape[0] == 0:
        return np.zeros((0,), dtype=np.float64)
    pred_corners = bev_corners_batch(pred_boxes)
    gt_corners = bev_corners_batch(gt_boxes)
    best = np.full(pred_boxes.shape[0], np.inf, dtype=np.float64)
    for shift in range(4):
        rolled = np.roll(pred_corners, shift, axis=1)
        mean_distance = np.mean(np.linalg.norm(rolled - gt_corners, axis=2), axis=1)
        best = np.minimum(best, mean_distance)
    return best


def corner_displacement_matrix(pred_boxes: np.ndarray, gt_boxes: np.ndarray) -> np.ndarray:
    """:func:`corner_displacement` for every pair, ``(P, 7+)`` x ``(G, 7+)`` to ``(P, G)``.

    Args:
        pred_boxes: Prediction box rows ``(P, 7+)``.
        gt_boxes: Ground-truth box rows ``(G, 7+)``.

    Returns:
        Mean corner distances ``(P, G)`` in meters.
    """
    if pred_boxes.shape[0] == 0 or gt_boxes.shape[0] == 0:
        return np.zeros((pred_boxes.shape[0], gt_boxes.shape[0]), dtype=np.float64)
    pred_corners = bev_corners_batch(pred_boxes)  # (P, 4, 2)
    gt_corners = bev_corners_batch(gt_boxes)  # (G, 4, 2)
    best = np.full((pred_boxes.shape[0], gt_boxes.shape[0]), np.inf, dtype=np.float64)
    for shift in range(4):
        rolled = np.roll(pred_corners, shift, axis=1)[:, None]  # (P, 1, 4, 2)
        mean_distance = np.mean(np.linalg.norm(rolled - gt_corners[None], axis=3), axis=2)
        best = np.minimum(best, mean_distance)
    return best


def nearest_surface_distances(boxes: np.ndarray) -> np.ndarray:
    """:func:`nearest_surface_distance` for ``(N, 7+)`` boxes at once, as ``(N,)``.

    Args:
        boxes: Box rows ``(N, 7+)``.

    Returns:
        Distances ``(N,)`` in meters.
    """
    boxes = np.asarray(boxes, dtype=np.float64)
    if boxes.shape[0] == 0:
        return np.zeros((0,), dtype=np.float64)
    half = boxes[:, 3:5] / 2.0
    cos_yaw, sin_yaw = np.cos(boxes[:, 6]), np.sin(boxes[:, 6])
    offset = -boxes[:, :2]
    local = np.stack(
        [
            offset[:, 0] * cos_yaw + offset[:, 1] * sin_yaw,
            -offset[:, 0] * sin_yaw + offset[:, 1] * cos_yaw,
        ],
        axis=1,
    )
    clamped = np.clip(local, -half, half)
    return np.linalg.norm(local - clamped, axis=1)
