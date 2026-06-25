"""Shared dense heatmap utilities for center-based 3D detection heads.

This module groups the Gaussian target drawing and center-distance NMS helpers
shared by CenterPoint-style dense heads.
"""

from __future__ import annotations

import math

import torch


def gaussian_radius(box_size: tuple[float, float], min_overlap: float = 0.1) -> int:
    """Compute the Gaussian radius used for dense heatmap supervision.

    Args:
        box_size: Box side lengths in feature-map cells. The formula is
            symmetric, so callers may pass either axis order consistently.
        min_overlap: Minimum Gaussian overlap with the target box.

    Returns:
        Integer Gaussian radius.
    """
    height, width = box_size
    a1 = 1
    b1 = height + width
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = math.sqrt(max(b1**2 - 4 * a1 * c1, 0.0))
    r1 = (b1 + sq1) / 2

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = math.sqrt(max(b2**2 - 4 * a2 * c2, 0.0))
    r2 = (b2 + sq2) / 2

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = math.sqrt(max(b3**2 - 4 * a3 * c3, 0.0))
    r3 = (b3 + sq3) / 2
    return int(min(r1, r2, r3))


def _gaussian2d(
    shape: tuple[int, int], sigma: float, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    """Create a 2D Gaussian kernel."""
    height, width = shape
    y, x = torch.meshgrid(
        torch.arange(-(height - 1) / 2, (height - 1) / 2 + 1, device=device, dtype=dtype),
        torch.arange(-(width - 1) / 2, (width - 1) / 2 + 1, device=device, dtype=dtype),
        indexing="ij",
    )
    gaussian = torch.exp(-(x * x + y * y) / (2 * sigma * sigma))
    gaussian[gaussian < torch.finfo(dtype).eps * gaussian.max()] = 0
    return gaussian


def draw_heatmap_gaussian(heatmap: torch.Tensor, center: tuple[int, int], radius: int) -> None:
    """Draw a Gaussian blob on a dense heatmap in place.

    Args:
        heatmap: Heatmap updated in place.
        center: Heatmap center as ``(x, y)``.
        radius: Gaussian radius in pixels.
    """
    diameter = 2 * radius + 1
    gaussian = _gaussian2d(
        (diameter, diameter),
        sigma=diameter / 6,
        device=heatmap.device,
        dtype=heatmap.dtype,
    )
    x_center, y_center = center
    height, width = heatmap.shape
    left, right = min(x_center, radius), min(width - x_center - 1, radius)
    top, bottom = min(y_center, radius), min(height - y_center - 1, radius)
    if left < 0 or right < 0 or top < 0 or bottom < 0:
        return
    masked_heatmap = heatmap[
        y_center - top : y_center + bottom + 1, x_center - left : x_center + right + 1
    ]
    masked_gaussian = gaussian[
        radius - top : radius + bottom + 1, radius - left : radius + right + 1
    ]
    torch.maximum(masked_heatmap, masked_gaussian, out=masked_heatmap)


def draw_heatmap_gaussian_oriented(
    heatmap: torch.Tensor,
    center: tuple[int, int],
    length_cells: float,
    width_cells: float,
    yaw: float,
    min_sigma: float = 1.0,
) -> None:
    """Draw an oriented elliptical Gaussian blob on a dense heatmap in place.

    Unlike :func:`draw_heatmap_gaussian`, the blob is stretched along the box
    length and rotated by ``yaw``. Elongated objects (for example a tractor and
    trailer rig) therefore receive a single connected positive region that
    covers the whole body, instead of a small round blob at the geometric
    center, which for a long rig falls in the low-density gap at the coupling.

    Args:
        heatmap: Heatmap updated in place.
        center: Heatmap center as ``(x, y)`` in cells.
        length_cells: Box length in heatmap cells (long axis).
        width_cells: Box width in heatmap cells (short axis).
        yaw: Box yaw in radians, measured from the heatmap x axis.
        min_sigma: Lower bound on each Gaussian sigma in cells. The default
            matches the effective sigma of a ``min_radius`` round blob.
    """
    sigma_length = max(length_cells / 6.0, min_sigma)
    sigma_width = max(width_cells / 6.0, min_sigma)
    radius = int(math.ceil(3.0 * max(sigma_length, sigma_width)))
    if radius < 1:
        return

    device = heatmap.device
    dtype = heatmap.dtype
    coords = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    offset_y, offset_x = torch.meshgrid(coords, coords, indexing="ij")
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    # Rotate grid offsets into the box frame: x_box is along the length axis.
    x_box = offset_x * cos_yaw + offset_y * sin_yaw
    y_box = -offset_x * sin_yaw + offset_y * cos_yaw
    gaussian = torch.exp(
        -(
            x_box * x_box / (2 * sigma_length * sigma_length)
            + y_box * y_box / (2 * sigma_width * sigma_width)
        )
    )
    gaussian[gaussian < torch.finfo(dtype).eps * gaussian.max()] = 0

    x_center, y_center = center
    height, width = heatmap.shape
    left, right = min(x_center, radius), min(width - x_center - 1, radius)
    top, bottom = min(y_center, radius), min(height - y_center - 1, radius)
    if left < 0 or right < 0 or top < 0 or bottom < 0:
        return
    masked_heatmap = heatmap[
        y_center - top : y_center + bottom + 1, x_center - left : x_center + right + 1
    ]
    masked_gaussian = gaussian[
        radius - top : radius + bottom + 1, radius - left : radius + right + 1
    ]
    torch.maximum(masked_heatmap, masked_gaussian, out=masked_heatmap)


def circle_nms(
    boxes: torch.Tensor, scores: torch.Tensor, min_radius: float, post_max_size: int
) -> torch.Tensor:
    """Apply class-wise center-distance NMS in the BEV plane.

    Args:
        boxes: Decoded boxes in metric space.
        scores: Confidence scores for the boxes.
        min_radius: Minimum center distance for suppression.
        post_max_size: Maximum number of boxes kept after suppression.

    Returns:
        Indices of boxes kept after suppression.
    """
    order = scores.argsort(descending=True)
    keep: list[int] = []
    centers = boxes[:, :2]
    while order.numel() > 0 and len(keep) < post_max_size:
        current = int(order[0].item())
        keep.append(current)
        if order.numel() == 1:
            break
        remaining = order[1:]
        distance = torch.norm(centers[remaining] - centers[current], dim=1)
        order = remaining[distance > min_radius]
    return scores.new_tensor(keep, dtype=torch.long)
