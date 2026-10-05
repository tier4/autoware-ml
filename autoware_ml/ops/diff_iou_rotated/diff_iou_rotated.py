# Copyright (c) OpenMMLab. All rights reserved.
# Copyright 2026 TIER IV, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# The polygon construction (box2corners, box_intersection, box_in_box, build_vertices,
# sort_indices, calculate_area) and the IoU entry points are adapted from
# mmcv/ops/diff_iou_rotated.py, itself adapted from
# https://github.com/lilanxiao/Rotated_IoU/blob/master/box_intersection_2d.py and
# https://github.com/lilanxiao/Rotated_IoU/blob/master/oriented_iou_loss.py
#
# New in this port: drop_duplicate_vertices and GEOMETRY_TOLERANCE, the zero-union guards,
# the float32 enforcement under autocast, and the enclosing_box_smallest and convex_hull_area
# enclosing regions.

"""Differentiable IoU of rotated 2D and 3D boxes.

The intersection polygon of two rotated boxes is built from the box corners and the
edge-edge intersection points, its vertices are sorted counter-clockwise by a CUDA kernel,
and the area follows from the shoelace formula. Every step except the (non-differentiable)
vertex ordering is plain tensor arithmetic, so the IoU carries gradients to the box
parameters and can be used as a loss or matching cost.
"""

from __future__ import annotations

from enum import StrEnum

from jaxtyping import Bool, Float32, Int32, Int64
import torch
from torch.autograd import Function

from autoware_ml.types.geometry import Box3DFieldIndex

from . import diff_iou_rotated_ext

EPSILON = 1e-8
# Relative tolerance below which two edges count as parallel (sine of the angle between them)
# and an intersection counts as sitting on an edge endpoint. Both cases are already covered by
# the corner-in-box test, which uses the same tolerance, so flagging them again as
# intersections only adds duplicate vertices. Float noise on rotated corners otherwise produces
# them, and the intersection polygon then exceeds the 8 vertices two rectangles can share.
GEOMETRY_TOLERANCE = 1e-6


class EnclosingType(StrEnum):
    """Shape of the region enclosing two rotated boxes, from loosest to tightest.

    Attributes:
        ALIGNED: Axis-aligned box.
        SMALLEST: Minimum-area rotated box.
        CONVEX_HULL: Convex hull of the two boxes.
    """

    ALIGNED = "aligned"
    SMALLEST = "smallest"
    CONVEX_HULL = "convex_hull"


class SortVertices(Function):
    """Sort the valid vertices of every intersection polygon counter-clockwise."""

    @staticmethod
    def forward(
        ctx,
        vertices: Float32[torch.Tensor, "batch_size num_boxes 24 2"],
        mask: Bool[torch.Tensor, "batch_size num_boxes 24"],
    ) -> Int32[torch.Tensor, "batch_size num_boxes 25"]:
        """Run the CUDA sorting kernel.

        Args:
            ctx: Autograd context.
            vertices: Candidate vertices normalized around their mean, ``(B, N, 24, 2)``.
            mask: Validity mask of the candidates, ``(B, N, 24)``.

        Returns:
            Sorted vertex indices of shape ``(B, N, 25)``.
        """
        idx = diff_iou_rotated_ext.diff_iou_rotated_sort_vertices_forward(
            vertices.contiguous(), mask.contiguous()
        )
        ctx.mark_non_differentiable(idx)
        return idx

    @staticmethod
    def backward(ctx, gradout: Int32[torch.Tensor, "batch_size num_boxes 25"]) -> tuple[None, None]:
        """Indices carry no gradient: one ``None`` per forward input (vertices, mask)."""
        return None, None


def enclosing_box_aligned(
    corners1: Float32[torch.Tensor, "batch_size num_boxes 4 2"],
    corners2: Float32[torch.Tensor, "batch_size num_boxes 4 2"],
) -> Float32[torch.Tensor, "batch_size num_boxes"]:
    """Area of the axis-aligned smallest box enclosing two rotated boxes.

    The cheapest of the enclosing variants, but the loosest one: the box is
    not allowed to rotate with the two inputs.

    Args:
        corners1: ``(B, N, 4, 2)`` First batch of boxes.
        corners2: ``(B, N, 4, 2)`` Second batch of boxes.

    Returns:
        ``(B, N)`` Area of the enclosing box.
    """
    corners = torch.cat([corners1, corners2], dim=2)  # (B, N, 8, 2)
    wh = corners.max(dim=2)[0] - corners.min(dim=2)[0]  # (B, N, 2)
    return wh[..., 0] * wh[..., 1]


def enclosing_box_smallest(
    corners1: Float32[torch.Tensor, "batch_size num_boxes 4 2"],
    corners2: Float32[torch.Tensor, "batch_size num_boxes 4 2"],
) -> Float32[torch.Tensor, "batch_size num_boxes"]:
    """Area of the minimum-area (rotated) box enclosing two rotated boxes.

    A side of the minimum-area enclosing rectangle is always flush with an
    edge of the convex hull of the points. Every hull edge joins two of the 8
    corners, so evaluating the bounding rectangle for all 28 corner pairs
    covers the optimum without building the hull; the redundant candidates are
    valid enclosing rectangles too, hence the minimum over them is exact.

    Args:
        corners1: ``(B, N, 4, 2)`` First batch of boxes.
        corners2: ``(B, N, 4, 2)`` Second batch of boxes.

    Returns:
        ``(B, N)`` Area of the enclosing box.
    """
    corners = torch.cat([corners1, corners2], dim=2)  # (B, N, 8, 2)
    num = corners.size(2)
    i, j = torch.triu_indices(num, num, offset=1, device=corners.device)
    # candidate orientations, (B, N, 28, 2)
    edges = corners[:, :, j, :] - corners[:, :, i, :]
    edges = edges / torch.norm(edges, dim=-1, keepdim=True).clamp(min=EPSILON)
    # rotate the corners into each candidate frame instead of rotating the
    # frame: project on the direction and on its normal, (B, N, 28, 8)
    u, v = edges.unsqueeze(3).split([1, 1], dim=-1)
    x, y = corners.unsqueeze(2).split([1, 1], dim=-1)
    proj_u = (x * u + y * v).squeeze(-1)
    proj_v = (y * u - x * v).squeeze(-1)
    w = proj_u.max(dim=-1)[0] - proj_u.min(dim=-1)[0]  # (B, N, 28)
    h = proj_v.max(dim=-1)[0] - proj_v.min(dim=-1)[0]
    return (w * h).min(dim=-1)[0]


def convex_hull_area(
    corners1: Float32[torch.Tensor, "batch_size num_boxes 4 2"],
    corners2: Float32[torch.Tensor, "batch_size num_boxes 4 2"],
) -> Float32[torch.Tensor, "batch_size num_boxes"]:
    """Area of the convex hull of two rotated boxes.

    The tightest convex region containing both boxes, i.e. the smallest
    enclosing convex object of the original GIoU formulation.

    Note:
        A corner with minimal x always lies on the hull, and the hull is
        star-shaped with respect to it, so the hull vertices are met in
        increasing angular order around it. The convex hull is also the
        maximum-area polygon on the point set, so its area is the largest fan
        area over all sub-sequences of that angular order, which a small
        dynamic program finds exactly. Ties, duplicated and collinear corners
        need no special casing because a chain through an interior corner is
        never the maximum.

    Args:
        corners1: ``(B, N, 4, 2)`` First batch of boxes.
        corners2: ``(B, N, 4, 2)`` Second batch of boxes.

    Returns:
        ``(B, N)`` Area of the convex hull.
    """
    corners = torch.cat([corners1, corners2], dim=2)  # (B, N, 8, 2)
    num = corners.size(2)
    anchor_idx = corners[..., 0].argmin(dim=2)  # (B, N)
    anchor_idx = anchor_idx[..., None, None].expand(-1, -1, 1, 2)
    # corners relative to the anchor, (B, N, 8, 2)
    rel = corners - torch.gather(corners, 2, anchor_idx)
    # all corners lie in the half plane x >= 0, so the angles fall in
    # [-pi / 2, pi / 2] and sorting them needs no branch cut handling
    order = torch.atan2(rel[..., 1], rel[..., 0]).argsort(dim=2)
    rel = torch.gather(rel, 2, order.unsqueeze(-1).expand(-1, -1, -1, 2))
    x, y = rel.split([1, 1], dim=-1)
    # cross[..., i, j] is twice the area of the triangle (anchor, i, j) and is
    # non-negative for i < j, (B, N, 8, 8)
    cross = x * y.transpose(-1, -2) - y * x.transpose(-1, -2)
    # best[j]: largest fan area of a chain ending at the j-th corner
    best = [torch.zeros_like(cross[..., 0, 0])]
    for j in range(1, num):
        best.append((torch.stack(best, dim=-1) + cross[..., :j, j]).max(dim=-1)[0])
    return torch.stack(best, dim=-1).max(dim=-1)[0] / 2


def enclosing_area(
    corners1: Float32[torch.Tensor, "batch_size num_boxes 4 2"],
    corners2: Float32[torch.Tensor, "batch_size num_boxes 4 2"],
    enclosing_type: EnclosingType = EnclosingType.SMALLEST,
) -> Float32[torch.Tensor, "batch_size num_boxes"]:
    """Area of the region enclosing two rotated boxes.

    Args:
        corners1: ``(B, N, 4, 2)`` First batch of boxes.
        corners2: ``(B, N, 4, 2)`` Second batch of boxes.
        enclosing_type: Shape of the enclosing region, see :class:`EnclosingType`. Defaults
            to the minimum-area rotated box.

    Returns:
        ``(B, N)`` Area of the enclosing region.

    Raises:
        ValueError: If ``enclosing_type`` is not a member of :class:`EnclosingType`.
    """
    if enclosing_type == EnclosingType.ALIGNED:
        return enclosing_box_aligned(corners1, corners2)
    if enclosing_type == EnclosingType.SMALLEST:
        return enclosing_box_smallest(corners1, corners2)
    if enclosing_type == EnclosingType.CONVEX_HULL:
        return convex_hull_area(corners1, corners2)
    supported = ", ".join(member.value for member in EnclosingType)
    raise ValueError(f"Unknown enclosing type {enclosing_type}. Supported: {supported}")


def box_intersection(
    corners1: Float32[torch.Tensor, "batch_size num_boxes 4 2"],
    corners2: Float32[torch.Tensor, "batch_size num_boxes 4 2"],
) -> tuple[
    Float32[torch.Tensor, "batch_size num_boxes 4 4 2"],
    Bool[torch.Tensor, "batch_size num_boxes 4 4"],
]:
    """Find intersection points of rectangles.

    Convention: if two edges are (nearly) parallel there is no intersection point, and an
    intersection within ``GEOMETRY_TOLERANCE`` of an edge endpoint is not counted either, since
    that endpoint is a box corner already handled by :func:`box1_in_box2`.

    Args:
        corners1: ``(B, N, 4, 2)`` First batch of boxes.
        corners2: ``(B, N, 4, 2)`` Second batch of boxes.

    Returns:
        Tuple of ``(B, N, 4, 4, 2)`` intersections and the ``(B, N, 4, 4)`` valid mask.
    """
    # build edges from corners
    # B, N, 4, 4: Batch, Box, edge, point
    line1 = torch.cat([corners1, corners1[:, :, [1, 2, 3, 0], :]], dim=3)
    line2 = torch.cat([corners2, corners2[:, :, [1, 2, 3, 0], :]], dim=3)
    # duplicate data to pair each edges from the boxes
    # (B, N, 4, 4) -> (B, N, 4, 4, 4) : Batch, Box, edge1, edge2, point
    line1_ext = line1.unsqueeze(3)
    line2_ext = line2.unsqueeze(2)
    x1, y1, x2, y2 = line1_ext.split([1, 1, 1, 1], dim=-1)
    x3, y3, x4, y4 = line2_ext.split([1, 1, 1, 1], dim=-1)
    # math: https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection
    numerator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    # numerator is |d1| * |d2| * sin(angle between the edges), so scaling by the edge lengths
    # turns the tolerance into a bound on that sine. Exactly parallel edges give zero here.
    edge_lengths = torch.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2) * torch.sqrt(
        (x3 - x4) ** 2 + (y3 - y4) ** 2
    )
    parallel = numerator.abs() <= GEOMETRY_TOLERANCE * edge_lengths
    denumerator_t = (x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)
    t = denumerator_t / numerator
    t[parallel] = -1.0
    # intersection strictly inside line segment 1, away from its endpoints
    mask_t = (t > GEOMETRY_TOLERANCE) & (t < 1 - GEOMETRY_TOLERANCE)
    denumerator_u = (x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)
    u = -denumerator_u / numerator
    u[parallel] = -1.0
    # intersection strictly inside line segment 2, away from its endpoints
    mask_u = (u > GEOMETRY_TOLERANCE) & (u < 1 - GEOMETRY_TOLERANCE)
    mask = mask_t & mask_u
    # overwrite with EPSILON. otherwise numerically unstable
    t = denumerator_t / (numerator + EPSILON)
    intersections = torch.stack([x1 + t * (x2 - x1), y1 + t * (y2 - y1)], dim=-1)
    intersections = intersections * mask.float().unsqueeze(-1)
    return intersections, mask


def box1_in_box2(
    corners1: Float32[torch.Tensor, "batch_size num_boxes 4 2"],
    corners2: Float32[torch.Tensor, "batch_size num_boxes 4 2"],
) -> Bool[torch.Tensor, "batch_size num_boxes 4"]:
    """Check if corners of box1 lie in box2.

    Convention: if a corner is exactly on the edge of the other box, it's also a valid point.

    Args:
        corners1: ``(B, N, 4, 2)`` First batch of boxes.
        corners2: ``(B, N, 4, 2)`` Second batch of boxes.

    Returns:
        ``(B, N, 4)`` True where the i-th corner of box1 lies inside box2.
    """
    # a, b, c, d - 4 vertices of box2
    a = corners2[:, :, 0:1, :]  # (B, N, 1, 2)
    b = corners2[:, :, 1:2, :]  # (B, N, 1, 2)
    d = corners2[:, :, 3:4, :]  # (B, N, 1, 2)
    # ab, am, ad - vectors between corresponding vertices
    ab = b - a  # (B, N, 1, 2)
    am = corners1 - a  # (B, N, 4, 2)
    ad = d - a  # (B, N, 1, 2)
    prod_ab = torch.sum(ab * am, dim=-1)  # (B, N, 4)
    norm_ab = torch.sum(ab * ab, dim=-1)  # (B, N, 1)
    prod_ad = torch.sum(ad * am, dim=-1)  # (B, N, 4)
    norm_ad = torch.sum(ad * ad, dim=-1)  # (B, N, 1)
    # NOTE: the expression looks ugly but is stable if the two boxes
    # are exactly the same also stable with different scale of bboxes
    # Projections onto the two edges of box2, normalized to [0, 1] inside it. (B, N, 4)
    along_ab = prod_ab / norm_ab
    along_ad = prod_ad / norm_ad
    cond1 = (along_ab > -GEOMETRY_TOLERANCE) & (along_ab < 1 + GEOMETRY_TOLERANCE)
    cond2 = (along_ad > -GEOMETRY_TOLERANCE) & (along_ad < 1 + GEOMETRY_TOLERANCE)
    return cond1 & cond2


def box_in_box(
    corners1: Float32[torch.Tensor, "batch_size num_boxes 4 2"],
    corners2: Float32[torch.Tensor, "batch_size num_boxes 4 2"],
) -> tuple[
    Bool[torch.Tensor, "batch_size num_boxes 4"], Bool[torch.Tensor, "batch_size num_boxes 4"]
]:
    """Check if corners of two boxes lie in each other.

    Args:
        corners1: ``(B, N, 4, 2)`` First batch of boxes.
        corners2: ``(B, N, 4, 2)`` Second batch of boxes.

    Returns:
        Tuple of two ``(B, N, 4)`` masks: corners of box1 in box2, and corners of box2 in box1.
    """
    c1_in_2 = box1_in_box2(corners1, corners2)
    c2_in_1 = box1_in_box2(corners2, corners1)
    return c1_in_2, c2_in_1


def build_vertices(
    corners1: Float32[torch.Tensor, "batch_size num_boxes 4 2"],
    corners2: Float32[torch.Tensor, "batch_size num_boxes 4 2"],
    c1_in_2: Bool[torch.Tensor, "batch_size num_boxes 4"],
    c2_in_1: Bool[torch.Tensor, "batch_size num_boxes 4"],
    intersections: Float32[torch.Tensor, "batch_size num_boxes 4 4 2"],
    valid_mask: Bool[torch.Tensor, "batch_size num_boxes 4 4"],
) -> tuple[
    Float32[torch.Tensor, "batch_size num_boxes 24 2"],
    Bool[torch.Tensor, "batch_size num_boxes 24"],
]:
    """Find vertices of intersection area.

    Args:
        corners1: ``(B, N, 4, 2)`` First batch of boxes.
        corners2: ``(B, N, 4, 2)`` Second batch of boxes.
        c1_in_2: ``(B, N, 4)`` True if i-th corner of box1 is in box2.
        c2_in_1: ``(B, N, 4)`` True if i-th corner of box2 is in box1.
        intersections: ``(B, N, 4, 4, 2)`` Intersections.
        valid_mask: ``(B, N, 4, 4)`` Valid intersections mask.

    Returns:
        Tuple of ``(B, N, 24, 2)`` candidate vertices, of which only some are valid, and the
        ``(B, N, 24)`` mask of valid candidates.
    """
    # NOTE: inter has elements equals zero and has zeros gradient
    # (masked by multiplying with 0); can be used as trick
    batch_size = corners1.size()[0]
    num_boxes = corners1.size()[1]
    # The 4x4 edge pairs are flattened with an explicit size so an empty batch (no boxes) does
    # not hit the ambiguous ``-1`` of a zero-element view.
    num_intersections = intersections.size(2) * intersections.size(3)
    # (B, N, 4 + 4 + 16, 2)
    vertices = torch.cat(
        [corners1, corners2, intersections.view([batch_size, num_boxes, num_intersections, 2])],
        dim=2,
    )
    # Bool (B, N, 4 + 4 + 16)
    mask = torch.cat(
        [c1_in_2, c2_in_1, valid_mask.view([batch_size, num_boxes, num_intersections])], dim=2
    )
    return vertices, mask


@torch.no_grad()
def drop_duplicate_vertices(
    vertices: Float32[torch.Tensor, "batch_size num_boxes 24 2"],
    mask: Bool[torch.Tensor, "batch_size num_boxes 24"],
) -> Bool[torch.Tensor, "batch_size num_boxes 24"]:
    """Invalidate candidate vertices that coincide with an earlier valid candidate.

    Only a boolean mask comes out, so the pairwise distances are computed without autograd,
    which skips the graph bookkeeping for the ``(B, N, 24, 24)`` distance tensors.

    Two boxes that share a corner, or that are (nearly) identical, produce the same point
    several times: once per box corner and once more per edge pair meeting there. The sorting
    kernel cannot order equal or nearly equal points and expects every valid candidate to be
    distinct. Keeping only the first occurrence removes both exact duplicates and points a few
    float ulps apart. The candidate order is (corners of box1, corners of box2, intersections),
    so box1's corners are always the ones kept and gradients keep flowing to them.

    Args:
        vertices: ``(B, N, 24, 2)`` Candidate vertices.
        mask: ``(B, N, 24)`` Validity mask of the candidates.

    Returns:
        ``(B, N, 24)`` Mask with later duplicates of a valid candidate set to False.
    """
    # The tolerance scales with the boxes: the largest distance between any two candidate
    # corners of the pair, i.e. roughly the enclosing diagonal. (B, N, 1, 1)
    corners = vertices[:, :, :8, :]
    extent = torch.cdist(corners, corners).amax(dim=(-2, -1)).clamp(min=EPSILON)[..., None, None]
    # (B, N, 24, 24) pairwise distances, close[..., j, i] compares candidate j with candidate i
    distances = torch.cdist(vertices, vertices)
    close = distances <= GEOMETRY_TOLERANCE * extent
    # Only an earlier *valid* candidate i < j can shadow candidate j. Invalid slots hold zeros,
    # so without the validity check on i they would swallow a real vertex at the origin.
    num_candidates = vertices.shape[2]
    earlier = torch.tril(
        torch.ones(num_candidates, num_candidates, dtype=torch.bool, device=vertices.device),
        diagonal=-1,
    )
    shadowed = (close & earlier & mask.unsqueeze(-2)).any(dim=-1)
    return mask & ~shadowed


def sort_indices(
    vertices: Float32[torch.Tensor, "batch_size num_boxes 24 2"],
    mask: Bool[torch.Tensor, "batch_size num_boxes 24"],
) -> Int64[torch.Tensor, "batch_size num_boxes 25"]:
    """Sort indices.

    Note:
        The row has the structure (A, B, C, ..., A, X, X, X): the valid vertices sorted
        counter-clockwise, the first one repeated to close the polygon, then X, the index of
        an arbitrary invalid intersection candidate, whose value and gradient are zero.

        Why 25 slots when two rectangles share at most 8 polygon vertices? The candidates are
        classified with float tolerances, and near-degenerate pairs (nearly identical boxes
        far from the origin, a box inscribed in the other) can leave more than 8 of the 24
        candidates valid. The row has room for all of them plus the closing duplicate, so no
        vertex is ever dropped; the extra ones sit on slivers of the polygon and add
        negligible area.

    Args:
        vertices: ``(B, N, 24, 2)`` Box vertices.
        mask: ``(B, N, 24)`` Mask.

    Returns:
        ``(B, N, 25)`` Sorted indices.
    """
    # Disjoint boxes have no valid candidate; the clamp keeps their (unused) mean finite.
    num_valid = torch.sum(mask.int(), dim=2, keepdim=True).unsqueeze(-1)  # (B, N, 1, 1)
    mean = torch.sum(vertices * mask.float().unsqueeze(-1), dim=2, keepdim=True) / num_valid.clamp(
        min=1
    )
    vertices_normalized = vertices - mean  # normalization makes sorting easier
    return SortVertices.apply(vertices_normalized, mask).long()


def calculate_area(
    idx_sorted: Int64[torch.Tensor, "batch_size num_boxes 25"],
    vertices: Float32[torch.Tensor, "batch_size num_boxes 24 2"],
) -> tuple[
    Float32[torch.Tensor, "batch_size num_boxes"],
    Float32[torch.Tensor, "batch_size num_boxes 25 2"],
]:
    """Calculate area of intersection.

    Args:
        idx_sorted: ``(B, N, 25)`` Sorted vertex ids.
        vertices: ``(B, N, 24, 2)`` Vertices.

    Returns:
        Tuple of the ``(B, N)`` intersection area and the ``(B, N, 25, 2)`` polygon vertices
        with zero padding.
    """
    idx_ext = idx_sorted.unsqueeze(-1).repeat([1, 1, 1, 2])
    selected = torch.gather(vertices, 2, idx_ext)
    total = (
        selected[:, :, 0:-1, 0] * selected[:, :, 1:, 1]
        - selected[:, :, 0:-1, 1] * selected[:, :, 1:, 0]
    )
    total = torch.sum(total, dim=2)
    area = torch.abs(total) / 2
    return area, selected


def oriented_box_intersection_2d(
    corners1: Float32[torch.Tensor, "batch_size num_boxes 4 2"],
    corners2: Float32[torch.Tensor, "batch_size num_boxes 4 2"],
) -> tuple[
    Float32[torch.Tensor, "batch_size num_boxes"],
    Float32[torch.Tensor, "batch_size num_boxes 25 2"],
]:
    """Calculate intersection area of 2d rotated boxes.

    Args:
        corners1: ``(B, N, 4, 2)`` First batch of boxes.
        corners2: ``(B, N, 4, 2)`` Second batch of boxes.

    Returns:
        Tuple of the ``(B, N)`` intersection area and the ``(B, N, 25, 2)`` polygon vertices
        with zero padding.
    """
    corners1, corners2 = corners1.float(), corners2.float()
    with torch.autocast(device_type=corners1.device.type, enabled=False):
        intersections, valid_mask = box_intersection(corners1, corners2)
        c12, c21 = box_in_box(corners1, corners2)
        vertices, mask = build_vertices(corners1, corners2, c12, c21, intersections, valid_mask)
        mask = drop_duplicate_vertices(vertices, mask)
        # The sorting kernel pads every row with an invalid intersection candidate and relies on
        # its value being zero, so the padding adds nothing to the shoelace sum. Candidates dropped
        # as duplicates still hold their coordinates, so zero every invalid candidate explicitly.
        vertices = vertices * mask.unsqueeze(-1)
        sorted_indices = sort_indices(vertices, mask)
        return calculate_area(sorted_indices, vertices)


def box2corners(
    box: Float32[torch.Tensor, "batch_size num_boxes 5"],
) -> Float32[torch.Tensor, "batch_size num_boxes 4 2"]:
    """Convert rotated 2d box coordinate to corners.

    Args:
        box: ``(B, N, 5)`` with (x, y, length, width, yaw); the length lies along the yaw
            direction and the width across it.

    Returns:
        ``(B, N, 4, 2)`` Corners.
    """
    # The geometry downstream compares coordinates against GEOMETRY_TOLERANCE and the sorting
    # kernel only accepts float32, so half precision boxes are upcast and autocast is kept
    # from demoting the rotation matmul. Gradients reach the original tensor through the cast.
    box = box.float()
    with torch.autocast(device_type=box.device.type, enabled=False):
        batch_size, num_boxes = box.size()[0], box.size()[1]
        x, y, length, width, yaw = box.split([1, 1, 1, 1, 1], dim=-1)
        x4 = box.new_tensor([0.5, -0.5, -0.5, 0.5])
        x4 = x4 * length  # (B, N, 4)
        y4 = box.new_tensor([0.5, 0.5, -0.5, -0.5])
        y4 = y4 * width  # (B, N, 4)
        corners = torch.stack([x4, y4], dim=-1)  # (B, N, 4, 2)
        sin = torch.sin(yaw)
        cos = torch.cos(yaw)
        row1 = torch.cat([cos, sin], dim=-1)
        row2 = torch.cat([-sin, cos], dim=-1)  # (B, N, 2)
        rot_t = torch.stack([row1, row2], dim=-2)  # (B, N, 2, 2)
        rotated = torch.bmm(corners.view([-1, 4, 2]), rot_t.view([-1, 2, 2]))
        rotated = rotated.view([batch_size, num_boxes, 4, 2])  # (B * N, 4, 2) -> (B, N, 4, 2)
        rotated[..., 0] += x
        rotated[..., 1] += y
        return rotated


def diff_iou_rotated_2d(
    box1: Float32[torch.Tensor, "batch_size num_boxes 5"],
    box2: Float32[torch.Tensor, "batch_size num_boxes 5"],
) -> Float32[torch.Tensor, "batch_size num_boxes"]:
    """Calculate differentiable iou of rotated 2d boxes.

    Args:
        box1: ``(B, N, 5)`` First box as (x, y, length, width, yaw).
        box2: ``(B, N, 5)`` Second box as (x, y, length, width, yaw).

    Returns:
        ``(B, N)`` IoU, float32 whatever the input dtype.
    """
    box1, box2 = box1.float(), box2.float()
    with torch.autocast(device_type=box1.device.type, enabled=False):
        corners1 = box2corners(box1)
        corners2 = box2corners(box2)
        intersection, _ = oriented_box_intersection_2d(corners1, corners2)  # (B, N)
        area1 = box1[:, :, 2] * box1[:, :, 3]
        area2 = box2[:, :, 2] * box2[:, :, 3]
        # A zero-size box with no overlap has a zero union; the clamp turns 0 / 0 into an IoU of 0
        # with a finite gradient instead of NaN. Callers must still provide non-negative sizes.
        union = (area1 + area2 - intersection).clamp(min=EPSILON)
        iou = intersection / union
        return iou


def diff_iou_rotated_3d(
    box3d1: Float32[torch.Tensor, "batch_size num_boxes 7"],
    box3d2: Float32[torch.Tensor, "batch_size num_boxes 7"],
) -> Float32[torch.Tensor, "batch_size num_boxes"]:
    """Calculate differentiable iou of rotated 3d boxes.

    Args:
        box3d1: ``(B, N, 7)`` First box as (x, y, z, length, width, height, yaw), the
            :class:`~autoware_ml.types.geometry.Box3DFieldIndex` layout.
        box3d2: ``(B, N, 7)`` Second box in the same layout.

    Returns:
        ``(B, N)`` IoU, float32 whatever the input dtype.
    """
    box3d1, box3d2 = box3d1.float(), box3d2.float()
    with torch.autocast(device_type=box3d1.device.type, enabled=False):
        bev_fields = [
            Box3DFieldIndex.X,
            Box3DFieldIndex.Y,
            Box3DFieldIndex.LENGTH,
            Box3DFieldIndex.WIDTH,
            Box3DFieldIndex.YAW,
        ]
        corners1 = box2corners(box3d1[..., bev_fields])
        corners2 = box2corners(box3d2[..., bev_fields])
        intersection, _ = oriented_box_intersection_2d(corners1, corners2)
        z1, height1 = box3d1[..., Box3DFieldIndex.Z], box3d1[..., Box3DFieldIndex.HEIGHT]
        z2, height2 = box3d2[..., Box3DFieldIndex.Z], box3d2[..., Box3DFieldIndex.HEIGHT]
        z_overlap = (
            torch.min(z1 + height1 * 0.5, z2 + height2 * 0.5)
            - torch.max(z1 - height1 * 0.5, z2 - height2 * 0.5)
        ).clamp_(min=0.0)
        intersection_3d = intersection * z_overlap
        volume1 = box3d1[..., Box3DFieldIndex.LENGTH] * box3d1[..., Box3DFieldIndex.WIDTH] * height1
        volume2 = box3d2[..., Box3DFieldIndex.LENGTH] * box3d2[..., Box3DFieldIndex.WIDTH] * height2
        # Same guard as in :func:`diff_iou_rotated_2d`: a zero-volume box with no overlap.
        union_3d = (volume1 + volume2 - intersection_3d).clamp(min=EPSILON)
        return intersection_3d / union_3d
