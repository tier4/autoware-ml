"""Shared dense heatmap utilities for center-based 3D detection heads.

This module groups the Gaussian target drawing and center-distance NMS helpers
shared by CenterPoint-style dense heads.
"""

from __future__ import annotations

from jaxtyping import Bool, Float32, Int32, Int64
import math

import torch


def batch_circle_nms(
    bboxes_centers: Float32[torch.Tensor, "batch_size num_classes max_num_boxes 2"],
    scores: Float32[torch.Tensor, "batch_size num_classes max_num_boxes"],
    min_radius: float,
    valid_bboxes_masks: Bool[torch.Tensor, "batch_size num_classes max_num_boxes"],
    post_max_size: int,
) -> Bool[torch.Tensor, "batch_size num_classes max_num_boxes"]:
    """
    Apply greedy center-distance NMS for each batch and classes in the BEV plane.
    This NMS checks only if two valid bboxes from the same classes heavily overlap by their
    L2 center distance without considering their box dimensions and orientations.
    Note that this NMS assumes bboxes from the same cluster/label share the same axis, for example,
    all vehicles from the first batch are in [0, 0, :, :]. Also, max_num_bboxes includes padded
    boxes for each batch and class.

    Args:
        bboxes_centers: Decoded box centers in metric space.
        scores: Confidence scores for the boxes.
        min_radius: Minimum center distance for suppression.
        valid_bboxes_masks: Boolean mask indicating which boxes are valid and should be considered for NMS.
        post_max_size: Maximum number of boxes kept after suppression, counted per class.

    Returns:
        Boolean mask of the boxes kept after suppression, aligned with the input order.
    """
    batch_size, num_classes, max_num_bboxes = scores.shape
    num_dimensions = bboxes_centers.shape[-1]

    # max_num_bboxes includes padded boxes for each batch and class.
    # (batch_size, num_classes, max_num_bboxes)
    orders = scores.argsort(dim=2, descending=True, stable=True)
    sorted_bboxes_valid_masks = torch.gather(valid_bboxes_masks, index=orders, dim=2)
    # (batch_size, num_classes, max_num_bboxes) -> (batch_size, num_classes, max_num_bboxes, 2)
    center_indices = orders.unsqueeze(-1).expand(-1, -1, -1, num_dimensions)
    sorted_bboxes_centers = torch.gather(bboxes_centers, index=center_indices, dim=2)

    # Pairwise center distances. compute_mode disables the matmul-based path so distances
    # are computed elementwise.
    # (batch_size, num_classes, max_num_bboxes, 2) -> (batch_size*num_classes, max_num_bboxes, 2) ->
    # (batch_size * num_classes, max_num_bboxes, max_num_bboxes) -> (batch_size, num_classes, max_num_bboxes, max_num_bboxes)
    flatten_bboxes_centers = sorted_bboxes_centers.reshape(
        batch_size * num_classes, max_num_bboxes, num_dimensions
    )
    pairwise_distances = torch.cdist(
        flatten_bboxes_centers,
        flatten_bboxes_centers,
        p=2.0,
        compute_mode="donot_use_mm_for_euclid_dist",
    ).view(batch_size, num_classes, max_num_bboxes, max_num_bboxes)

    # Keeps a box only when its center distance is strictly greater than min_radius.
    # (batch_size, num_classes, max_num_bboxes, max_num_bboxes)
    pairwise_suppression = pairwise_distances <= min_radius

    # Only a higher scoring box may suppress a lower scoring one. After the descending sort
    # that is exactly the strict upper triangle.
    # Setting triu(diagonal=1) ensures that a box does not suppress itself.
    # (max_num_bboxes, max_num_bboxes)
    higher_score_masks = torch.ones(
        (max_num_bboxes, max_num_bboxes), dtype=torch.bool, device=scores.device
    ).triu(diagonal=1)
    pairwise_suppression &= higher_score_masks.view(1, 1, max_num_bboxes, max_num_bboxes)

    # Greedy suppression is sequential by nature, because whether a box survives depends on
    # previous suppression results, for example, A -> B -> C, where A suppresses B and B suppresses
    # C, but A does not suppress C directly.
    # (batch_size, num_classes, max_num_bboxes)
    candidate_masks = sorted_bboxes_valid_masks.clone()
    sorted_keep_masks = torch.zeros_like(candidate_masks)

    # Loop over bboxes across batch and class dimensions.
    for rank in range(max_num_bboxes):
        # First, move the valid status of the current rank to the keep mask.
        # (batch_size, num_classes)
        kept_masks = candidate_masks[:, :, rank]
        sorted_keep_masks[:, :, rank] = kept_masks
        # Next, check if the current rank suppresses any bboxes.
        # (kept_masks.unsqueeze(-1) & pairwise_suppression[:, :, rank, :]) is only True when the
        # current rank is valid and it suppresses other bboxes.
        # Then, it negates the suppression result to mark the suppressed bboxes as invalid or valid
        # for the next rank.
        candidate_masks &= ~(kept_masks.unsqueeze(-1) & pairwise_suppression[:, :, rank, :])

    # Accumulate sum for each class to ensure that no more than post_max_size boxes
    # are kept per class.
    # (batch_size, num_classes, max_num_bboxes)
    sorted_keep_masks &= sorted_keep_masks.cumsum(dim=2) <= post_max_size

    # Inverse the order to get the original index of each bbox in the unsorted position.
    inverse_orders = orders.argsort(dim=2, stable=True)
    # (batch_size, num_classes, max_num_bboxes)
    keep_masks = torch.gather(sorted_keep_masks, index=inverse_orders, dim=2)
    return keep_masks.bool()


def vectorize_gaussian_radii(
    widths: Float32[torch.Tensor, "batch_size max_num_boxes"],
    heights: Float32[torch.Tensor, "batch_size max_num_boxes"],
    min_overlap: float = 0.1,
) -> Int32[torch.Tensor, "batch_size max_num_boxes"]:
    """
    Compute the Gaussian radius used for dense heatmap supervision in a vectorized manner
    The formula is symmetric, so callers may pass either axis for height and width order
    consistently.

    Args:
        widths: Box widths in meters.
        heights: Box heights in meters.
        min_overlap: Minimum Gaussian overlap with the target box.

    Returns:
        Gaussian (2D) radius in feature-map cells.
    """
    a1 = 1
    b1 = heights + widths
    c1 = widths * heights * (1 - min_overlap) / (1 + min_overlap)
    sq1 = torch.sqrt(torch.clamp(b1**2 - 4 * a1 * c1, min=0.0))
    r1 = (b1 + sq1) / 2

    a2 = 4
    b2 = 2 * (heights + widths)
    c2 = (1 - min_overlap) * widths * heights
    sq2 = torch.sqrt(torch.clamp(b2**2 - 4 * a2 * c2, min=0.0))
    r2 = (b2 + sq2) / 2

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (heights + widths)
    c3 = (min_overlap - 1) * widths * heights
    sq3 = torch.sqrt(torch.clamp(b3**2 - 4 * a3 * c3, min=0.0))
    r3 = (b3 + sq3) / 2
    return torch.minimum(torch.minimum(r1, r2), r3).to(torch.int32)


def _vectorize_gaussian2d(
    heights: Int32[torch.Tensor, "batch_size max_num_boxes"],
    widths: Int32[torch.Tensor, "batch_size max_num_boxes"],
    sigmas: Float32[torch.Tensor, "batch_size max_num_boxes"],
    valid_masks: Bool[torch.Tensor, "batch_size max_num_boxes"],
    device: torch.device,
    dtype: torch.dtype,
) -> Float32[torch.Tensor, "batch_size max_num_boxes max_height max_width"]:
    """
    Create a 2D Gaussian kernel based on the maximum of height and width over the batch and boxes.
    Padded with zeros for padded height and width values.
    """
    batch_size, max_num_boxes = heights.shape
    if max_num_boxes == 0:
        return torch.zeros((batch_size, 0, 0, 0), device=device, dtype=dtype)

    # Find the maximum height and width across the batch and boxes
    max_height = int(heights.max())
    max_width = int(widths.max())

    # (max_height)
    ys = torch.arange(max_height, device=device, dtype=dtype)
    # (max_width)
    xs = torch.arange(max_width, device=device, dtype=dtype)

    batch_heights = heights.view(batch_size, max_num_boxes, 1, 1)  # (B, M, 1, 1)
    batch_widths = widths.view(batch_size, max_num_boxes, 1, 1)  # (B, M, 1, 1)

    # per-box centered coordinates: y in [-(h-1)/2, (h-1)/2], same for x [-(w-1)/2, (w-1)/2]
    # (1, 1, max_height, 1) - (B, M, 1, 1) -> (B, M, max_height, 1)
    y = ys.view(1, 1, max_height, 1) - (batch_heights - 1) / 2  # (B, N, H, 1)
    # (1, 1, 1, max_width) - (B, M, 1, 1) -> (B, M, 1, max_width)
    x = xs.view(1, 1, 1, max_width) - (batch_widths - 1) / 2  # (B, N, 1, W)

    # Update sigmas to set invalid sigmas to 1.0 to avoid NaNs in the Gaussian computation
    # avoid in-place mutation of the sigmas tensor
    updated_sigmas = torch.where(valid_masks.bool(), sigmas, torch.ones_like(sigmas)).view(
        batch_size, max_num_boxes, 1, 1
    )

    # (B, M, max_height, 1) * (B, M, 1, max_width) -> (B, M, max_height, max_width)
    # (B, M, max_height, max_width) / (B, M) -> (B, M, max_height, max_width)
    # (B, N, max_height, max_width)
    gaussian = torch.exp(-(x * x + y * y) / (2 * updated_sigmas * updated_sigmas))

    # zero everything outside each box's own h×w region
    # (1, 1, max_height, 1) < (B, M, 1, 1) -> (B, M, max_height, 1) &
    # (1, 1, 1, max_width) < (B, M, 1, 1) -> (B, M, 1, max_width) ->
    # (B, M, max_height, max_width), where 1.0 means inside the box's own h×w region, 0.0 means outside
    inside_size = (ys.view(1, 1, max_height, 1) < batch_heights) & (
        xs.view(1, 1, 1, max_width) < batch_widths
    )
    gaussian = gaussian * inside_size

    # Set gaussian for invalid boxes to zero
    gaussian = gaussian * valid_masks.view(
        batch_size, max_num_boxes, 1, 1
    )  # (B, M, max_height, max_width)

    # threshold against each kernel's own max (not the global max)
    # (B, M, max_height, max_width) -> (B, M, max_height*max_width) -> (B, M) -> (B, M, 1, 1)
    kernel_max = gaussian.flatten(2).amax(dim=-1).view(batch_size, max_num_boxes, 1, 1)
    # Find out maximum value of the gaussian kernel for each box and threshold against it
    gaussian = torch.where(
        gaussian < torch.finfo(dtype).eps * kernel_max,
        torch.zeros_like(gaussian),
        gaussian,
    )
    return gaussian


def create_gaussian_heatmaps(
    heatmap_width: int,
    heatmap_height: int,
    num_classes: int,
    centers: Int64[torch.Tensor, "batch_size max_num_boxes 2"],
    gaussian_radii: Int32[torch.Tensor, "batch_size max_num_boxes"],
    gt_bboxes_labels: Int64[torch.Tensor, "batch_size max_num_boxes"],
    valid_masks: Bool[torch.Tensor, "batch_size max_num_boxes"],
    device: torch.device,
) -> Float32[torch.Tensor, "batch_size num_classes heatmap_height heatmap_width"]:
    """
    Create per-class heatmaps with Gaussian blobs for all valid boxes, fully vectorized
    in a batch.

    For each valid box, a 2D Gaussian kernel of size ``diameter = (2 * radius + 1)`` with
    ``sigma = diameter / 6`` is drawn onto the heatmap channel of its class label,
    centered at the given ``(x, y)`` center. Where blobs overlap (or a blob overlaps
    existing values), the element-wise maximum is kept, matching the standard max-splat.

    Algorithm:
      1. Batched kernel generation: ``_vectorize_gaussian2d`` builds all kernels
         at once, padded to a common shape
         ``(batch_size, max_num_boxes, max_diameter, max_diameter)``. Each kernel
         occupies the top-left ``diameter x diameter`` (diameter from each bbox)
         corner of its slot. The invalid bbox and padding are set to zero.
      2. Coordinate mapping: for every kernel cell, its global heatmap coordinate
         is computed as ``center - radius + cell_index``, broadcast to a
         ``(batch_size, max_num_boxes, max_diameter, max_diameter)`` grid of ``(yy, xx)`` positions.
      3. Masking: cells that fall outside the heatmap, or belong to an invalid
         (padded) box per ``valid_masks``, have their Gaussian value set to zero.
         Out-of-heatmap parts of a blob are therefore clipped per-cell, while the
         in-heatmap remainder is still drawn.
      4. Flat indexing: each cell's target position is linearized as
         ``label * H * W + y * W + x`` into a per-sample flat buffer of size
         ``num_classes * H * W``. Labels and coordinates are clamped into valid
         range so indexing is always legal; clamped cells carry value zero and
         cannot win the max.
      5. Scatter: a single ``scatter_reduce_(reduce="amax", include_self=True)``
         writes all cells for the whole batch, resolving overlaps between boxes
         and with pre-existing heatmap values by maximum.

    Notes:
        - Invalid boxes (``valid_masks == 0``) contribute nothing: their values
          are zeroed in step 3, and zeros never beat the (non-negative) heatmap
          under ``amax``. Padding labels such as ``-1`` are safe via clamping.
        - Memory scales with ``B * N * max_diameter**2``; a single large radius
          in the batch inflates the padded kernel tensor for all boxes. Only
          valid boxes count: radii of invalid (padded) boxes are normalized to
          zero first, so they never drive the kernel size.
        - A box whose center lies outside the heatmap is expected to be marked
          invalid in ``valid_masks`` (the caller filters these via a center-in-bounds check).
          Such boxes contribute nothing, matching the scalar reference. The per-cell ``in_bounds``
          clipping then only handles blob *tails* of valid boxes that extend past the
          heatmap border, which are clipped cell-by-cell.

    Args:
        heatmap_width: Width of the heatmap.
        heatmap_height: Height of the heatmap.
        num_classes: Number of classes.
        centers: Heatmap centers as ``(x, y)`` for each box.
        gaussian_radii: Gaussian radius in pixels for each box. Radii of boxes marked invalid in
            ``valid_masks`` are ignored, so callers need not pad them with any particular value.
        gt_bboxes_labels: Class labels for each bounding box.
        valid_masks: Mask indicating valid bounding boxes.
        device: Torch device.

    Returns:
        Heatmap tensor of shape ``(batch_size, num_classes, heatmap_height, heatmap_width)``.
    """
    batch_size = centers.shape[0]
    if batch_size != gaussian_radii.shape[0] or batch_size != gt_bboxes_labels.shape[0]:
        raise ValueError(
            "Batch size mismatch: centers, gaussian_radii, and gt_bboxes_labels must have the same batch size."
        )
    heatmaps = torch.zeros(
        (batch_size, num_classes, heatmap_height, heatmap_width), device=device, dtype=torch.float32
    )

    # _vectorize_gaussian2d sizes every kernel slot from the global maximum diameter,
    # so a single large padded radius would inflate the padded kernel
    # tensor for the whole batch (and a negative one could shrink it below the valid boxes' needs).
    # Normalize invalid radii to zero, the smallest safe radius, so only valid boxes drive the
    # kernel size. Invalid boxes still contribute nothing, because valid_masks zeroes them out.
    # (batch_size, max_num_boxes)
    normalized_gaussian_radii = torch.where(
        valid_masks.bool(), gaussian_radii, torch.zeros_like(gaussian_radii)
    )

    # (batch_size, max_num_boxes)
    diameters = 2 * normalized_gaussian_radii + 1

    # (batch_size, max_num_boxes, max_height, max_width)
    batch_gaussians_2d = _vectorize_gaussian2d(
        heights=diameters,
        widths=diameters,
        sigmas=diameters / 6.0,
        valid_masks=valid_masks,
        device=device,
        dtype=torch.float32,
    )

    # max_diameter == max_height == max_width since it uses the same diameters
    _, max_num_bboxes, max_diameter, _ = batch_gaussians_2d.shape

    # global pixel coordinates of every kernel cell
    # (0, 1, 2, ..., max_diameter-1)
    idx = torch.arange(max_diameter, device=device)  # (max_diameter,)

    # The normalized radii keep the coordinates of padded boxes in a sane range; their cells are
    # dropped by valid_masks in in_bounds regardless.
    # (B, max_num_bboxes, 1) - (B, max_num_bboxes, 1) + (1, 1, max_diameter) -> (B, N, max_diameter)
    # From [center_y - radius, center_y + radius], for each box, broadcast to all boxes in the batch
    ys = centers[..., 1].unsqueeze(-1) - normalized_gaussian_radii.unsqueeze(-1) + idx
    # From [center_x - radius, center_x + radius], for each box, broadcast to all boxes in the batch
    xs = centers[..., 0].unsqueeze(-1) - normalized_gaussian_radii.unsqueeze(-1) + idx

    # meshgrid to get all combinations of (y, x) for each box in the batch
    # (B, max_num_bboxes, max_diameter, 1) -> (B, max_num_bboxes, max_diameter, max_diameter)
    # All y-coordinates of the kernel cells for each box in the batch, broadcast to all x-coordinates
    yy = ys.unsqueeze(-1).expand(batch_size, max_num_bboxes, max_diameter, max_diameter)
    # (B, max_num_bboxes, max_diameter, 1) -> (B, max_num_bboxes, max_diameter, max_diameter)
    # All x-coordinates of the kernel cells for each box in the batch, broadcast to all y-coordinates
    xx = xs.unsqueeze(-2).expand(batch_size, max_num_bboxes, max_diameter, max_diameter)

    # (B, max_num_bboxes, max_diameter, max_diameter)
    in_bounds = (
        (yy >= 0)
        & (yy < heatmap_height)
        & (xx >= 0)
        & (xx < heatmap_width)
        & valid_masks.view(batch_size, max_num_bboxes, 1, 1)
    )

    # Set invalid heatmap positions to zero
    updated_batch_gaussian_2d = torch.where(
        in_bounds, batch_gaussians_2d, torch.zeros_like(batch_gaussians_2d)
    )

    # Labels for each box, shape (B, N, 1, 1)
    labels = gt_bboxes_labels.view(batch_size, max_num_bboxes, 1, 1)
    # Check if labels are more than num_classes, if so, raise an error
    if (labels >= num_classes).any():
        raise ValueError(
            f"Found label(s) >= num_classes ({num_classes}). "
            "Please ensure that all labels are in the range [0, num_classes - 1]."
        )

    # clamp invalid labels (e.g. -1 padding) so indexing stays legal;
    clamp_labels = labels.clamp(min=0)

    # (B, max_num_bboxes, 1, 1) + (B, max_num_bboxes, max_diameter, max_diameter) -> (B, max_num_bboxes, max_diameter, max_diameter)
    flat_idx = (
        clamp_labels * heatmap_height * heatmap_width
        + yy.clamp(0, heatmap_height - 1) * heatmap_width
        + xx.clamp(0, heatmap_width - 1)
    )

    # For the same pixel, it keeps the maximum value across all bboxes in the same batch,
    # so it uses scatter_reduce with "amax"
    # For invalid bboxes, they will not contribute to the heatmap since their valid_masks are False,
    # and thus their updated_batch_gaussian_2d values are zero.
    heatmaps.view(batch_size, -1).scatter_reduce_(
        dim=1,
        index=flat_idx.view(batch_size, -1),
        src=updated_batch_gaussian_2d.view(batch_size, -1),
        reduce="amax",
        include_self=True,
    )
    return heatmaps


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
