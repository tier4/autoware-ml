import numpy as np
import numpy.typing as npt
import cv2


def project_depth_map(pts_c, fx, fy, cx, cy, w, h):
    """Projects points to a Z-buffer depth map."""
    z = pts_c[:, 2]
    valid = z > 0.1
    pts_c, z = pts_c[valid], z[valid]

    u = np.round((fx * pts_c[:, 0] / z) + cx).astype(int)
    v = np.round((fy * pts_c[:, 1] / z) + cy).astype(int)

    valid_uv = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    u, v, z = u[valid_uv], v[valid_uv], z[valid_uv]

    depth_img = np.zeros((h, w), dtype=np.float32)
    sort_idx = np.argsort(z)[::-1]  # Ensure closest points are drawn last
    depth_img[v[sort_idx], u[sort_idx]] = z[sort_idx]
    return depth_img


def project_pointcloud_to_image(
    image: npt.NDArray[np.uint8],
    pointcloud_ics: npt.NDArray[np.float32],
    pointcloud_ccs: npt.NDArray[np.float32],
    intensities: npt.NDArray[np.float32],
    max_depth: float,
    dilation_size: int,
) -> npt.NDArray[np.float32]:
    """Create fused image with depth and intensity channels.

    Returns:
        Fused image (H, W, 5) in BGRDI format, normalized to [0, 1].
    """
    h, w = image.shape[:2]
    depth_image = np.zeros((h, w), dtype=np.float32)
    intensity_image = np.zeros((h, w), dtype=np.float32)

    valid_mask = (
        (pointcloud_ics[:, 0] >= 0)
        & (pointcloud_ics[:, 0] <= w - 1)
        & (pointcloud_ics[:, 1] >= 0)
        & (pointcloud_ics[:, 1] <= h - 1)
        & (pointcloud_ccs[:, 2] > 0.0)
        & (pointcloud_ccs[:, 2] < max_depth)
    )

    valid_ics = pointcloud_ics[valid_mask]
    valid_ccs = pointcloud_ccs[valid_mask]
    valid_intensities = intensities[valid_mask]

    if valid_ics.size > 0:
        y_offsets, x_offsets = np.mgrid[
            -dilation_size : dilation_size + 1,
            -dilation_size : dilation_size + 1,
        ]
        y_offsets = y_offsets.flatten()
        x_offsets = x_offsets.flatten()

        center_rows = valid_ics[:, 1].astype(np.int32)
        center_cols = valid_ics[:, 0].astype(np.int32)

        patch_rows = center_rows[:, np.newaxis] + y_offsets[np.newaxis, :]
        patch_cols = center_cols[:, np.newaxis] + x_offsets[np.newaxis, :]

        in_bounds_mask = (patch_rows >= 0) & (patch_rows < h) & (patch_cols >= 0) & (patch_cols < w)

        center_depths = 255 * valid_ccs[:, 2] / max_depth

        broadcasted_depths = np.broadcast_to(center_depths[:, np.newaxis], patch_rows.shape)
        broadcasted_intensities = np.broadcast_to(
            valid_intensities[:, np.newaxis], patch_rows.shape
        )

        final_rows = patch_rows[in_bounds_mask]
        final_cols = patch_cols[in_bounds_mask]
        final_depths = broadcasted_depths[in_bounds_mask]
        final_intensities = broadcasted_intensities[in_bounds_mask]

        sort_indices = np.argsort(final_depths)[::-1]
        sorted_rows = final_rows[sort_indices]
        sorted_cols = final_cols[sort_indices]
        sorted_depths = final_depths[sort_indices]
        sorted_intensities = final_intensities[sort_indices]

        depth_image[sorted_rows, sorted_cols] = sorted_depths
        intensity_image[sorted_rows, sorted_cols] = sorted_intensities

    depth_image = np.expand_dims(depth_image, axis=2)
    intensity_image = np.expand_dims(intensity_image, axis=2)

    fused = np.concatenate([image, depth_image, intensity_image], axis=2)
    return fused.astype(np.float32) / 255.0


def project_spherical(points, w, h, dilation_size: int):
    """Projects points to a range and intensity image using spherical projection."""
    x, y, z_c = points[:, 0], points[:, 1], points[:, 2]
    r = np.linalg.norm(points[:, :3], axis=1)
    r_clip = np.clip(r, 1e-5, None)

    # RDF Frame: x=Right, y=Down, z=Forward
    azimuth = np.arctan2(x, z_c)
    elevation = np.arcsin(-y / r_clip)  # -y maps Up to positive values

    # Map angles to pixel coordinates
    u = np.round((azimuth / np.pi + 1.0) * 0.5 * (w - 1)).astype(int)
    v = np.round((1.0 - (elevation / (np.pi / 2))) * 0.5 * (h - 1)).astype(int)

    intensities = points[:, 3]

    range_img = np.zeros((h, w, 2), dtype=np.float32)

    # Use a mask to filter points that are initially within image bounds
    initial_valid_mask = (u >= 0) & (u < w) & (v >= 0) & (v < h)

    if np.any(initial_valid_mask):
        valid_u = u[initial_valid_mask]
        valid_v = v[initial_valid_mask]
        valid_r = r[initial_valid_mask]
        valid_intensities = intensities[initial_valid_mask]

        if dilation_size > 0:
            y_offsets, x_offsets = np.mgrid[
                -dilation_size : dilation_size + 1,
                -dilation_size : dilation_size + 1,
            ]
            y_offsets = y_offsets.flatten()
            x_offsets = x_offsets.flatten()

            center_rows = valid_v.astype(np.int32)
            center_cols = valid_u.astype(np.int32)

            patch_rows = center_rows[:, np.newaxis] + y_offsets[np.newaxis, :]
            patch_cols = center_cols[:, np.newaxis] + x_offsets[np.newaxis, :]

            in_bounds_mask = (
                (patch_rows >= 0) & (patch_rows < h) & (patch_cols >= 0) & (patch_cols < w)
            )

            # Broadcast range and intensity values to the new patch shape
            broadcasted_r = np.broadcast_to(valid_r[:, np.newaxis], patch_rows.shape)
            broadcasted_intensities = np.broadcast_to(
                valid_intensities[:, np.newaxis], patch_rows.shape
            )

            final_rows = patch_rows[in_bounds_mask]
            final_cols = patch_cols[in_bounds_mask]
            final_r = broadcasted_r[in_bounds_mask]
            final_intensities = broadcasted_intensities[in_bounds_mask]
        else:  # No dilation, just use initially valid points
            final_rows = valid_v
            final_cols = valid_u
            final_r = valid_r
            final_intensities = valid_intensities

        # Sort descending so closest points (smaller r) are drawn last
        sort_idx = np.argsort(final_r)[::-1]

        # Assign Range to channel 0, Intensity to channel 1
        range_img[final_rows[sort_idx], final_cols[sort_idx], 0] = final_r[sort_idx]
        range_img[final_rows[sort_idx], final_cols[sort_idx], 1] = final_intensities[sort_idx]

    return range_img


def get_4x4_matrix(t, r_deg):
    """Creates a 4x4 transformation matrix from translation and ZYX Euler rotations."""
    r, p, y = map(np.radians, r_deg)
    c_y, s_y = np.cos(y), np.sin(y)
    c_p, s_p = np.cos(p), np.sin(p)
    c_r, s_r = np.cos(r), np.sin(r)

    R_z = np.array([[c_y, -s_y, 0], [s_y, c_y, 0], [0, 0, 1]])
    R_y = np.array([[c_p, 0, s_p], [0, 1, 0], [-s_p, 0, c_p]])
    R_x = np.array([[1, 0, 0], [0, c_r, -s_r], [0, s_r, c_r]])
    R = R_z @ R_y @ R_x

    M = np.eye(4)
    M[:3, :3] = R
    M[:3, 3] = t
    return M


def create_lidar_image(
    h: int,
    w: int,
    camera_matrix,
    max_depth: float,
    dilation_size: int,
    points_xyz: npt.NDArray[np.float32],
    intensities: npt.NDArray[np.float32],
) -> npt.NDArray[np.float32]:
    """Project 3D points to 2D image with depth and intensity channels."""
    depth_img = np.zeros((h, w), dtype=np.float32)
    intensity_img = np.zeros((h, w), dtype=np.float32)

    if points_xyz.size == 0:
        return np.stack([depth_img, intensity_img], axis=-1)

    # Ensure points are float32/float64 and contiguous for OpenCV
    if points_xyz.dtype not in [np.float32, np.float64]:
        points_xyz = points_xyz.astype(np.float32)
    if not points_xyz.flags.c_contiguous:
        points_xyz = np.ascontiguousarray(points_xyz)

    # Project to image coordinates
    pointcloud_ics, _ = cv2.projectPoints(
        points_xyz,
        np.zeros(3, dtype=np.float32),
        np.zeros(3, dtype=np.float32),
        camera_matrix,
        np.zeros(5, dtype=np.float32),
    )
    pointcloud_ics = pointcloud_ics.reshape(-1, 2)

    valid_mask = (
        (pointcloud_ics[:, 0] >= 0)
        & (pointcloud_ics[:, 0] <= w - 1)
        & (pointcloud_ics[:, 1] >= 0)
        & (pointcloud_ics[:, 1] <= h - 1)
        # & (points_xyz[:, 2] > 0.0)  # Z > 0
        & (points_xyz[:, 2] < max_depth)
    )
    valid_ics = pointcloud_ics[valid_mask]
    valid_xyz = points_xyz[valid_mask]
    valid_intensities = intensities[valid_mask]

    if valid_ics.size > 0:
        y_offsets, x_offsets = np.mgrid[
            -dilation_size : dilation_size + 1,
            -dilation_size : dilation_size + 1,
        ]
        y_offsets = y_offsets.flatten()
        x_offsets = x_offsets.flatten()
        center_rows = valid_ics[:, 1].astype(np.int32)
        center_cols = valid_ics[:, 0].astype(np.int32)
        patch_rows = center_rows[:, np.newaxis] + y_offsets[np.newaxis, :]
        patch_cols = center_cols[:, np.newaxis] + x_offsets[np.newaxis, :]
        in_bounds_mask = (patch_rows >= 0) & (patch_rows < h) & (patch_cols >= 0) & (patch_cols < w)
        center_depths = valid_xyz[:, 2] / max_depth
        broadcasted_depths = np.broadcast_to(center_depths[:, np.newaxis], patch_rows.shape)
        broadcasted_intensities = np.broadcast_to(
            valid_intensities[:, np.newaxis], patch_rows.shape
        )
        final_rows = patch_rows[in_bounds_mask]
        final_cols = patch_cols[in_bounds_mask]
        final_depths = broadcasted_depths[in_bounds_mask]
        final_intensities = broadcasted_intensities[in_bounds_mask]
        # Use inverse depth for z-buffering (painter's algorithm)
        sort_indices = np.argsort(final_depths)[::-1]
        sorted_rows = final_rows[sort_indices]
        sorted_cols = final_cols[sort_indices]
        sorted_depths = final_depths[sort_indices]
        sorted_intensities = final_intensities[sort_indices]
        depth_img[sorted_rows, sorted_cols] = sorted_depths
        intensity_img[sorted_rows, sorted_cols] = sorted_intensities / 255.0
    return np.stack([depth_img, intensity_img], axis=2)
