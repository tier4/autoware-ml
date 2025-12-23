# Copyright 2025 TIER IV, Inc.
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

from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import transforms3d
from scipy.stats import truncnorm

from autoware_ml.datamodule.t4dataset.calibration_status import (
    CalibrationData,
    CalibrationStatus,
)
from autoware_ml.transforms import BaseTransform


class CalibrationMisalignment(BaseTransform):
    """Calibration misalignment augmentation for camera-LiDAR calibration.

    Args:
        min_angle: Minimum rotation angle in degrees.
        max_angle: Maximum rotation angle in degrees.
        min_radius: Minimum translation radius in meters.
        max_radius: Maximum translation radius in meters.
        enable_random_sign: Whether to randomly flip translation direction.
        p: Probability of applying augmentation.
    """

    def __init__(
        self,
        min_angle: float = 1.0,
        max_angle: float = 10.0,
        min_radius: float = 0.05,
        max_radius: float = 0.2,
        enable_random_sign: bool = True,
        p: float = 0.5,
    ):
        super().__init__()
        self.min_angle = min_angle
        self.max_angle = max_angle
        self.min_radius = min_radius
        self.max_radius = max_radius
        self.enable_random_sign = enable_random_sign
        self.p = p

    def transform(self, input_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Apply calibration misalignment augmentation.

        Args:
            input_dict: Dictionary with 'calibration_data'.

        Returns:
            Dictionary with modified calibration data and is_miscalibrated flag.
        """
        assert "calibration_data" in input_dict, "Missing required key: 'calibration_data'"

        if np.random.rand() >= self.p:
            input_dict["gt_calibration_status"] = CalibrationStatus.CALIBRATED.value
            return input_dict

        calibration_data: CalibrationData = input_dict["calibration_data"]
        original_transform = calibration_data.lidar_to_camera_transformation
        noisy_transform = self.alter_calibration(original_transform)
        calibration_data.lidar_to_camera_transformation = noisy_transform
        input_dict["calibration_data"] = calibration_data
        input_dict["gt_calibration_status"] = CalibrationStatus.MISCALIBRATED.value

        return input_dict

    def bounded_gaussian(
        self, center: float, min_value: float, max_value: float, scale: float
    ) -> float:
        """Generate a value from a truncated normal distribution."""
        if min_value >= max_value:
            raise ValueError(f"min_value ({min_value}) must be less than max_value ({max_value})")
        if scale <= 0:
            raise ValueError(f"scale ({scale}) must be positive")

        a = (min_value - center) / scale
        b = (max_value - center) / scale
        return truncnorm.rvs(a, b, loc=center, scale=scale)

    def random_unit_sphere(self) -> np.ndarray:
        """Generate a random point on the unit sphere."""
        u1 = np.random.rand()
        u2 = np.random.rand()

        z = 2 * u1 - 1.0
        r = np.sqrt(1 - z * z)
        x = r * np.cos(2 * np.pi * u2)
        y = r * np.sin(2 * np.pi * u2)

        return np.array([x, y, z])

    def random_rotation_matrix(self, angle: float) -> np.ndarray:
        """Generate a random rotation matrix for a given angle in radians."""
        if angle < 0:
            raise ValueError(f"Angle must be non-negative, got {angle}")

        x, y, z = self.random_unit_sphere()
        w = np.sqrt(0.5 * (1.0 + np.cos(angle)))
        factor = np.sqrt(0.5 * (1.0 - np.cos(angle)))
        return transforms3d.quaternions.quat2mat((w, factor * x, factor * y, factor * z))

    def alter_calibration(self, transform: np.ndarray) -> np.ndarray:
        """Apply random noise to a 4x4 transformation matrix."""
        if transform.shape != (4, 4):
            raise ValueError(f"Transform must be 4x4 matrix, got shape {transform.shape}")

        noise_angle = self.bounded_gaussian(
            center=self.min_angle,
            min_value=self.min_angle,
            max_value=self.max_angle,
            scale=(self.max_angle - self.min_angle) / 1.5,
        )
        noise_radius = self.bounded_gaussian(
            center=self.min_radius,
            min_value=self.min_radius,
            max_value=self.max_radius,
            scale=(self.max_radius - self.min_radius) / 1.5,
        )

        if self.enable_random_sign and np.random.rand() > 0.5:
            noise_radius = -noise_radius

        noise_transform = np.eye(4)
        noise_transform[0:3, 0:3] = self.random_rotation_matrix(np.pi * abs(noise_angle) / 180.0)
        noise_transform[0:3, 3] = noise_radius * self.random_unit_sphere()

        return transform @ noise_transform


class LidarCameraFusion(BaseTransform):
    """Fuse LiDAR points with camera image to create depth and intensity channels.

    Projects LiDAR points onto the camera image plane and creates 5-channel
    fused images in BGRDI format (BGR + depth + intensity). The BGR format
    comes from cv2.imread which loads images in BGR order. Operates on single samples.

    Args:
        max_depth: Maximum depth for projected LiDAR points in meters.
        dilation_size: Size of dilation kernel for point cloud rendering.
    """

    def __init__(
        self,
        max_depth: float = 128.0,
        dilation_size: int = 1,
    ):
        super().__init__()
        self.max_depth = max_depth
        self.dilation_size = dilation_size

    def transform(self, input_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Create fused image from camera and LiDAR data.

        Args:
            input_dict: Dictionary with:
                - img: Image (H, W, 3) in BGR format (from cv2.imread)
                - points: Point cloud (N, 4+) [x, y, z, intensity, ...]
                - calibration_data: CalibrationData object
                - affine_transform (optional): 3x3 affine transformation matrix

        Returns:
            Dictionary with added 'fused_img': (H, W, 5) float32 [0, 1] in BGRDI format.
        """
        assert "img" in input_dict, "Missing required key: 'img'"
        assert "points" in input_dict, "Missing required key: 'points'"
        assert "calibration_data" in input_dict, "Missing required key: 'calibration_data'"

        image = input_dict["img"]
        points = input_dict["points"]
        calibration_data = input_dict["calibration_data"]
        affine_transform = input_dict.get("affine_transform")

        fused_img = self._create_fused_image(image, points, calibration_data, affine_transform)

        input_dict["fused_img"] = fused_img
        return input_dict

    def _create_fused_image(
        self,
        image: npt.NDArray[np.uint8],
        points: npt.NDArray[np.float32],
        calibration_data: CalibrationData,
        affine_transform: npt.NDArray[np.float64] = None,
    ) -> npt.NDArray[np.float32]:
        """Create fused image with RGB, depth, and intensity channels."""
        xyz = points[:, :3]
        intensities = points[:, 3]

        pointcloud_ccs = self._transform_points_to_camera(xyz, calibration_data)

        valid_mask = pointcloud_ccs[:, 2] > 0.0
        pointcloud_ccs = pointcloud_ccs[valid_mask]
        intensities = intensities[valid_mask]

        pointcloud_ics = self._project_points_to_image(pointcloud_ccs, calibration_data)

        if affine_transform is not None:
            pointcloud_ics = self._apply_affine_to_points(pointcloud_ics, affine_transform)

        return self._create_lidar_images(image, pointcloud_ics, pointcloud_ccs, intensities)

    def _transform_points_to_camera(
        self,
        points: npt.NDArray[np.float32],
        calibration_data: CalibrationData,
    ) -> npt.NDArray[np.float32]:
        """Transform LiDAR points to camera coordinate system."""
        num_points = points.shape[0]
        points_hom = np.concatenate([points, np.ones((num_points, 1), dtype=points.dtype)], axis=1)

        lidar2cam = calibration_data.lidar_to_camera_transformation
        points_cam = (lidar2cam @ points_hom.T).T

        return points_cam[:, :3]

    def _project_points_to_image(
        self,
        pointcloud_ccs: npt.NDArray[np.float32],
        calibration_data: CalibrationData,
    ) -> npt.NDArray[np.float32]:
        """Project 3D points to 2D image coordinates."""
        camera_matrix = calibration_data.new_camera_matrix
        distortion_coefficients = calibration_data.distortion_coefficients

        pointcloud_ics, _ = cv2.projectPoints(
            pointcloud_ccs,
            np.zeros(3),
            np.zeros(3),
            camera_matrix,
            distortion_coefficients,
        )
        if pointcloud_ics is None:
            return np.zeros((0, 2), dtype=np.float32)

        return pointcloud_ics.reshape(-1, 2)

    def _apply_affine_to_points(
        self,
        points_2d: npt.NDArray[np.float32],
        affine_matrix: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float32]:
        """Apply affine transformation to 2D points.

        Args:
            points_2d: 2D points in image coordinates (N, 2).
            affine_matrix: 3x3 affine transformation matrix.

        Returns:
            Transformed 2D points (N, 2).
        """
        num_points = points_2d.shape[0]
        homogeneous = np.hstack([points_2d, np.ones((num_points, 1))])
        transformed = (affine_matrix @ homogeneous.T).T[:, :2]
        return transformed.astype(np.float32)

    def _create_lidar_images(
        self,
        image: npt.NDArray[np.uint8],
        pointcloud_ics: npt.NDArray[np.float32],
        pointcloud_ccs: npt.NDArray[np.float32],
        intensities: npt.NDArray[np.float32],
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
            & (pointcloud_ccs[:, 2] < self.max_depth)
        )

        valid_ics = pointcloud_ics[valid_mask]
        valid_ccs = pointcloud_ccs[valid_mask]
        valid_intensities = intensities[valid_mask]

        if valid_ics.size > 0:
            y_offsets, x_offsets = np.mgrid[
                -self.dilation_size : self.dilation_size + 1,
                -self.dilation_size : self.dilation_size + 1,
            ]
            y_offsets = y_offsets.flatten()
            x_offsets = x_offsets.flatten()

            center_rows = valid_ics[:, 1].astype(np.int32)
            center_cols = valid_ics[:, 0].astype(np.int32)

            patch_rows = center_rows[:, np.newaxis] + y_offsets[np.newaxis, :]
            patch_cols = center_cols[:, np.newaxis] + x_offsets[np.newaxis, :]

            in_bounds_mask = (
                (patch_rows >= 0) & (patch_rows < h) & (patch_cols >= 0) & (patch_cols < w)
            )

            center_depths = 255 * valid_ccs[:, 2] / self.max_depth

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


class RandomAffine(BaseTransform):
    """Random affine transformation augmentation for images.

    Applies controlled affine distortion to the image and stores the affine
    matrix in input_dict for later application to projected LiDAR points.

    Args:
        p: Probability of applying augmentation.
        max_distortion: Maximum corner displacement as fraction of image size.
    """

    def __init__(self, p: float = 0.5, max_distortion: float = 0.1):
        super().__init__()
        self.p = p
        self.max_distortion = max_distortion

    def transform(self, input_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Apply random affine transformation to the image.

        Args:
            input_dict: Dictionary with 'img'.

        Returns:
            Dictionary with transformed image and 'affine_transform' matrix.
        """
        assert "img" in input_dict, "Missing required key: 'img'"

        if np.random.rand() >= self.p:
            return input_dict

        image: np.ndarray = input_dict["img"]

        h, w = image.shape[:2]

        max_offset_x = self.max_distortion * w
        max_offset_y = self.max_distortion * h

        src_pts = np.float32([[0, 0], [w - 1, 0], [0, h - 1]])
        dst_pts = src_pts + np.random.uniform(
            low=-np.array([[max_offset_x, max_offset_y]] * 3),
            high=np.array([[max_offset_x, max_offset_y]] * 3),
        ).astype(np.float32)

        affine_matrix_2x3 = cv2.getAffineTransform(src_pts, dst_pts)
        image = cv2.warpAffine(image, affine_matrix_2x3, (w, h), borderMode=cv2.BORDER_CONSTANT)

        affine_matrix_3x3 = np.eye(3, dtype=np.float64)
        affine_matrix_3x3[:2, :3] = affine_matrix_2x3

        input_dict["img"] = image
        input_dict["affine_transform"] = affine_matrix_3x3
        return input_dict


class SaveFusionPreview(BaseTransform):
    """Save preview images of fused RGB-LiDAR data for visualization.

    Creates two overlay images per sample:
    - RGB with depth points overlay using colormap
    - RGB with intensity points overlay using colormap

    Args:
        out_dir: Output directory for saving preview images.
        max_depth: Maximum depth value used during fusion (for recovery).
        alpha: Blending factor for overlay (0.0 = RGB only, 1.0 = overlay only).
        depth_colormap: Matplotlib colormap name for depth visualization.
        intensity_colormap: Matplotlib colormap name for intensity visualization.
    """

    def __init__(
        self,
        out_dir: str,
        max_depth: float = 128.0,
        alpha: float = 0.5,
        depth_colormap: str = "turbo",
        intensity_colormap: str = "jet",
    ):
        super().__init__()
        self.out_dir = Path(out_dir)
        self.max_depth = max_depth
        self.alpha = alpha
        self.depth_cmap = plt.get_cmap(depth_colormap)
        self.intensity_cmap = plt.get_cmap(intensity_colormap)

        self.out_dir.mkdir(parents=True, exist_ok=True)

    def transform(self, input_dict: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        """Save preview images for each sample in the batch.

        Args:
            input_dict: Dictionary containing:
                - fused_img: List of fused images (H, W, 5) float32 [0, 1] in BGRDI format
                - metadata: List of metadata dicts with image path info
                - gt_calibration_status: List of calibration status labels (0=calibrated, 1=miscalibrated)

        Returns:
            Unmodified input_dict (pass-through).
        """
        assert "fused_img" in input_dict, "Missing required key: 'fused_img'"
        assert "metadata" in input_dict, "Missing required key: 'metadata'"
        assert "gt_calibration_status" in input_dict, (
            "Missing required key: 'gt_calibration_status'"
        )

        fused_img = input_dict["fused_img"]
        metadata_dict = input_dict["metadata"]
        calibration_status = input_dict["gt_calibration_status"]

        self._save_preview(fused_img, metadata_dict, calibration_status)

        return input_dict

    def _save_preview(
        self,
        fused_img: npt.NDArray[np.float32],
        metadata: Dict[str, Any],
        calibration_status: int,
    ) -> None:
        """Save depth and intensity preview images for a single sample."""
        bgr, depth, intensity = self._recover_channels(fused_img)

        base_name = self._get_base_filename(metadata)
        status_suffix = (
            "calibrated"
            if calibration_status == CalibrationStatus.CALIBRATED.value
            else "miscalibrated"
        )

        # Convert BGR to RGB for matplotlib colormap processing
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        depth_overlay = self._create_overlay(rgb, depth, self.depth_cmap, self.alpha)
        intensity_overlay = self._create_overlay(rgb, intensity, self.intensity_cmap, self.alpha)

        depth_path = self.out_dir / f"{base_name}_{status_suffix}_depth.png"
        intensity_path = self.out_dir / f"{base_name}_{status_suffix}_intensity.png"

        # Convert RGB back to BGR for cv2.imwrite
        cv2.imwrite(str(depth_path), cv2.cvtColor(depth_overlay, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(intensity_path), cv2.cvtColor(intensity_overlay, cv2.COLOR_RGB2BGR))

    def _recover_channels(
        self, fused_img: npt.NDArray[np.float32]
    ) -> Tuple[npt.NDArray[np.uint8], npt.NDArray[np.float32], npt.NDArray[np.float32]]:
        """Recover original BGR, depth, and intensity values from fused image.

        Args:
            fused_img: Fused image (H, W, 5) with normalized values [0, 1] in BGRDI format.

        Returns:
            Tuple of (bgr, depth, intensity) with recovered values.
        """
        bgr = (fused_img[:, :, :3] * 255).astype(np.uint8)

        depth = fused_img[:, :, 3] * self.max_depth

        intensity = fused_img[:, :, 4] * 255

        return bgr, depth, intensity

    def _create_overlay(
        self,
        rgb: npt.NDArray[np.uint8],
        values: npt.NDArray[np.float32],
        cmap: plt.Colormap,
        alpha: float,
    ) -> npt.NDArray[np.uint8]:
        """Create alpha-blended overlay of RGB with colorized point values.

        Args:
            rgb: RGB image (H, W, 3) uint8.
            values: Value array (H, W) to colorize.
            cmap: Matplotlib colormap object.
            alpha: Blending factor for overlay points.

        Returns:
            Blended image (H, W, 3) uint8.
        """
        mask = values > 0

        max_val = values.max() if values.max() > 0 else 1.0
        normalized = values / max_val

        colored: npt.NDArray[np.uint8] = cmap(normalized)[:, :, :3]
        colored = (colored * 255).astype(np.uint8)

        result = rgb.copy()
        result[mask] = (
            (1 - alpha) * rgb[mask].astype(np.float32) + alpha * colored[mask].astype(np.float32)
        ).astype(np.uint8)

        return result

    def _get_base_filename(self, metadata: Dict[str, Any]) -> str:
        """Extract base filename from metadata.

        Args:
            metadata: Sample metadata dictionary.

        Returns:
            Base filename string without extension.
        """
        img_path = metadata["image"]["img_path"]
        return Path(img_path).stem
