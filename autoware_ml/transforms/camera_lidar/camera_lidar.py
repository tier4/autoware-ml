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

    Each rotation (roll, pitch, yaw) and translation (x, y, z) component has
    separate negative and positive ranges. During augmentation, one of the two
    ranges is randomly selected for each component. Each component can be
    individually activated or deactivated.

    All parameters are specified as positive magnitudes. The `_neg` suffix
    indicates the value will be negated when applied. This keeps min < max
    intuitive in the config.

    Required keys:
        - calibration_data: CalibrationData object with lidar_to_camera_transformation.

    Optional keys:
        - None

    Generated keys:
        - gt_calibration_status: int (CalibrationStatus.CALIBRATED or MISCALIBRATED).
        - calibration_data.noise: 4x4 noise transform matrix (when augmentation applied).
        - calibration_data.lidar_to_camera_transformation: Modified with noise (when applied).

    Args:
        p: Probability of applying augmentation.
        activate_roll: Whether to apply roll miscalibration.
        activate_pitch: Whether to apply pitch miscalibration.
        activate_yaw: Whether to apply yaw miscalibration.
        activate_x: Whether to apply x translation miscalibration.
        activate_y: Whether to apply y translation miscalibration.
        activate_z: Whether to apply z translation miscalibration.
        min_roll_neg: Min magnitude for negative roll in degrees (applied as negative).
        max_roll_neg: Max magnitude for negative roll in degrees (applied as negative).
        min_roll_pos: Min magnitude for positive roll in degrees.
        max_roll_pos: Max magnitude for positive roll in degrees.
        min_pitch_neg: Min magnitude for negative pitch in degrees (applied as negative).
        max_pitch_neg: Max magnitude for negative pitch in degrees (applied as negative).
        min_pitch_pos: Min magnitude for positive pitch in degrees.
        max_pitch_pos: Max magnitude for positive pitch in degrees.
        min_yaw_neg: Min magnitude for negative yaw in degrees (applied as negative).
        max_yaw_neg: Max magnitude for negative yaw in degrees (applied as negative).
        min_yaw_pos: Min magnitude for positive yaw in degrees.
        max_yaw_pos: Max magnitude for positive yaw in degrees.
        min_x_neg: Min magnitude for negative x translation in meters (applied as negative).
        max_x_neg: Max magnitude for negative x translation in meters (applied as negative).
        min_x_pos: Min magnitude for positive x translation in meters.
        max_x_pos: Max magnitude for positive x translation in meters.
        min_y_neg: Min magnitude for negative y translation in meters (applied as negative).
        max_y_neg: Max magnitude for negative y translation in meters (applied as negative).
        min_y_pos: Min magnitude for positive y translation in meters.
        max_y_pos: Max magnitude for positive y translation in meters.
        min_z_neg: Min magnitude for negative z translation in meters (applied as negative).
        max_z_neg: Max magnitude for negative z translation in meters (applied as negative).
        min_z_pos: Min magnitude for positive z translation in meters.
        max_z_pos: Max magnitude for positive z translation in meters.
    """

    _required_keys = ["calibration_data"]

    def __init__(
        self,
        p: float,
        activate_roll: bool = False,
        activate_pitch: bool = False,
        activate_yaw: bool = False,
        activate_x: bool = False,
        activate_y: bool = False,
        activate_z: bool = False,
        min_roll_neg: float = 0.0,
        max_roll_neg: float = 0.0,
        min_roll_pos: float = 0.0,
        max_roll_pos: float = 0.0,
        min_pitch_neg: float = 0.0,
        max_pitch_neg: float = 0.0,
        min_pitch_pos: float = 0.0,
        max_pitch_pos: float = 0.0,
        min_yaw_neg: float = 0.0,
        max_yaw_neg: float = 0.0,
        min_yaw_pos: float = 0.0,
        max_yaw_pos: float = 0.0,
        min_x_neg: float = 0.0,
        max_x_neg: float = 0.0,
        min_x_pos: float = 0.0,
        max_x_pos: float = 0.0,
        min_y_neg: float = 0.0,
        max_y_neg: float = 0.0,
        min_y_pos: float = 0.0,
        max_y_pos: float = 0.0,
        min_z_neg: float = 0.0,
        max_z_neg: float = 0.0,
        min_z_pos: float = 0.0,
        max_z_pos: float = 0.0,
    ):
        super().__init__()

        self.activate_roll = activate_roll
        self.activate_pitch = activate_pitch
        self.activate_yaw = activate_yaw
        self.activate_x = activate_x
        self.activate_y = activate_y
        self.activate_z = activate_z

        # Validate all parameters are >= 0 (magnitudes)
        for name, value in [
            ("min_roll_neg", min_roll_neg),
            ("max_roll_neg", max_roll_neg),
            ("min_roll_pos", min_roll_pos),
            ("max_roll_pos", max_roll_pos),
            ("min_pitch_neg", min_pitch_neg),
            ("max_pitch_neg", max_pitch_neg),
            ("min_pitch_pos", min_pitch_pos),
            ("max_pitch_pos", max_pitch_pos),
            ("min_yaw_neg", min_yaw_neg),
            ("max_yaw_neg", max_yaw_neg),
            ("min_yaw_pos", min_yaw_pos),
            ("max_yaw_pos", max_yaw_pos),
            ("min_x_neg", min_x_neg),
            ("max_x_neg", max_x_neg),
            ("min_x_pos", min_x_pos),
            ("max_x_pos", max_x_pos),
            ("min_y_neg", min_y_neg),
            ("max_y_neg", max_y_neg),
            ("min_y_pos", min_y_pos),
            ("max_y_pos", max_y_pos),
            ("min_z_neg", min_z_neg),
            ("max_z_neg", max_z_neg),
            ("min_z_pos", min_z_pos),
            ("max_z_pos", max_z_pos),
        ]:
            self._validate_non_negative(name, value)

        # Validate min <= max for each range
        self._validate_range("roll_neg", min_roll_neg, max_roll_neg)
        self._validate_range("roll_pos", min_roll_pos, max_roll_pos)
        self._validate_range("pitch_neg", min_pitch_neg, max_pitch_neg)
        self._validate_range("pitch_pos", min_pitch_pos, max_pitch_pos)
        self._validate_range("yaw_neg", min_yaw_neg, max_yaw_neg)
        self._validate_range("yaw_pos", min_yaw_pos, max_yaw_pos)
        self._validate_range("x_neg", min_x_neg, max_x_neg)
        self._validate_range("x_pos", min_x_pos, max_x_pos)
        self._validate_range("y_neg", min_y_neg, max_y_neg)
        self._validate_range("y_pos", min_y_pos, max_y_pos)
        self._validate_range("z_neg", min_z_neg, max_z_neg)
        self._validate_range("z_pos", min_z_pos, max_z_pos)

        self.min_roll_neg = min_roll_neg
        self.max_roll_neg = max_roll_neg
        self.min_roll_pos = min_roll_pos
        self.max_roll_pos = max_roll_pos
        self.min_pitch_neg = min_pitch_neg
        self.max_pitch_neg = max_pitch_neg
        self.min_pitch_pos = min_pitch_pos
        self.max_pitch_pos = max_pitch_pos
        self.min_yaw_neg = min_yaw_neg
        self.max_yaw_neg = max_yaw_neg
        self.min_yaw_pos = min_yaw_pos
        self.max_yaw_pos = max_yaw_pos
        self.min_x_neg = min_x_neg
        self.max_x_neg = max_x_neg
        self.min_x_pos = min_x_pos
        self.max_x_pos = max_x_pos
        self.min_y_neg = min_y_neg
        self.max_y_neg = max_y_neg
        self.min_y_pos = min_y_pos
        self.max_y_pos = max_y_pos
        self.min_z_neg = min_z_neg
        self.max_z_neg = max_z_neg
        self.min_z_pos = min_z_pos
        self.max_z_pos = max_z_pos
        self.p = p

    def _validate_non_negative(self, name: str, value: float) -> None:
        """Validate that a parameter is >= 0 (magnitude)."""
        if value < 0:
            raise ValueError(f"{name} must be >= 0 (specify as magnitude), got {value}")

    def _validate_range(self, name: str, min_val: float, max_val: float) -> None:
        """Validate that min <= max for a range."""
        if min_val > max_val:
            raise ValueError(f"min_{name} ({min_val}) must be <= max_{name} ({max_val})")

    def on_skip(self, input_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Set calibration status to CALIBRATED when transform is skipped."""
        input_dict["gt_calibration_status"] = CalibrationStatus.CALIBRATED.value
        return input_dict

    def transform(self, input_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Apply calibration misalignment augmentation.

        Args:
            input_dict: Dictionary with 'calibration_data'.

        Returns:
            Dictionary with modified calibration data and gt_calibration_status flag.
        """
        calibration_data: CalibrationData = input_dict["calibration_data"]
        original_transform = calibration_data.lidar_to_camera_transformation
        noisy_transform, noise = self.alter_calibration(original_transform)
        calibration_data.lidar_to_camera_transformation = noisy_transform
        calibration_data.noise = noise
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

    def _sample_component(
        self, min_neg: float, max_neg: float, min_pos: float, max_pos: float
    ) -> float:
        """Sample a component value from either negative or positive range.

        Randomly selects between negative and positive range, then samples
        from a truncated gaussian within that range. All input values are
        positive magnitudes; negative range values are negated after sampling.

        Args:
            min_neg: Minimum magnitude for negative range (will be negated).
            max_neg: Maximum magnitude for negative range (will be negated).
            min_pos: Minimum magnitude for positive range.
            max_pos: Maximum magnitude for positive range.

        Returns:
            Sampled value (negative if from neg range, positive if from pos range).
        """
        use_negative = np.random.rand() > 0.5

        if use_negative:
            min_val, max_val = min_neg, max_neg
            if min_val >= max_val:
                return -min_val  # Negate for negative range
            value = self.bounded_gaussian(
                center=min_val,  # center towards least extreme (threshold)
                min_value=min_val,
                max_value=max_val,
                scale=(max_val - min_val) / 1.5,
            )
            return -value  # Negate for negative range
        else:
            min_val, max_val = min_pos, max_pos
            if min_val >= max_val:
                return min_val
            value = self.bounded_gaussian(
                center=min_val,  # center towards least extreme (threshold)
                min_value=min_val,
                max_value=max_val,
                scale=(max_val - min_val) / 1.5,
            )
            return value

    def alter_calibration(self, transform: np.ndarray) -> np.ndarray:
        """Apply random noise to a 4x4 transformation matrix.

        Uses separate RPY angles and xyz translations for more precise control.
        Each component randomly selects between its negative and positive range.
        Only activated components are applied.
        """
        if transform.shape != (4, 4):
            raise ValueError(f"Transform must be 4x4 matrix, got shape {transform.shape}")

        # Sample rotation angles (in degrees, then convert to radians)
        roll = (
            self._sample_component(
                self.min_roll_neg, self.max_roll_neg, self.min_roll_pos, self.max_roll_pos
            )
            if self.activate_roll
            else 0.0
        )
        pitch = (
            self._sample_component(
                self.min_pitch_neg, self.max_pitch_neg, self.min_pitch_pos, self.max_pitch_pos
            )
            if self.activate_pitch
            else 0.0
        )
        yaw = (
            self._sample_component(
                self.min_yaw_neg, self.max_yaw_neg, self.min_yaw_pos, self.max_yaw_pos
            )
            if self.activate_yaw
            else 0.0
        )

        roll_rad = np.deg2rad(roll)
        pitch_rad = np.deg2rad(pitch)
        yaw_rad = np.deg2rad(yaw)

        # Sample translation components (in meters)
        tx = (
            self._sample_component(self.min_x_neg, self.max_x_neg, self.min_x_pos, self.max_x_pos)
            if self.activate_x
            else 0.0
        )
        ty = (
            self._sample_component(self.min_y_neg, self.max_y_neg, self.min_y_pos, self.max_y_pos)
            if self.activate_y
            else 0.0
        )
        tz = (
            self._sample_component(self.min_z_neg, self.max_z_neg, self.min_z_pos, self.max_z_pos)
            if self.activate_z
            else 0.0
        )

        # Build rotation matrix from RPY (ZYX convention: yaw, pitch, roll)
        rotation_matrix = transforms3d.euler.euler2mat(roll_rad, pitch_rad, yaw_rad, axes="sxyz")

        noise_transform = np.eye(4)
        noise_transform[0:3, 0:3] = rotation_matrix
        noise_transform[0:3, 3] = [tx, ty, tz]

        return transform @ noise_transform, noise_transform


class LidarCameraFusion(BaseTransform):
    """Fuse LiDAR points with camera image to create depth and intensity channels.

    Projects LiDAR points onto the camera image plane and creates 5-channel
    fused images in BGRDI format (BGR + depth + intensity). The BGR format
    comes from cv2.imread which loads images in BGR order. Operates on single samples.

    Required keys:
        - img: (H, W, 3) uint8 BGR image.
        - points: (N, 4+) float32 point cloud [x, y, z, intensity, ...].
        - calibration_data: CalibrationData object with camera matrix and transforms.

    Optional keys:
        - affine_transform: (3, 3) float64 affine matrix from Affine transform.
          If present, applies the affine transformation to projected LiDAR points.

    Generated keys:
        - fused_img: (H, W, 5) float32 [0, 1] in BGRDI format (BGR + depth + intensity).

    Args:
        max_depth: Maximum depth for projected LiDAR points in meters.
        dilation_size: Size of dilation kernel for point cloud rendering.
        ego_box: List of 6 floats [x_min, y_min, z_min, x_max, y_max, z_max].
        occlusion_adjust_margin: Distance (meters) to leave between camera and adjusted box wall."""

    _required_keys = ["img", "points", "calibration_data"]
    _optional_keys = ["affine_transform"]

    def __init__(
        self,
        max_depth: float = 128.0,
        dilation_size: int = 1,
        ego_box: List[float] | None = None,
        occlusion_adjust_margin: float = 0.01,
    ):
        super().__init__()
        self.max_depth = max_depth
        self.dilation_size = dilation_size
        self.ego_box = ego_box
        self.occlusion_adjust_margin = occlusion_adjust_margin

    def apply_defaults(self, input_dict: Dict[str, Any]) -> None:
        """Set default affine_transform to None."""
        if "affine_transform" not in input_dict:
            input_dict["affine_transform"] = None

    def transform(self, input_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Create fused image from camera and LiDAR data.

        Args:
            input_dict: Dictionary with:
                - img: Image (H, W, 3) in BGR format (from cv2.imread)
                - points: Point cloud (N, 4+) [x, y, z, intensity, ...]
                - calibration_data: CalibrationData object
                - affine_transform: 3x3 affine transformation matrix (or None)

        Returns:
            Dictionary with added 'fused_img': (H, W, 5) float32 [0, 1] in BGRDI format.
        """
        image = input_dict["img"]
        points = input_dict["points"]
        calibration_data = input_dict["calibration_data"]
        affine_transform = input_dict["affine_transform"]

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
        if self.ego_box is not None:
            points = self._filter_occluded_points(
                points, calibration_data, affine_transform, image.shape[:2]
            )

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

    def _filter_occluded_points(
        self,
        points: npt.NDArray[np.float32],
        calibration_data: CalibrationData,
        affine_transform: npt.NDArray[np.float64] = None,
        image_shape: Tuple[int, int] = None,
    ) -> npt.NDArray[np.float32]:
        """Filter out points occluded by the ego vehicle chassis using ray casting.

        When miscalibration is present (via calibration_data.noise or affine_transform),
        accounts for the miscalibration to properly filter occluded points.

        Automatically shrinks the ego_box if the camera is found inside it.
        Adjusts ONLY the closest X and Y walls to be behind the camera.
        The Z bounds are left untouched.

        Args:
            points: Point cloud (N, 4+) [x, y, z, intensity, ...]
            calibration_data: CalibrationData object (may contain noise for miscalibration)
            affine_transform: Optional 3x3 affine transformation matrix (2D miscalibration)
            image_shape: Optional (height, width) of the image for visibility filtering
        """
        if self.ego_box is None:
            return points

        # 1. Calculate camera center in LiDAR frame
        # If noise exists (miscalibration), we need to use the true (original) transform
        # to get the correct camera position for occlusion filtering
        lidar2cam = calibration_data.lidar_to_camera_transformation
        if calibration_data.noise is not None:
            # Undo the noise to get the true camera position
            # noisy_transform = original_transform @ noise_transform
            # Therefore: original_transform = noisy_transform @ inv(noise_transform)
            noise_inv = np.linalg.inv(calibration_data.noise)
            lidar2cam_true = lidar2cam @ noise_inv
            R = lidar2cam_true[:3, :3]
            t = lidar2cam_true[:3, 3]
        else:
            R = lidar2cam[:3, :3]
            t = lidar2cam[:3, 3]

        # Inverse of T_l2c: [R^T | -R^T t]
        camera_center_lidar = -R.T @ t

        # 2. Prepare Box Bounds
        # Copy to avoid modifying the class attribute persistently
        box_min = np.array(self.ego_box[:3])
        box_max = np.array(self.ego_box[3:])

        # 3. Check if Camera is inside the Ego Box
        if np.all(camera_center_lidar >= box_min) and np.all(camera_center_lidar <= box_max):
            # Calculate distances to walls
            d_min = camera_center_lidar - box_min
            d_max = box_max - camera_center_lidar

            # Iterate over ONLY x (0) and y (1) axes. Ignore z (2).
            for i in range(2):
                if d_min[i] < d_max[i]:
                    # The 'min' wall is closer
                    # Move wall to: camera_position + margin
                    new_val = camera_center_lidar[i] + self.occlusion_adjust_margin
                    box_min[i] = new_val
                else:
                    # The 'max' wall is closer (or equal)
                    # Move wall to: camera_position - margin
                    new_val = camera_center_lidar[i] - self.occlusion_adjust_margin
                    box_max[i] = new_val

        # 4. Perform Ray Casting
        # Slab method for ray intersection
        ray_origins = camera_center_lidar
        ray_directions = points[:, :3] - ray_origins

        with np.errstate(divide="ignore", invalid="ignore"):
            t1 = (box_min - ray_origins) / ray_directions
            t2 = (box_max - ray_origins) / ray_directions

        t_min = np.minimum(t1, t2)
        t_max = np.maximum(t1, t2)

        t_enter = np.max(t_min, axis=1)
        t_exit = np.min(t_max, axis=1)

        hits_box = (t_enter <= t_exit) & (t_exit >= 0)

        # Ray hits box BEFORE reaching the point (t < 0.999)
        occluded = hits_box & (t_enter < 0.999)

        # 5. When affine_transform is present (2D miscalibration), also filter based on
        # whether points would be visible in the transformed image space
        if affine_transform is not None and image_shape is not None:
            # Project points to camera coordinates using the current (possibly noisy) transform
            xyz = points[:, :3]
            pointcloud_ccs = self._transform_points_to_camera(xyz, calibration_data)

            # Filter points behind camera
            valid_mask_3d = pointcloud_ccs[:, 2] > 0.0
            if not np.any(valid_mask_3d):
                return points[~occluded]

            pointcloud_ccs_valid = pointcloud_ccs[valid_mask_3d]
            occluded_valid = occluded[valid_mask_3d]

            # Project to image coordinates
            pointcloud_ics = self._project_points_to_image(pointcloud_ccs_valid, calibration_data)

            # Apply affine transform to see where points would actually appear
            pointcloud_ics_transformed = self._apply_affine_to_points(
                pointcloud_ics, affine_transform
            )

            # Check if transformed points are within image bounds
            h, w = image_shape
            in_bounds = (
                (pointcloud_ics_transformed[:, 0] >= 0)
                & (pointcloud_ics_transformed[:, 0] < w)
                & (pointcloud_ics_transformed[:, 1] >= 0)
                & (pointcloud_ics_transformed[:, 1] < h)
            )

            # Combine occlusion filtering with visibility in transformed image space
            # Points that are occluded OR outside transformed image bounds should be filtered
            occluded_valid = occluded_valid | ~in_bounds

            # Reconstruct full mask
            full_occluded = np.zeros(len(points), dtype=bool)
            full_occluded[valid_mask_3d] = occluded_valid
            full_occluded[~valid_mask_3d] = True  # Points behind camera are considered occluded

            return points[~full_occluded]

        return points[~occluded]

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


class Affine(BaseTransform):
    """Affine transformation augmentation for images.

    Applies controlled affine distortion to the image and stores the affine
    matrix in input_dict for later application to projected LiDAR points.
    Automatically applies zoom to ensure the transformed image covers the
    entire viewport without black borders.

    Required keys:
        - img: (H, W, 3) uint8 image.

    Optional keys:
        - None

    Generated keys:
        - affine_transform: (3, 3) float64 affine matrix. Always set - identity matrix
          if augmentation not applied, actual transform matrix if applied.
        - img: Modified in-place with affine transformation (when applied).

    Args:
        p: Probability of applying augmentation.
        max_distortion: Maximum corner displacement as fraction of image size.
    """

    _required_keys = ["img"]

    def __init__(self, p: float = 0.5, max_distortion: float = 0.1):
        super().__init__()
        self.p = p
        self.max_distortion = max_distortion

    def on_skip(self, input_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Set identity matrix when transform is skipped."""
        input_dict["affine_transform"] = np.eye(3, dtype=np.float64)
        return input_dict

    def transform(self, input_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Apply random affine transformation to the image.

        Args:
            input_dict: Dictionary with 'img'.

        Returns:
            Dictionary with transformed image and 'affine_transform' matrix.
        """
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

        # Calculate the inverse transform to map destination corners to source space
        inv_affine = cv2.invertAffineTransform(affine_matrix_2x3)

        # Destination corners (the full image view)
        corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
        corners_hom = np.hstack([corners, np.ones((4, 1), dtype=np.float32)])
        src_corners = (inv_affine @ corners_hom.T).T

        # Calculate required zoom to ensure source corners are within image bounds
        cx, cy = w / 2.0, h / 2.0
        max_x = np.max(np.abs(src_corners[:, 0] - cx))
        max_y = np.max(np.abs(src_corners[:, 1] - cy))
        scale = max(1.0, max_x / cx, max_y / cy)

        # Apply zoom to the affine transform
        zoom_mat = np.array(
            [[scale, 0, cx * (1 - scale)], [0, scale, cy * (1 - scale)], [0, 0, 1]],
            dtype=np.float64,
        )

        affine_matrix_3x3 = np.eye(3, dtype=np.float64)
        affine_matrix_3x3[:2, :3] = affine_matrix_2x3
        affine_matrix_3x3 = affine_matrix_3x3 @ zoom_mat
        affine_matrix_2x3 = affine_matrix_3x3[:2]

        image = cv2.warpAffine(image, affine_matrix_2x3, (w, h), borderMode=cv2.BORDER_CONSTANT)

        input_dict["img"] = image
        input_dict["affine_transform"] = affine_matrix_3x3
        return input_dict


class SaveFusionPreview(BaseTransform):
    """Save preview images of fused RGB-LiDAR data for visualization.

    Creates two overlay images per sample:
    - RGB with depth points overlay using colormap
    - RGB with intensity points overlay using colormap

    Required keys:
        - fused_img: (H, W, 5) float32 [0, 1] in BGRDI format (from LidarCameraFusion).
        - metadata: Dict with image path info (must contain metadata["image"]["img_path"]).

    Optional keys:
        - gt_calibration_status: int (0=calibrated, 1=miscalibrated). If not present,
          uses "unknown" as the status suffix in output filenames.

    Generated keys:
        - None (pass-through transform, only saves files to disk).

    Args:
        p: Probability of saving preview images (default: 1.0, always save).
        out_dir: Output directory for saving preview images.
        max_depth: Maximum depth value used during fusion (for recovery).
        alpha: Blending factor for overlay (0.0 = RGB only, 1.0 = overlay only).
        depth_colormap: Matplotlib colormap name for depth visualization.
        intensity_colormap: Matplotlib colormap name for intensity visualization.
    """

    _required_keys = ["fused_img", "metadata"]
    _optional_keys = ["gt_calibration_status"]

    def __init__(
        self,
        p: float = 1.0,
        out_dir: str = "",
        max_depth: float = 128.0,
        alpha: float = 0.5,
        depth_colormap: str = "turbo",
        intensity_colormap: str = "jet",
    ):
        super().__init__()
        self.p = p
        self.out_dir = Path(out_dir)
        self.max_depth = max_depth
        self.alpha = alpha
        self.depth_cmap = plt.get_cmap(depth_colormap)
        self.intensity_cmap = plt.get_cmap(intensity_colormap)

        self.out_dir.mkdir(parents=True, exist_ok=True)

    def apply_defaults(self, input_dict: Dict[str, Any]) -> None:
        """Set default calibration status to None (unknown)."""
        if "gt_calibration_status" not in input_dict:
            input_dict["gt_calibration_status"] = None

    def transform(self, input_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Save preview images for each sample in the batch.

        Args:
            input_dict: Dictionary containing:
                - fused_img: Fused image (H, W, 5) float32 [0, 1] in BGRDI format
                - metadata: Metadata dict with image path info
                - gt_calibration_status: Calibration status label
                  (0=calibrated, 1=miscalibrated, or None).

        Returns:
            Unmodified input_dict (pass-through).
        """
        fused_img = input_dict["fused_img"]
        metadata_dict = input_dict["metadata"]
        calibration_status = input_dict["gt_calibration_status"]
        self._save_preview(fused_img, metadata_dict, calibration_status)

        return input_dict

    def _save_preview(
        self,
        fused_img: npt.NDArray[np.float32],
        metadata: Dict[str, Any],
        calibration_status: int | None,
    ) -> None:
        """Save depth and intensity preview images for a single sample."""
        bgr, depth, intensity = self._recover_channels(fused_img)

        base_name = self._get_base_filename(metadata)
        if calibration_status is None:
            status_suffix = ""
        elif calibration_status == CalibrationStatus.CALIBRATED.value:
            status_suffix = "_calibrated"
        else:
            status_suffix = "_miscalibrated"

        # Convert BGR to RGB for matplotlib colormap processing
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        depth_overlay = self._create_overlay(rgb, depth, self.depth_cmap, self.alpha)
        intensity_overlay = self._create_overlay(rgb, intensity, self.intensity_cmap, self.alpha)

        depth_path = self.out_dir / f"{base_name}{status_suffix}_depth.png"
        intensity_path = self.out_dir / f"{base_name}{status_suffix}_intensity.png"

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
