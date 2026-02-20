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

from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import numpy.typing as npt
import transforms3d
from scipy.stats import truncnorm

from autoware_ml.datamodule.t4dataset.lidar_calibration_status import (
    CalibrationData,
    CalibrationStatus,
)
from autoware_ml.transforms.base import BaseTransform


class CropBoxOuter(BaseTransform):
    """Remove points that are OUTSIDE a 3D bounding box (Keep Inside).

    Required keys:
        - points: (N, C) float32 point cloud array where first 3 columns are [x, y, z].

    Optional keys:
        - None

    Generated keys:
        - points: Modified in-place with filtered points (only inside the box).

    Args:
        crop_box: List of 6 floats defining the box [x_min, y_min, z_min, x_max, y_max, z_max].
    """

    _required_keys = ["points"]

    def __init__(self, crop_box: List[float]):
        super().__init__()
        if len(crop_box) != 6:
            raise ValueError(f"crop_box must have 6 elements, got {len(crop_box)}")
        self.crop_box = np.array(crop_box, dtype=np.float32)

    def transform(self, input_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Remove points outside the box.

        Args:
            input_dict: Dictionary with 'points' key containing (N, C) float32 array.

        Returns:
            Dictionary with points inside the box.
        """
        points: npt.NDArray[np.float32] = input_dict["points"]

        x_min, y_min, z_min, x_max, y_max, z_max = self.crop_box

        # Keep points INSIDE the box
        mask = (
            (points[:, 0] >= x_min)
            & (points[:, 0] <= x_max)
            & (points[:, 1] >= y_min)
            & (points[:, 1] <= y_max)
            & (points[:, 2] >= z_min)
            & (points[:, 2] <= z_max)
        )

        input_dict["points"] = points[mask]
        return input_dict


class CropBoxInner(BaseTransform):
    """Remove points that are INSIDE a 3D bounding box (Keep Outside).

    Typically used to remove ego vehicle points from the point cloud.

    Required keys:
        - points: (N, C) float32 point cloud array where first 3 columns are [x, y, z].

    Optional keys:
        - None

    Generated keys:
        - points: Modified in-place with filtered points (only outside the box).

    Args:
        crop_box: List of 6 floats defining the box [x_min, y_min, z_min, x_max, y_max, z_max].
    """

    _required_keys = ["points"]

    def __init__(self, crop_box: List[float]):
        super().__init__()
        if len(crop_box) != 6:
            raise ValueError(f"crop_box must have 6 elements, got {len(crop_box)}")
        self.crop_box = np.array(crop_box, dtype=np.float32)

    def transform(self, input_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Remove points inside the box.

        Args:
            input_dict: Dictionary with 'points' key containing (N, C) float32 array.

        Returns:
            Dictionary with points outside the box.
        """
        points: npt.NDArray[np.float32] = input_dict["points"]

        x_min, y_min, z_min, x_max, y_max, z_max = self.crop_box

        # Keep points OUTSIDE the box
        mask = (
            (points[:, 0] < x_min)
            | (points[:, 0] > x_max)
            | (points[:, 1] < y_min)
            | (points[:, 1] > y_max)
            | (points[:, 2] < z_min)
            | (points[:, 2] > z_max)
        )

        input_dict["points"] = points[mask]
        return input_dict


class LidarLidarCalibrationMisalignment(BaseTransform):
    """Calibration misalignment augmentation for LiDAR-LiDAR calibration.

    Each rotation (roll, pitch, yaw) and translation (x, y, z) component has
    separate negative and positive ranges. During augmentation, one of the two
    ranges is randomly selected for each component. Each component can be
    individually activated or deactivated.

    Required keys:
        - calibration_data: CalibrationData object with lidar1_to_lidar2_transform.

    Generated keys:
        - gt_calibration_status: int (CalibrationStatus.CALIBRATED or MISCALIBRATED).
        - calibration_data.noise: 4x4 noise transform matrix (when augmentation applied).
        - calibration_data.lidar1_to_lidar2_transform: Modified with noise (when applied).

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
        """Apply LiDAR-LiDAR calibration misalignment augmentation.

        Args:
            input_dict: Dictionary with 'calibration_data'.

        Returns:
            Dictionary with modified calibration data and gt_calibration_status flag.
        """
        calibration_data: CalibrationData = input_dict["calibration_data"]
        original_transform = calibration_data.lidar1_to_lidar2_transform
        noisy_transform, noise = self.alter_calibration(original_transform)
        calibration_data.lidar1_to_lidar2_transform = noisy_transform
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
        """Sample a component value from either negative or positive range."""
        use_negative = np.random.rand() > 0.5

        if use_negative:
            min_val, max_val = min_neg, max_neg
            if min_val >= max_val:
                return -min_val
            value = self.bounded_gaussian(
                center=min_val,
                min_value=min_val,
                max_value=max_val,
                scale=(max_val - min_val) / 1.5,
            )
            return -value
        else:
            min_val, max_val = min_pos, max_pos
            if min_val >= max_val:
                return min_val
            value = self.bounded_gaussian(
                center=min_val,
                min_value=min_val,
                max_value=max_val,
                scale=(max_val - min_val) / 1.5,
            )
            return value

    def alter_calibration(self, transform: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply random noise to a 4x4 transformation matrix."""
        if transform.shape != (4, 4):
            raise ValueError(f"Transform must be 4x4 matrix, got shape {transform.shape}")

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

        rotation_matrix = transforms3d.euler.euler2mat(roll_rad, pitch_rad, yaw_rad, axes="sxyz")

        noise_transform = np.eye(4)
        noise_transform[0:3, 0:3] = rotation_matrix
        noise_transform[0:3, 3] = [tx, ty, tz]

        return transform @ noise_transform, noise_transform


class LidarLidarFusion(BaseTransform):
    """Fuse two LiDAR point clouds into a 2D multi-channel representation.

    Projects both LiDAR point clouds onto a common virtual perspective camera
    view and creates 5-channel fused images.

    Required keys:
        - lidar1_points: (N, 4+) float32 point cloud [x, y, z, intensity, ...].
        - lidar2_points: (M, 4+) float32 point cloud [x, y, z, intensity, ...].
        - calibration_data: CalibrationData object with lidar1_to_lidar2_transform.

    Generated keys:
        - fused_img: (H, W, 5) float32 [0, 1] in format:
          [L1_depth, L1_intensity, L2_depth, L2_intensity, Depth_diff].

    Args:
        width: Width of the 2D representation.
        height: Height of the 2D representation.
        fx: Focal length in x.
        fy: Focal length in y.
        cx: Principal point x.
        cy: Principal point y.
        max_depth: Maximum depth for projected LiDAR points in meters.
        dilation_size: Size of dilation kernel for point cloud rendering.
    """

    _required_keys = ["lidar1_points", "lidar2_points", "calibration_data"]

    def __init__(
        self,
        width: int = 1024,
        height: int = 512,
        fx: float = 500.0,
        fy: float = 500.0,
        cx: float = 512.0,
        cy: float = 256.0,
        max_depth: float = 80.0,
        dilation_size: int = 1,
    ):
        super().__init__()
        self.width = width
        self.height = height
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.max_depth = max_depth
        self.dilation_size = dilation_size
        self.camera_matrix = np.array(
            [[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32
        )

    def transform(self, input_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Create fused image from two LiDAR point clouds.

        Args:
            input_dict: Dictionary with:
                - lidar1_points: Point cloud 1
                - lidar2_points: Point cloud 2
                - calibration_data: CalibrationData object

        Returns:
            Dictionary with added 'fused_img': (H, W, 5) float32 [0, 1].
        """
        l1_points = input_dict["lidar1_points"]
        l2_points = input_dict["lidar2_points"]
        calib = input_dict["calibration_data"]

        # L1 points are assumed to be in L1 coordinate system
        l1_xyz = l1_points[:, :3]
        l1_int = l1_points[:, 3]

        # Transform L2 points to L1 frame (virtual camera frame)
        # T_L1_to_L2 = calib.lidar1_to_lidar2_transform
        # P_L1 = T_L1_to_L2^-1 * P_L2
        l2_to_l1 = np.linalg.inv(calib.lidar1_to_lidar2_transform)
        l2_xyz = l2_points[:, :3]
        l2_hom = np.concatenate([l2_xyz, np.ones((l2_xyz.shape[0], 1), dtype=np.float32)], axis=1)
        l2_xyz_in_l1 = (l2_to_l1 @ l2_hom.T).T[:, :3]
        l2_int = l2_points[:, 3]

        # Project both to 2D
        l1_img = self._create_lidar_image(l1_xyz, l1_int)
        l2_img = self._create_lidar_image(l2_xyz_in_l1, l2_int)

        # Combine into 5 channels
        # [L1_depth, L1_intensity, L2_depth, L2_intensity, Depth_diff]
        fused = np.zeros((self.height, self.width, 5), dtype=np.float32)
        fused[..., 0] = l1_img[..., 0]
        fused[..., 1] = l1_img[..., 1]
        fused[..., 2] = l2_img[..., 0]
        fused[..., 3] = l2_img[..., 1]

        # Channel 4: Depth difference
        mask = (l1_img[..., 0] > 0) & (l2_img[..., 0] > 0)
        fused[mask, 4] = np.abs(l1_img[mask, 0] - l2_img[mask, 0])

        input_dict["fused_img"] = fused
        return input_dict

    def _create_lidar_image(
        self, points_xyz: npt.NDArray[np.float32], intensities: npt.NDArray[np.float32]
    ) -> npt.NDArray[np.float32]:
        """Project 3D points to 2D image with depth and intensity channels."""
        h, w = self.height, self.width
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
            self.camera_matrix,
            np.zeros(5, dtype=np.float32),
        )
        pointcloud_ics = pointcloud_ics.reshape(-1, 2)

        valid_mask = (
            (pointcloud_ics[:, 0] >= 0)
            & (pointcloud_ics[:, 0] <= w - 1)
            & (pointcloud_ics[:, 1] >= 0)
            & (pointcloud_ics[:, 1] <= h - 1)
            & (points_xyz[:, 2] > 0.0)  # Z > 0
            & (points_xyz[:, 2] < self.max_depth)
        )

        valid_ics = pointcloud_ics[valid_mask]
        valid_xyz = points_xyz[valid_mask]
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

            center_depths = valid_xyz[:, 2] / self.max_depth

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
