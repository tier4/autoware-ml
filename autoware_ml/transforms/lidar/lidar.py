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

from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import numpy.typing as npt
import transforms3d
from scipy.stats import truncnorm
import matplotlib.pyplot as plt

from autoware_ml.datamodule.t4dataset.lidar_calibration_status import (
    CalibrationData,
    CalibrationStatus,
)
from autoware_ml.transforms.base import BaseTransform
from autoware_ml.transforms.lidar.utils.projections import project_spherical, create_lidar_image


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
        noisy_transform, noise, translation_noise, rotation_noise = self.alter_calibration(original_transform)
        calibration_data.lidar1_to_lidar2_transform = noisy_transform
        calibration_data.noise = noise
        input_dict["calibration_data"] = calibration_data
        # ROTATION_NOISE_THRESHOLD = 0.1
        # TRANSLATION_NOISE_THRESHOLD = 0.3
        # if(abs(translation_noise) >= TRANSLATION_NOISE_THRESHOLD or abs(rotation_noise) >= ROTATION_NOISE_THRESHOLD) :
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

    def alter_calibration(self, transform: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float, float]:
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

        # Calculate translation/rotation noise magnitude
        translation_magnitude = float(np.linalg.norm([tx, ty, tz]))
        # trace of a 3x3 rotation matrix is R[0,0] + R[1,1] + R[2,2]
        trace = np.trace(rotation_matrix)
        # Clip to [-1, 1] to avoid floating point errors with arccos
        angular_deviation_rad = np.arccos(np.clip((trace - 1.0) / 2.0, -1.0, 1.0))
        rotation_magnitude_deg = np.rad2deg(angular_deviation_rad)

        return transform @ noise_transform, noise_transform, translation_magnitude, rotation_magnitude_deg


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
        max_depth: Maximum depth for projected LiDAR points in meters.
    """

    _required_keys = ["lidar1_points", "lidar2_points", "calibration_data"]

    def __init__(
        self,
        width: int = 1024,
        height: int = 512,
        fx: float = 600.0,
        fy: float = 600.0,
        cx: float = 640.0,
        cy: float = 360.0,
        max_depth: float = 80.0,
        dilation_size: int = 1,
        projection: str = "spherical",

    ):
        super().__init__()
        self.width = width
        self.height = height
        self.max_depth = max_depth
        self.dilation_size = dilation_size
        self.camera_matrix = np.array(
            [[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32
        )
        self.projection = projection


    def transform(self, input_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Create fused image from two LiDAR point clouds.

        Args:
            input_dict: Dictionary with:
                - lidar1_points: Point cloud 1
                - lidar2_points: Point cloud 2
                - calibration_data: CalibrationData object

        Returns:
            Dictionary with added 'fused_img': (H, W, 3) float32 [0, 1].
        """
        l1_points = input_dict["lidar1_points"]
        l2_points = input_dict["lidar2_points"]
        calib = input_dict["calibration_data"]
        if calib.noise is not None:
            # Misalign l1 points using the provided calibration transform
            # l1_points and l2_points are in ground-truth baselink frame.
            T_base_l1 = calib.baselink_to_lidar1_transform
            T_base_l2 = calib.baselink_to_lidar2_transform
            T_l2_l1 = calib.lidar1_to_lidar2_transform
            # This represents l1 points in lidar2 frame using current (noisy) calibration,
            # then back to baselink using ground-truth T_base_l2
            T_misalign = T_base_l2 @ T_l2_l1 @ np.linalg.inv(T_base_l1)
            l1_xyz = l1_points[:, :3]
            l1_xyz_h = np.column_stack([l1_xyz, np.ones(l1_xyz.shape[0], dtype=np.float32)])
            l1_points[:, :3] = (T_misalign @ l1_xyz_h.T).T[:, :3]


        # All points are assumed to be in "base_link" frame
        if self.projection == "spherical":
          avg_lidar_height = np.mean([calib.baselink_to_lidar1_transform[:, 3][2], calib.baselink_to_lidar2_transform[:, 3][2]])
          l1_points[:, 1] += avg_lidar_height
          l2_points[:, 1] += avg_lidar_height
          l1_img = project_spherical(l1_points, self.width, self.height, self.dilation_size)
          l2_img = project_spherical(l2_points, self.width, self.height, self.dilation_size)

          # Combine into multiple channels
          fused = np.zeros((self.height, self.width, 4), dtype=np.float32)
          fused[:, :, 0] = l1_img[:, :, 0]
          fused[:, :, 1] = l1_img[:, :, 1]
          fused[:, :, 2] = l2_img[:, :, 0]
          fused[:, :, 3] = l2_img[:, :, 1]
          # fused[:, :, 2] = np.abs(l1_img - l2_img)

          # Normalize between 0 and 1
          fused = fused.astype(np.float32) / 255.0
          input_dict["fused_img"] = fused
        elif self.projection == "pinhole":
          # virtual camera is defined in baselink frame
          virtual_camera_xyz = [5.0, 0.0, 0.75]
          virtual_camera_roll_pitch_yaw_deg = [-90.0, 0.0, -107]

          # Transformation from base_link to virtual camera
          roll, pitch, yaw = np.radians(virtual_camera_roll_pitch_yaw_deg)
          cam_rot = transforms3d.euler.euler2mat(roll, pitch, yaw, axes="sxyz")
          T_base_cam = np.eye(4, dtype=np.float32)
          T_base_cam[:3, :3] = cam_rot
          T_base_cam[:3, 3] = virtual_camera_xyz
          T_cam_base = np.linalg.inv(T_base_cam)

          l1_xyz = l1_points[:, :3]
          l1_int = l1_points[:, 3]
          l1_xyz_h = np.column_stack([l1_xyz, np.ones(l1_xyz.shape[0], dtype=np.float32)])

          l2_xyz = l2_points[:, :3]
          l2_int = l2_points[:, 3]
          l2_xyz_h = np.column_stack([l2_xyz, np.ones(l2_xyz.shape[0], dtype=np.float32)])

          # Transform points to virtual camera frame
          # l1_xyz_cam = (T_cam_base @ T_misalign @ l1_xyz_h.T).T[:, :3]
          l1_xyz_cam = (T_cam_base @ l1_xyz_h.T).T[:, :3]
          l2_xyz_cam = (T_cam_base @ l2_xyz_h.T).T[:, :3]

          # Project both to 2D
          l1_img = create_lidar_image(self.height, self.width, self.camera_matrix, self.max_depth, self.dilation_size, l1_xyz_cam, l1_int)
          l2_img = create_lidar_image(self.height, self.width, self.camera_matrix, self.max_depth, self.dilation_size, l2_xyz_cam, l2_int)

          # Combine into 5 channels
          # [L1_depth, L1_intensity, L2_depth, L2_intensity, Depth_diff]
          channels = 4
          fused = np.zeros((self.height, self.width, channels), dtype=np.float32)
          fused[..., 0] = l1_img[..., 0]
          fused[..., 1] = l1_img[..., 1]
          fused[..., 2] = l2_img[..., 0]
          fused[..., 3] = l2_img[..., 1]

          # Channel 4: Depth difference
          # mask = (l1_img[..., 0] > 0) & (l2_img[..., 0] > 0)
          # fused[mask, 4] = np.abs(l1_img[mask, 0] - l2_img[mask, 0])
          input_dict["fused_img"] = fused
        return input_dict

class SaveFusionPreview(BaseTransform):
    """Save preview images of fused LiDAR-LiDAR data for visualization.

    For each sample, generate an image for each channel

    Required keys:
        - fused_img: (H, W, N) float32 [0, 1] (from LidarLidarFusion).

    Optional keys:
        - gt_calibration_status: int (0=calibrated, 1=miscalibrated). If not present,
          uses "unknown" as the status suffix in output filenames.

    Generated keys:
        - None (pass-through transform, only saves files to disk).

    Args:
        p: Probability of saving preview images (default: 1.0, always save).
        out_dir: Output directory for saving preview images.
    """

    _required_keys = ["fused_img"]
    _optional_keys = ["gt_calibration_status"]

    def __init__(
        self,
        p: float = 1.0,
        out_dir: str = "",
        colormap: str = "turbo",
    ):
        super().__init__()
        self.p = p
        self.out_dir = Path(out_dir)
        self.cmap = plt.get_cmap(colormap) # pyright: ignore[reportAttributeAccessIssue]
        self._counter = 0

        self.out_dir.mkdir(parents=True, exist_ok=True)

    def transform(self, input_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Save preview images for each sample in the batch.

        Args:
            input_dict: Dictionary containing:
                - fused_img: Fused image (H, W, N) float32 [0, 1]
                - gt_calibration_status: Calibration status label
                  (0=calibrated, 1=miscalibrated, or None).

        Returns:
            Unmodified input_dict (pass-through).
        """
        fused_img = input_dict["fused_img"]
        calibration_status = input_dict.get("gt_calibration_status")
        self._save_preview(fused_img, calibration_status)

        return input_dict

    def _save_preview(
        self,
        fused_img: npt.NDArray[np.float32],
        calibration_status: int | None,
    ) -> None:
        """Save preview images for a single sample."""
        if self.p is not None and np.random.rand() > self.p:
            return

        if calibration_status is None:
            status_suffix = ""
        elif calibration_status == CalibrationStatus.CALIBRATED.value:
            status_suffix = "_calibrated"
        else:
            status_suffix = "_miscalibrated"

        idx = self._counter
        self._counter += 1

        for channel in range(fused_img.shape[2]):
          # Normalize for visualization
          img = np.clip(fused_img[:, :, channel], 0, 1)
          # Apply colormaps (matplotlib returns RGBA float [0, 1])
          img_colored = (self.cmap(img)[:, :, :3] * 255).astype(np.uint8)

          # Save images (OpenCV expects BGR)
          cv2.imwrite(
              str(self.out_dir / f"{idx:06d}_channel{channel:06d}_{status_suffix}.png"),
              cv2.cvtColor(img_colored, cv2.COLOR_RGB2BGR),
          )