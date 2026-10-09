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

"""Camera-LiDAR fusion transforms to support ModelGTSample.

This module contains calibration-status augmentations and the projection of the points onto
the images the calibration classifier reads.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np
import numpy.typing as npt
import torch
import transforms3d
from scipy.stats import truncnorm

from autoware_ml.dataclasses.batch.sample_batch import ModelGTSample
from autoware_ml.geometry.cameras.base_images import BaseImages, compose_lidar2images
from autoware_ml.geometry.points.base_points import BasePoints
from autoware_ml.transforms.base import BaseTransform
from autoware_ml.utils.calibration import CalibrationData, CalibrationStatus


def camera_calibration_data(camera_image_data: BaseImages, index: int) -> CalibrationData:
    """Bundle the calibration of one camera the way the projection math reads it.

    Args:
        camera_image_data: Images of the sample.
        index: Index of the camera.

    Returns:
        CalibrationData: Calibration of that camera.
    """
    return CalibrationData(
        camera_matrix=camera_image_data.camera_intrinsics[index].numpy(),
        distortion_coefficients=camera_image_data.distortion_coefficients[index].numpy(),
        lidar_to_camera_transformation=camera_image_data.lidar2cams[index].numpy(),
        distortion_model=camera_image_data.distortion_models[index],
        noise=(
            None if camera_image_data.noises is None else camera_image_data.noises[index].numpy()
        ),
        new_camera_matrix=camera_image_data.augmented_camera_intrinsics[index].numpy(),
    )


class CalibrationMisalignment(BaseTransform):
    """Calibration misalignment augmentation for camera-LiDAR calibration.

    Each rotation (roll, pitch, yaw) and translation (x, y, z) component has
    separate negative and positive ranges. During augmentation, one of the two
    ranges is randomly selected for each component. Each component can be
    individually activated or deactivated.

    All parameters are specified as positive magnitudes. The `_neg` suffix
    indicates the value will be negated when applied. This keeps min < max
    intuitive in the config.

    The perturbation is written onto the lidar to camera matrices of the images, together
    with the noise it was built from and the calibration status the classifier is trained on.
    """

    _required_keys = ["camera_image_data"]

    def __init__(
        self,
        *,
        probability: float,
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
        """Initialize the CalibrationMisalignment transform.

        Args:
            probability: Probability of applying augmentation.
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
        super().__init__(probability=probability)

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

    def _validate_non_negative(self, name: str, value: float) -> None:
        """Validate that a parameter is non-negative.

        Args:
            name: Parameter name.
            value: Parameter value.

        Raises:
            ValueError: If the value is negative.
        """
        if value < 0:
            raise ValueError(f"{name} must be >= 0 (specify as magnitude), got {value}")

    def _validate_range(self, name: str, min_val: float, max_val: float) -> None:
        """Validate that a configured range is well ordered.

        Args:
            name: Range name.
            min_val: Lower bound.
            max_val: Upper bound.

        Raises:
            ValueError: If the lower bound exceeds the upper bound.
        """
        if min_val > max_val:
            raise ValueError(f"min_{name} ({min_val}) must be <= max_{name} ({max_val})")

    def on_skip(self, model_gt_sample: ModelGTSample) -> ModelGTSample:
        """Mark every camera calibrated when the transform is skipped.

        Args:
            model_gt_sample: ModelGTSample instance holding loaded images.

        Returns:
            Updated ModelGTSample instance with the calibration statuses set.
        """
        # This is checked in the _validate_required_keys()
        camera_image_data: BaseImages = (
            model_gt_sample.camera_image_data  # type: ignore[reportOptionalMemberAccess]
        )

        num_cameras = len(camera_image_data.camera_names)
        return model_gt_sample._replace(
            camera_image_data=BaseImages.model_validate(
                camera_image_data.model_copy(
                    update={
                        "noises": torch.eye(4, dtype=torch.float32).repeat(num_cameras, 1, 1),
                        "calibration_statuses": torch.full(
                            (num_cameras,), CalibrationStatus.CALIBRATED.value, dtype=torch.int64
                        ),
                    }
                )
            )
        )

    def transform(self, model_gt_sample: ModelGTSample) -> ModelGTSample:
        """Perturb the lidar to camera calibration of every camera.

        Args:
            model_gt_sample: ModelGTSample instance holding loaded images.

        Returns:
            Updated ModelGTSample instance with the perturbed calibration.
        """
        # This is checked in the _validate_required_keys()
        camera_image_data: BaseImages = (
            model_gt_sample.camera_image_data  # type: ignore[reportOptionalMemberAccess]
        )

        lidar2cams = []
        noises = []
        for lidar2cam in camera_image_data.lidar2cams:
            noisy_transform, noise = self.alter_calibration(lidar2cam.numpy())
            lidar2cams.append(torch.from_numpy(noisy_transform.astype(np.float32)))
            noises.append(torch.from_numpy(noise.astype(np.float32)))

        num_cameras = len(camera_image_data.camera_names)
        noisy_lidar2cams = torch.stack(lidar2cams)
        return model_gt_sample._replace(
            camera_image_data=BaseImages.model_validate(
                camera_image_data.model_copy(
                    update={
                        "lidar2cams": noisy_lidar2cams,
                        "lidar2images": compose_lidar2images(
                            camera_image_data.augmented_camera_intrinsics, noisy_lidar2cams
                        ),
                        "noises": torch.stack(noises),
                        "calibration_statuses": torch.full(
                            (num_cameras,),
                            CalibrationStatus.MISCALIBRATED.value,
                            dtype=torch.int64,
                        ),
                    }
                )
            )
        )

    def bounded_gaussian(
        self, center: float, min_value: float, max_value: float, scale: float
    ) -> float:
        """Generate a value from a truncated normal distribution.

        Args:
            center: Distribution center before truncation.
            min_value: Lower truncation bound.
            max_value: Upper truncation bound.
            scale: Distribution scale parameter.

        Returns:
            Sampled scalar value.

        Raises:
            ValueError: If the bounds are invalid or the scale is non-positive.
        """
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

    def alter_calibration(self, transform: npt.NDArray) -> npt.NDArray:
        """Apply random noise to a 4x4 transformation matrix.

        The noise is applied in the camera frame so that, e.g., an x-translation
        shifts projected points along the camera's x-axis (horizontal in image).
        Mathematically: T_noisy = T_noise @ T_l2c, producing the pipeline
        lidar -> camera -> miscalibration.

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

        return noise_transform @ transform, noise_transform


class LidarCameraFusion(BaseTransform):
    """Project the points onto every image as a depth and an intensity channel.

    The classifier reads the images together with the projected channels, so a perturbed
    calibration shows up as points landing next to the objects they belong to.
    """

    _required_keys = ["camera_image_data", "point_cloud_data"]

    def __init__(
        self,
        *,
        max_depth: float = 128.0,
        dilation_size: int = 1,
        ego_box: Sequence[float] | None = None,
        occlusion_adjust_margin: float = 0.01,
    ):
        """Initialize the LidarCameraFusion transform.

        Args:
            max_depth: Maximum depth for projected LiDAR points in meters.
            dilation_size: Size of dilation kernel for point cloud rendering.
            ego_box: List of 6 floats [x_min, y_min, z_min, x_max, y_max, z_max].
            occlusion_adjust_margin: Distance (meters) to leave between camera and adjusted box
                wall.
        """
        super().__init__()
        self.max_depth = max_depth
        self.dilation_size = dilation_size
        self.ego_box = ego_box
        self.occlusion_adjust_margin = occlusion_adjust_margin

    def transform(self, model_gt_sample: ModelGTSample) -> ModelGTSample:
        """Project the points onto every image of the sample.

        Args:
            model_gt_sample: ModelGTSample instance holding loaded images and points.

        Returns:
            Updated ModelGTSample instance whose images carry the projected channels.
        """
        # This is checked in the _validate_required_keys()
        camera_image_data: BaseImages = (
            model_gt_sample.camera_image_data  # type: ignore[reportOptionalMemberAccess]
        )
        point_cloud_data: BasePoints = (
            model_gt_sample.point_cloud_data  # type: ignore[reportOptionalMemberAccess]
        )

        points = point_cloud_data.points.numpy()
        pixel_affines = camera_image_data.image_augmentation_pixel_affines()
        depth_maps = []
        for index, image in enumerate(camera_image_data.images):
            depth_maps.append(
                torch.from_numpy(
                    self._create_fused_image(
                        np.transpose(image.numpy(), (1, 2, 0)),
                        points,
                        camera_calibration_data(camera_image_data, index),
                        pixel_affines[index].numpy().astype(np.float64),
                    )
                )
            )

        return model_gt_sample._replace(
            camera_image_data=BaseImages.model_validate(
                camera_image_data.model_copy(update={"depth_maps": torch.stack(depth_maps)})
            )
        )

    def _create_fused_image(
        self,
        image: npt.NDArray[np.float32],
        points: npt.NDArray[np.float32],
        calibration_data: CalibrationData,
        affine_transform: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float32]:
        """Create the projected depth and intensity channels of one camera.

        Args:
            image: Image in (height, width, num_channels) layout.
            points: Point cloud with intensity values.
            calibration_data: Camera-LiDAR calibration data.
            affine_transform: Image affine the image augmentations applied.

        Returns:
            The depth and intensity channels in (2, height, width) layout.
        """
        if self.ego_box is not None:
            points = self._filter_occluded_points(
                points, calibration_data, affine_transform, image.shape[:2]
            )

        xyz = points[:, :3]
        intensities = points[:, 3]

        point_cloud_ccs = self._transform_points_to_camera(xyz, calibration_data)

        valid_mask = point_cloud_ccs[:, 2] > 0.0
        point_cloud_ccs = point_cloud_ccs[valid_mask]
        intensities = intensities[valid_mask]

        point_cloud_ics = self._project_points_to_image(point_cloud_ccs, calibration_data)
        point_cloud_ics = self._apply_affine_to_points(point_cloud_ics, affine_transform)

        return self._create_lidar_images(
            image.shape[:2], point_cloud_ics, point_cloud_ccs, intensities
        )

    def _filter_occluded_points(
        self,
        points: npt.NDArray[np.float32],
        calibration_data: CalibrationData,
        affine_transform: npt.NDArray[np.float64],
        image_shape: tuple[int, int],
    ) -> npt.NDArray[np.float32]:
        """Filter out points occluded by the ego vehicle chassis using ray casting.

        The rays are cast from the true camera position, so a miscalibration in
        calibration_data.noise does not move them. Points that leave the image after the
        affine_transform are dropped as well.

        When the camera lies inside the ego box, the closest x and y walls of the box move
        behind the camera. The z bounds stay the same.

        Args:
            points: Point cloud (N, 4+) [x, y, z, intensity, ...]
            calibration_data: CalibrationData object, its noise holds the miscalibration.
            affine_transform: 3x3 affine of the image augmentations.
            image_shape: (height, width) of the image.

        Returns:
            The points that are neither occluded nor outside the image.
        """
        # 1. Calculate camera center in LiDAR frame
        # If noise exists (miscalibration), we need to use the true (original) transform
        # to get the correct camera position for occlusion filtering
        lidar2cam = calibration_data.lidar_to_camera_transformation
        if calibration_data.noise is not None:
            # Undo the noise to get the true camera position
            # noisy_transform = noise_transform @ original_transform
            # Therefore: original_transform = inv(noise_transform) @ noisy_transform
            noise_inv = np.linalg.inv(calibration_data.noise)
            lidar2cam_true = noise_inv @ lidar2cam
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

        # 5. Also drop the points that leave the image after the affine transform
        # Project points to camera coordinates using the current (possibly noisy) transform
        xyz = points[:, :3]
        point_cloud_ccs = self._transform_points_to_camera(xyz, calibration_data)

        # Filter points behind camera
        valid_mask_3d = point_cloud_ccs[:, 2] > 0.0
        if not np.any(valid_mask_3d):
            return points[~occluded]

        point_cloud_ccs_valid = point_cloud_ccs[valid_mask_3d]
        occluded_valid = occluded[valid_mask_3d]

        # Project to image coordinates
        point_cloud_ics = self._project_points_to_image(point_cloud_ccs_valid, calibration_data)

        # Apply affine transform to see where points would actually appear
        point_cloud_ics_transformed = self._apply_affine_to_points(
            point_cloud_ics, affine_transform
        )

        # Check if transformed points are within image bounds
        h, w = image_shape
        in_bounds = (
            (point_cloud_ics_transformed[:, 0] >= 0)
            & (point_cloud_ics_transformed[:, 0] < w)
            & (point_cloud_ics_transformed[:, 1] >= 0)
            & (point_cloud_ics_transformed[:, 1] < h)
        )

        # Combine occlusion filtering with visibility in transformed image space
        # Points that are occluded OR outside transformed image bounds should be filtered
        occluded_valid = occluded_valid | ~in_bounds

        # Reconstruct full mask
        full_occluded = np.zeros(len(points), dtype=bool)
        full_occluded[valid_mask_3d] = occluded_valid
        full_occluded[~valid_mask_3d] = True  # Points behind camera are considered occluded

        return points[~full_occluded]

    def _transform_points_to_camera(
        self,
        points: npt.NDArray[np.float32],
        calibration_data: CalibrationData,
    ) -> npt.NDArray[np.float32]:
        """Transform LiDAR points to the camera coordinate system.

        Args:
            points: Point coordinates in the LiDAR frame.
            calibration_data: Camera-LiDAR calibration data.

        Returns:
            Point coordinates in the camera frame.
        """
        num_points = points.shape[0]
        points_hom = np.concatenate([points, np.ones((num_points, 1), dtype=points.dtype)], axis=1)

        lidar2cam = calibration_data.lidar_to_camera_transformation
        points_cam = (lidar2cam @ points_hom.T).T

        return points_cam[:, :3]

    def _project_points_to_image(
        self,
        point_cloud_ccs: npt.NDArray[np.float32],
        calibration_data: CalibrationData,
    ) -> npt.NDArray[np.float32]:
        """Project 3D points to 2D image coordinates.

        Args:
            point_cloud_ccs: Point coordinates in the camera frame.
            calibration_data: Camera-LiDAR calibration data.

        Returns:
            Projected image coordinates.
        """
        camera_matrix = calibration_data.new_camera_matrix
        distortion_coefficients = calibration_data.distortion_coefficients

        point_cloud_ics, _ = cv2.projectPoints(
            point_cloud_ccs,
            np.zeros(3),
            np.zeros(3),
            camera_matrix,
            distortion_coefficients,
        )
        if point_cloud_ics is None:
            return np.zeros((0, 2), dtype=np.float32)

        return point_cloud_ics.reshape(-1, 2)

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
        image_shape: tuple[int, int],
        point_cloud_ics: npt.NDArray[np.float32],
        point_cloud_ccs: npt.NDArray[np.float32],
        intensities: npt.NDArray[np.float32],
    ) -> npt.NDArray[np.float32]:
        """Render the projected points as a depth and an intensity channel.

        Args:
            image_shape: Height and width of the image the points are rendered onto.
            point_cloud_ics: Projected image coordinates of the points.
            point_cloud_ccs: Camera frame coordinates of the points.
            intensities: Intensity of every point.

        Returns:
            The two channels in (2, height, width) layout, normalized to [0, 1].
        """
        h, w = image_shape
        depth_image = np.zeros((h, w), dtype=np.float32)
        intensity_image = np.zeros((h, w), dtype=np.float32)

        valid_mask = (
            (point_cloud_ics[:, 0] >= 0)
            & (point_cloud_ics[:, 0] <= w - 1)
            & (point_cloud_ics[:, 1] >= 0)
            & (point_cloud_ics[:, 1] <= h - 1)
            & (point_cloud_ccs[:, 2] > 0.0)
            & (point_cloud_ccs[:, 2] < self.max_depth)
        )

        valid_ics = point_cloud_ics[valid_mask]
        valid_ccs = point_cloud_ccs[valid_mask]
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

            center_depths = valid_ccs[:, 2] / self.max_depth

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

        # Depth is scaled by the maximum depth, intensity arrives in [0, 1] from the loader
        return np.stack([depth_image, intensity_image]).astype(np.float32)


class Affine(BaseTransform):
    """Affine transformation augmentation for images.

    Applies controlled affine distortion to the image and composes the affine into the image
    augmentation matrices, so the projection of the points follows the pixels. Automatically
    applies zoom to ensure the transformed image covers the entire viewport without black
    borders.
    """

    _required_keys = ["camera_image_data"]

    def __init__(self, probability: float = 0.5, max_distortion: float = 0.1):
        """Initialize the Affine transform.

        Args:
            probability: Probability of applying augmentation.
            max_distortion: Maximum corner displacement as fraction of image size.
        """
        super().__init__(probability=probability)
        self.max_distortion = max_distortion

    def transform(self, model_gt_sample: ModelGTSample) -> ModelGTSample:
        """Warp every image of the sample by a random affine.

        Args:
            model_gt_sample: ModelGTSample instance holding loaded images.

        Returns:
            Updated ModelGTSample instance with warped images.
        """
        # This is checked in the _validate_required_keys()
        camera_image_data: BaseImages = (
            model_gt_sample.camera_image_data  # type: ignore[reportOptionalMemberAccess]
        )

        images = []
        pixel_affines = []
        for image in camera_image_data.images:
            affine_matrix, warped = self.warp_image(np.transpose(image.numpy(), (1, 2, 0)))
            images.append(torch.from_numpy(np.transpose(warped, (2, 0, 1)).copy()))
            pixel_affines.append(torch.from_numpy(affine_matrix.astype(np.float32)))

        augmentation_matrices = BaseImages.homogeneous_image_augmentation_matrices(
            torch.stack(pixel_affines)
        )
        return model_gt_sample._replace(
            camera_image_data=BaseImages.model_validate(
                camera_image_data.model_copy(
                    update={
                        "images": torch.stack(images),
                        "image_augmentation_matrices": augmentation_matrices
                        @ camera_image_data.image_augmentation_matrices,
                    }
                )
            )
        )

    def warp_image(
        self, image: npt.NDArray[np.float32]
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float32]]:
        """Warp one image by a random affine and return the affine it was warped with.

        Args:
            image: Image in (height, width, num_channels) layout.

        Returns:
            tuple: The 3x3 affine and the warped image.
        """
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

        return affine_matrix_3x3, image


class SaveFusionPreview(BaseTransform):
    """Write a preview of the projected points over the image, for visual inspection."""

    _required_keys = ["camera_image_data"]

    def __init__(
        self,
        out_dir: str,
        probability: float = 0.0,
        max_depth: float = 128.0,
        alpha: float = 1.0,
        depth_colormap: str = "jet",
    ) -> None:
        """Initialize the SaveFusionPreview transform.

        Args:
            out_dir: Directory the previews are written to.
            probability: Probability of writing a preview for a sample.
            max_depth: Depth the color map saturates at, in meters.
            alpha: Opacity of the projected points over the image.
            depth_colormap: Name of the OpenCV color map applied to the depth.
        """
        super().__init__(probability=probability)
        self.out_dir = Path(out_dir)
        self.max_depth = max_depth
        self.alpha = alpha
        self.depth_colormap = getattr(cv2, f"COLORMAP_{depth_colormap.upper()}")

    def transform(self, model_gt_sample: ModelGTSample) -> ModelGTSample:
        """Write one preview per camera of the sample.

        Args:
            model_gt_sample: ModelGTSample instance whose images carry the projected channels.

        Returns:
            The ModelGTSample instance unchanged.
        """
        # This is checked in the _validate_required_keys()
        camera_image_data: BaseImages = (
            model_gt_sample.camera_image_data  # type: ignore[reportOptionalMemberAccess]
        )
        if camera_image_data.depth_maps is None:
            raise ValueError(
                "SaveFusionPreview runs after the projection, the images carry no depth maps."
            )

        self.out_dir.mkdir(parents=True, exist_ok=True)
        for index, camera_name in enumerate(camera_image_data.camera_names):
            overlay = self.build_overlay(
                np.transpose(camera_image_data.images[index].numpy(), (1, 2, 0)),
                camera_image_data.depth_maps[index, 0].numpy(),
            )
            timestamp = float(camera_image_data.timestamps[index])
            cv2.imwrite(
                str(self.out_dir / f"{camera_name}_{timestamp:.6f}.png"),
                cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR),
            )
        return model_gt_sample

    def build_overlay(
        self, image: npt.NDArray[np.float32], depth: npt.NDArray[np.float32]
    ) -> npt.NDArray[np.uint8]:
        """Color the projected depth and blend it over the image.

        Args:
            image: Image in (height, width, num_channels) layout with values in [0, 1].
            depth: Projected depth channel with values in [0, 1].

        Returns:
            npt.NDArray[np.uint8]: The blended preview.
        """
        overlay = (np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)
        colored = cv2.applyColorMap(
            (np.clip(depth, 0.0, 1.0) * 255.0).astype(np.uint8),
            self.depth_colormap,
        )
        points = depth > 0.0
        overlay[points] = (
            self.alpha * colored[points] + (1.0 - self.alpha) * overlay[points]
        ).astype(np.uint8)
        return overlay
