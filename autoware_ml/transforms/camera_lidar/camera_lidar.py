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

"""Camera-LiDAR fusion transforms.

This module contains the calibration-misalignment augmentation, the projection of the lidar
points onto the images and preview utilities for camera-LiDAR fusion inputs.
"""

from __future__ import annotations

from collections.abc import Sequence
import math
from pathlib import Path
from typing import Literal

import cv2
from jaxtyping import Float32, Int64
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, NonNegativeFloat, model_validator
import torch

from autoware_ml.dataclasses.batch.sample_batch import ModelGTSample
from autoware_ml.geometry.cameras.base_images import BaseImages
from autoware_ml.transforms.base import BaseTransform
from autoware_ml.transforms.camera.resize import ImageSpaceTransform
from autoware_ml.types.geometry import PointFeatureName
from autoware_ml.utils.calibration import CalibrationStatus


TranslationAxis = Literal["x", "y", "z"]
_TRANSLATION_AXES: tuple[TranslationAxis, ...] = ("x", "y", "z")


def _validate_ordered_range(name: str, min_value: float, max_value: float) -> None:
    """Validate that a configured range is well ordered.

    Args:
        name: Range name without the ``min_``/``max_`` prefix, used in the error message.
        min_value: Lower bound.
        max_value: Upper bound.

    Raises:
        ValueError: If the lower bound exceeds the upper bound.
    """
    if min_value > max_value:
        raise ValueError(f"min_{name} ({min_value}) must be <= max_{name} ({max_value})")


class AxisActivation(BaseModel):
    """Which rotation and translation components the misalignment perturbs."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    roll: bool = False
    pitch: bool = False
    yaw: bool = False
    x: bool = False
    y: bool = False
    z: bool = False


class MagnitudeRange(BaseModel):
    """Negative and positive magnitude ranges of a single misalignment component.

    All values are non-negative magnitudes. The ``_neg`` range is negated when applied,
    which keeps ``min < max`` intuitive in the config. Rotations are in degrees and
    translations in meters.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    min_neg: NonNegativeFloat = 0.0
    max_neg: NonNegativeFloat = 0.0
    min_pos: NonNegativeFloat = 0.0
    max_pos: NonNegativeFloat = 0.0

    @model_validator(mode="after")
    def _validate_ranges(self) -> MagnitudeRange:
        """Ensure both the negative and the positive range are well ordered.

        Returns:
            The validated model.

        Raises:
            ValueError: If a lower bound exceeds its upper bound.
        """
        _validate_ordered_range("neg", self.min_neg, self.max_neg)
        _validate_ordered_range("pos", self.min_pos, self.max_pos)
        return self


class TranslationRange(BaseModel):
    """Negative and positive magnitude ranges of the x, y and z translations in meters.

    All values are non-negative magnitudes. The ``_neg`` values are negated when applied.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    min_x_neg: NonNegativeFloat = 0.0
    max_x_neg: NonNegativeFloat = 0.0
    min_x_pos: NonNegativeFloat = 0.0
    max_x_pos: NonNegativeFloat = 0.0
    min_y_neg: NonNegativeFloat = 0.0
    max_y_neg: NonNegativeFloat = 0.0
    min_y_pos: NonNegativeFloat = 0.0
    max_y_pos: NonNegativeFloat = 0.0
    min_z_neg: NonNegativeFloat = 0.0
    max_z_neg: NonNegativeFloat = 0.0
    min_z_pos: NonNegativeFloat = 0.0
    max_z_pos: NonNegativeFloat = 0.0

    @model_validator(mode="after")
    def _validate_ranges(self) -> TranslationRange:
        """Ensure every per-axis negative and positive range is well ordered.

        Returns:
            The validated model.

        Raises:
            ValueError: If a lower bound exceeds its upper bound.
        """
        for axis in _TRANSLATION_AXES:
            for sign in ("neg", "pos"):
                name = f"{axis}_{sign}"
                _validate_ordered_range(
                    name, getattr(self, f"min_{name}"), getattr(self, f"max_{name}")
                )
        return self

    def axis(self, axis: TranslationAxis) -> MagnitudeRange:
        """Return the ranges of a single translation axis.

        Args:
            axis: One of ``x``, ``y`` or ``z``.

        Returns:
            The four bounds of ``axis`` as a ``MagnitudeRange``.
        """
        return MagnitudeRange(
            min_neg=getattr(self, f"min_{axis}_neg"),
            max_neg=getattr(self, f"max_{axis}_neg"),
            min_pos=getattr(self, f"min_{axis}_pos"),
            max_pos=getattr(self, f"max_{axis}_pos"),
        )


class CalibrationMisalignment(BaseTransform):
    """Calibration misalignment augmentation for camera-LiDAR calibration.

    Each rotation (roll, pitch, yaw) and translation (x, y, z) component has separate
    negative and positive ranges. During augmentation, one of the two ranges is randomly
    selected for each component. Each component can be individually activated or
    deactivated, and every camera of the sample draws its own perturbation.

    All parameters are specified as positive magnitudes. The ``_neg`` suffix indicates the
    value will be negated when applied. This keeps min < max intuitive in the config.
    Parameters are grouped into the frozen pydantic models :class:`AxisActivation`,
    :class:`MagnitudeRange` and :class:`TranslationRange`, which carry the validation. From
    hydra, build them with nested ``_target_`` entries.

    Required keys:
        - camera_image_data: BaseImages with ``lidar2cams`` and ``augmented_camera_intrinsics``.

    Generated keys:
        - camera_image_data.calibration_statuses: per-camera ``CalibrationStatus`` value.
        - camera_image_data.noises: per-camera 4x4 noise transform (when applied).
        - camera_image_data.lidar2cams / lidar2images: perturbed by the noise (when applied).
    """

    _required_keys = ["camera_image_data"]

    def __init__(
        self,
        *,
        probability: float,
        activate: AxisActivation = AxisActivation(),
        roll: MagnitudeRange = MagnitudeRange(),
        pitch: MagnitudeRange = MagnitudeRange(),
        yaw: MagnitudeRange = MagnitudeRange(),
        translation: TranslationRange = TranslationRange(),
    ) -> None:
        """Initialize the CalibrationMisalignment transform.

        The group models are frozen, so sharing the default instances between transforms is
        safe. Validation of the magnitudes happens when the groups are constructed.

        Args:
            probability: Probability of applying augmentation.
            activate: Which components to perturb. Defaults to none.
            roll: Roll ranges in degrees. Defaults to zero.
            pitch: Pitch ranges in degrees. Defaults to zero.
            yaw: Yaw ranges in degrees. Defaults to zero.
            translation: x, y and z ranges in meters. Defaults to zero.
        """
        super().__init__(probability=probability)

        self.activate = activate
        self.roll = roll
        self.pitch = pitch
        self.yaw = yaw
        self.translation = translation

    def on_skip(self, model_gt_sample: ModelGTSample) -> ModelGTSample:
        """Mark every camera as calibrated when the augmentation is skipped.

        Args:
            model_gt_sample: Sample whose ``camera_image_data`` receives the statuses.

        Returns:
            Updated ModelGTSample with all cameras flagged ``CalibrationStatus.CALIBRATED``.
        """
        assert model_gt_sample.camera_image_data is not None
        camera_image_data = model_gt_sample.camera_image_data
        num_cameras = camera_image_data.lidar2cams.shape[0]
        calibration_statuses = torch.full(
            (num_cameras,), CalibrationStatus.CALIBRATED.value, dtype=torch.int64
        )
        return model_gt_sample._replace(
            camera_image_data=BaseImages.model_validate(
                camera_image_data.model_copy(update={"calibration_statuses": calibration_statuses})
            )
        )

    def transform(self, model_gt_sample: ModelGTSample) -> ModelGTSample:
        """Perturb the lidar-to-camera extrinsics of every camera.

        Args:
            model_gt_sample: Sample holding ``camera_image_data``.

        Returns:
            Updated ModelGTSample with perturbed ``lidar2cams`` and ``lidar2images``, the
            per-camera noise transforms and all cameras flagged ``MISCALIBRATED``.
        """
        assert model_gt_sample.camera_image_data is not None
        camera_image_data = model_gt_sample.camera_image_data
        lidar2cams = camera_image_data.lidar2cams
        num_cameras = lidar2cams.shape[0]

        noises = torch.stack([self.sample_noise_transform() for _ in range(num_cameras)]).to(
            lidar2cams
        )
        calibration_statuses = torch.full(
            (num_cameras,), CalibrationStatus.MISCALIBRATED.value, dtype=torch.int64
        )
        return model_gt_sample._replace(
            camera_image_data=self._apply_noises(camera_image_data, noises, calibration_statuses)
        )

    @staticmethod
    def _apply_noises(
        camera_image_data: BaseImages,
        noises: Float32[torch.Tensor, "num_cameras 4 4"],
        calibration_statuses: Int64[torch.Tensor, " num_cameras"],
    ) -> BaseImages:
        """Compose the noise into the extrinsics and refresh the projections.

        The noise acts in the camera frame, ``T_noisy = T_noise @ T_lidar2cam``, so an
        x-translation shifts projected points along the camera's x-axis. ``lidar2images`` is
        rebuilt from ``augmented_camera_intrinsics`` so it stays consistent with any
        image-space augmentation that already ran.

        Args:
            camera_image_data: Container whose extrinsics are perturbed.
            noises: Per-camera 4x4 noise transforms.
            calibration_statuses: Per-camera status written alongside the noise.

        Returns:
            A validated copy of ``camera_image_data`` with the perturbation applied.
        """
        noisy_lidar2cams = noises @ camera_image_data.lidar2cams

        intrinsics = camera_image_data.augmented_camera_intrinsics
        homogeneous_intrinsics = torch.eye(4, dtype=intrinsics.dtype, device=intrinsics.device)
        homogeneous_intrinsics = homogeneous_intrinsics.repeat(intrinsics.shape[0], 1, 1)
        homogeneous_intrinsics[:, :3, :3] = intrinsics

        return BaseImages.model_validate(
            camera_image_data.model_copy(
                update={
                    "lidar2cams": noisy_lidar2cams,
                    "lidar2images": homogeneous_intrinsics @ noisy_lidar2cams,
                    "noises": noises,
                    "calibration_statuses": calibration_statuses,
                }
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

        sample = torch.nn.init.trunc_normal_(
            torch.empty(1, dtype=torch.float64), mean=center, std=scale, a=min_value, b=max_value
        )
        return float(sample.item())

    def _sample_component(self, bounds: MagnitudeRange) -> float:
        """Sample a component value from either negative or positive range.

        Randomly selects between negative and positive range, then samples from a truncated
        gaussian within that range. All bounds are positive magnitudes; negative range
        values are negated after sampling.

        Args:
            bounds: Negative and positive magnitude ranges of the component.

        Returns:
            Sampled value (negative if from neg range, positive if from pos range).
        """
        use_negative = bool(torch.rand(1).item() > 0.5)
        min_val, max_val = (
            (bounds.min_neg, bounds.max_neg) if use_negative else (bounds.min_pos, bounds.max_pos)
        )
        sign = -1.0 if use_negative else 1.0

        if min_val >= max_val:
            return sign * min_val
        # Center towards the least extreme value (the threshold).
        value = self.bounded_gaussian(
            center=min_val,
            min_value=min_val,
            max_value=max_val,
            scale=(max_val - min_val) / 1.5,
        )
        return sign * value

    @staticmethod
    def rotation_matrix(
        roll_rad: float, pitch_rad: float, yaw_rad: float
    ) -> Float32[torch.Tensor, "3 3"]:
        """Build a rotation matrix from roll, pitch and yaw about the static x, y and z axes.

        Equivalent to ``transforms3d.euler.euler2mat(roll, pitch, yaw, axes="sxyz")``:
        ``R = Rz(yaw) @ Ry(pitch) @ Rx(roll)``.

        Args:
            roll_rad: Rotation about x in radians.
            pitch_rad: Rotation about y in radians.
            yaw_rad: Rotation about z in radians.

        Returns:
            The 3x3 rotation matrix.
        """
        cos_r, sin_r = math.cos(roll_rad), math.sin(roll_rad)
        cos_p, sin_p = math.cos(pitch_rad), math.sin(pitch_rad)
        cos_y, sin_y = math.cos(yaw_rad), math.sin(yaw_rad)
        rotation_x = torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, cos_r, -sin_r], [0.0, sin_r, cos_r]], dtype=torch.float64
        )
        rotation_y = torch.tensor(
            [[cos_p, 0.0, sin_p], [0.0, 1.0, 0.0], [-sin_p, 0.0, cos_p]], dtype=torch.float64
        )
        rotation_z = torch.tensor(
            [[cos_y, -sin_y, 0.0], [sin_y, cos_y, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float64
        )
        return (rotation_z @ rotation_y @ rotation_x).to(torch.float32)

    def sample_noise_transform(self) -> Float32[torch.Tensor, "4 4"]:
        """Draw one 4x4 noise transform from the configured ranges.

        Each component randomly selects between its negative and positive range. Only
        activated components are perturbed, the rest stay at zero.

        Returns:
            Homogeneous transform holding the sampled rotation and translation.
        """
        roll = self._sample_component(self.roll) if self.activate.roll else 0.0
        pitch = self._sample_component(self.pitch) if self.activate.pitch else 0.0
        yaw = self._sample_component(self.yaw) if self.activate.yaw else 0.0
        translation_x, translation_y, translation_z = (
            self._sample_component(self.translation.axis(axis))
            if getattr(self.activate, axis)
            else 0.0
            for axis in _TRANSLATION_AXES
        )

        noise_transform = torch.eye(4, dtype=torch.float32)
        noise_transform[:3, :3] = self.rotation_matrix(
            math.radians(roll), math.radians(pitch), math.radians(yaw)
        )
        noise_transform[:3, 3] = torch.tensor(
            [translation_x, translation_y, translation_z], dtype=torch.float32
        )
        return noise_transform

    def alter_calibration(
        self, transform: Float32[torch.Tensor, "4 4"]
    ) -> tuple[Float32[torch.Tensor, "4 4"], Float32[torch.Tensor, "4 4"]]:
        """Apply random noise to a single 4x4 lidar-to-camera transform.

        Args:
            transform: The 4x4 lidar-to-camera transform to perturb.

        Returns:
            Tuple of the perturbed transform ``T_noise @ transform`` and the noise itself.

        Raises:
            ValueError: If ``transform`` is not a 4x4 matrix.
        """
        if tuple(transform.shape) != (4, 4):
            raise ValueError(f"Transform must be 4x4 matrix, got shape {tuple(transform.shape)}")
        noise_transform = self.sample_noise_transform().to(transform)
        return noise_transform @ transform, noise_transform


class LidarCameraFusion(BaseTransform):
    """Project the lidar points onto every camera image as depth and intensity channels.

    Projects the lidar points onto each camera and writes a depth and an intensity channel
    next to the RGB image, in ``camera_image_data.depth_maps``. The batch stacks them onto the
    images, so the models read five-channel RGBDI images. The projection uses the
    ``augmented_camera_intrinsics`` and the distortion coefficients of every camera, so it
    follows the undistortion and the image-space augmentations (crop, scale, affine) that ran
    earlier in the pipeline. This transform must therefore run after every transform that
    changes the image geometry, and before the consumers of the fused images.

    The depth channel holds ``depth / max_depth`` and the intensity channel the intensity of
    the point, which the point cloud loader serves in unit range, so both channels share the
    unit pixel range of the images. Points behind the camera or beyond ``max_depth`` are
    dropped, and when ``ego_box`` is given, points whose ray from the camera crosses the ego
    vehicle box are dropped as occluded.

    Required keys:
        - camera_image_data: RGB images with their calibration, ``noises`` when the
          calibration-misalignment augmentation ran.
        - point_cloud_data: points with XYZ and intensity features.

    Generated keys:
        - camera_image_data.depth_maps: (num_cameras, 2, height, width) float32 depth and
          intensity channels.
    """

    _required_keys = ["camera_image_data", "point_cloud_data"]

    def __init__(
        self,
        *,
        max_depth: float = 128.0,
        dilation_size: int = 1,
        ego_box: Sequence[float] | None = None,
        occlusion_adjust_margin: float = 0.01,
    ) -> None:
        """Initialize the LidarCameraFusion transform.

        Args:
            max_depth: Maximum depth of the projected points in meters.
            dilation_size: Half size of the square patch painted around every projected point.
            ego_box: Ego vehicle box ``[x_min, y_min, z_min, x_max, y_max, z_max]`` in the lidar
                frame used to drop occluded points, ``None`` to keep every point.
            occlusion_adjust_margin: Distance in meters kept between a camera lying inside the
                ego box and the box wall moved behind it.

        Raises:
            ValueError: If ``ego_box`` does not hold six values or a bound is not positive.
        """
        super().__init__(probability=None)
        if ego_box is not None and len(ego_box) != 6:
            raise ValueError(f"ego_box must hold six values, got {len(ego_box)}")
        if max_depth <= 0.0:
            raise ValueError(f"max_depth must be positive, got {max_depth}")
        if dilation_size < 0:
            raise ValueError(f"dilation_size must be >= 0, got {dilation_size}")

        self.max_depth = max_depth
        self.dilation_size = dilation_size
        self.ego_box = None if ego_box is None else tuple(float(value) for value in ego_box)
        self.occlusion_adjust_margin = occlusion_adjust_margin

    def transform(self, model_gt_sample: ModelGTSample) -> ModelGTSample:
        """Write the projected depth and intensity channels of every camera image.

        Args:
            model_gt_sample: Sample holding ``camera_image_data`` and ``point_cloud_data``.

        Returns:
            Updated ModelGTSample whose ``camera_image_data.depth_maps`` hold the depth and
            intensity channels of every camera.

        Raises:
            ValueError: If the points carry no intensity feature.
        """
        assert model_gt_sample.camera_image_data is not None
        assert model_gt_sample.point_cloud_data is not None
        camera_image_data = model_gt_sample.camera_image_data
        point_cloud_data = model_gt_sample.point_cloud_data

        feature_names = list(point_cloud_data.point_feature_names)
        if PointFeatureName.INTENSITY not in feature_names:
            raise ValueError(
                f"{self.__class__.__name__}: points must carry an intensity feature, "
                f"got {feature_names}"
            )
        points = point_cloud_data.points.detach().cpu().numpy().astype(np.float32)
        xyz = points[:, :3]
        intensities = points[:, feature_names.index(PointFeatureName.INTENSITY)]

        images = camera_image_data.images
        num_cameras, _, height, width = images.shape
        lidar2cams = camera_image_data.lidar2cams.cpu().numpy().astype(np.float64)
        camera_matrices = camera_image_data.augmented_camera_intrinsics.cpu().numpy()
        camera_matrices = camera_matrices.astype(np.float64)
        noises = (
            None
            if camera_image_data.noises is None
            else camera_image_data.noises.cpu().numpy().astype(np.float64)
        )

        depth_maps = []
        for index in range(num_cameras):
            distortion_coefficients = camera_image_data.distortion_coefficients[index]
            depth_channel, intensity_channel = self.render_lidar_channels(
                xyz=xyz,
                intensities=intensities,
                lidar2cam=lidar2cams[index],
                noise=None if noises is None else noises[index],
                camera_matrix=camera_matrices[index],
                distortion_coefficients=distortion_coefficients.cpu().numpy().astype(np.float64),
                image_size=(int(height), int(width)),
            )
            depth_maps.append(
                torch.from_numpy(np.stack([depth_channel, intensity_channel])).to(images)
            )

        # model_copy does not validate what it is given, so the copy is validated explicitly.
        return model_gt_sample._replace(
            camera_image_data=BaseImages.model_validate(
                camera_image_data.model_copy(update={"depth_maps": torch.stack(depth_maps, dim=0)})
            )
        )

    def render_lidar_channels(
        self,
        xyz: npt.NDArray[np.float32],
        intensities: npt.NDArray[np.float32],
        lidar2cam: npt.NDArray[np.float64],
        noise: npt.NDArray[np.float64] | None,
        camera_matrix: npt.NDArray[np.float64],
        distortion_coefficients: npt.NDArray[np.float64],
        image_size: tuple[int, int],
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
        """Render the depth and intensity channels of one camera.

        Args:
            xyz: Point coordinates in the lidar frame, shape ``(num_points, 3)``.
            intensities: Point intensities in unit range, shape ``(num_points,)``.
            lidar2cam: 4x4 lidar-to-camera transform, including the misalignment noise when
                the augmentation ran.
            noise: 4x4 misalignment noise composed into ``lidar2cam``, ``None`` when none ran.
            camera_matrix: 3x3 intrinsics of the image plane the channels are rendered for.
            distortion_coefficients: OpenCV distortion coefficients, empty for undistorted
                images.
            image_size: Height and width of the channels.

        Returns:
            Depth and intensity channels of shape ``(height, width)`` in unit range. Pixels
            without a point are zero.
        """
        if self.ego_box is not None:
            camera_center = self.camera_center_in_lidar(lidar2cam, noise)
            visible = ~self.ego_occlusion_mask(xyz, camera_center)
            xyz = xyz[visible]
            intensities = intensities[visible]

        points_cam = self.transform_points_to_camera(xyz, lidar2cam)
        in_front = points_cam[:, 2] > 0.0
        points_cam = points_cam[in_front]
        intensities = intensities[in_front]

        pixels = self.project_points_to_image(points_cam, camera_matrix, distortion_coefficients)
        return self.rasterize_points(pixels, points_cam[:, 2], intensities, image_size)

    @staticmethod
    def camera_center_in_lidar(
        lidar2cam: npt.NDArray[np.float64], noise: npt.NDArray[np.float64] | None
    ) -> npt.NDArray[np.float64]:
        """Return the true camera center in the lidar frame.

        The misalignment noise acts in the camera frame, ``T_noisy = T_noise @ T_lidar2cam``,
        so it is undone before inverting the transform to keep the occlusion filtering
        anchored at the physical camera position.

        Args:
            lidar2cam: 4x4 lidar-to-camera transform, noisy when ``noise`` is given.
            noise: 4x4 noise composed into ``lidar2cam``, ``None`` when none was applied.

        Returns:
            Camera center in the lidar frame, shape ``(3,)``.
        """
        lidar2cam_true = lidar2cam if noise is None else np.linalg.inv(noise) @ lidar2cam
        rotation = lidar2cam_true[:3, :3]
        translation = lidar2cam_true[:3, 3]
        return -rotation.T @ translation

    def ego_occlusion_mask(
        self, xyz: npt.NDArray[np.float32], camera_center: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.bool_]:
        """Flag the points whose ray from the camera crosses the ego vehicle box.

        When the camera lies inside the ego box, the closest x and y walls are moved
        ``occlusion_adjust_margin`` behind the camera so the box does not occlude everything.
        The z bounds are left untouched.

        Args:
            xyz: Point coordinates in the lidar frame, shape ``(num_points, 3)``.
            camera_center: Camera center in the lidar frame, shape ``(3,)``.

        Returns:
            Boolean mask, ``True`` for occluded points.
        """
        assert self.ego_box is not None
        box_min = np.array(self.ego_box[:3], dtype=np.float64)
        box_max = np.array(self.ego_box[3:], dtype=np.float64)

        if np.all(camera_center >= box_min) and np.all(camera_center <= box_max):
            distance_to_min = camera_center - box_min
            distance_to_max = box_max - camera_center
            for axis in range(2):
                if distance_to_min[axis] < distance_to_max[axis]:
                    box_min[axis] = camera_center[axis] + self.occlusion_adjust_margin
                else:
                    box_max[axis] = camera_center[axis] - self.occlusion_adjust_margin

        # Slab method: the ray from the camera to each point is parameterized by t in [0, 1].
        ray_directions = xyz.astype(np.float64) - camera_center
        with np.errstate(divide="ignore", invalid="ignore"):
            t_near = (box_min - camera_center) / ray_directions
            t_far = (box_max - camera_center) / ray_directions
        t_enter = np.max(np.minimum(t_near, t_far), axis=1)
        t_exit = np.min(np.maximum(t_near, t_far), axis=1)

        hits_box = (t_enter <= t_exit) & (t_exit >= 0.0)
        # The ray enters the box before reaching the point.
        return hits_box & (t_enter < 0.999)

    @staticmethod
    def transform_points_to_camera(
        xyz: npt.NDArray[np.float32], lidar2cam: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Transform lidar points into the camera frame.

        Args:
            xyz: Point coordinates in the lidar frame, shape ``(num_points, 3)``.
            lidar2cam: 4x4 lidar-to-camera transform.

        Returns:
            Point coordinates in the camera frame, shape ``(num_points, 3)``.
        """
        return xyz.astype(np.float64) @ lidar2cam[:3, :3].T + lidar2cam[:3, 3]

    @staticmethod
    def project_points_to_image(
        points_cam: npt.NDArray[np.float64],
        camera_matrix: npt.NDArray[np.float64],
        distortion_coefficients: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Project camera-frame points onto the image plane.

        Args:
            points_cam: Point coordinates in the camera frame, shape ``(num_points, 3)``.
            camera_matrix: 3x3 camera intrinsics.
            distortion_coefficients: OpenCV distortion coefficients, empty for none.

        Returns:
            Pixel coordinates ``(u, v)``, shape ``(num_points, 2)``.
        """
        if points_cam.shape[0] == 0:
            return np.zeros((0, 2), dtype=np.float64)
        coefficients = distortion_coefficients if distortion_coefficients.size > 0 else None
        pixels, _ = cv2.projectPoints(
            points_cam.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), camera_matrix, coefficients
        )
        return pixels.reshape(-1, 2)

    def rasterize_points(
        self,
        pixels: npt.NDArray[np.float64],
        depths: npt.NDArray[np.float64],
        intensities: npt.NDArray[np.float32],
        image_size: tuple[int, int],
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
        """Paint the projected points into depth and intensity channels.

        Every point paints a ``(2 * dilation_size + 1)`` square patch. Where patches overlap,
        the closest point wins.

        Args:
            pixels: Pixel coordinates ``(u, v)`` of the points, shape ``(num_points, 2)``.
            depths: Depth of the points along the optical axis, shape ``(num_points,)``.
            intensities: Intensities of the points in unit range, shape ``(num_points,)``.
            image_size: Height and width of the channels.

        Returns:
            Depth channel scaled by ``max_depth`` and intensity channel clipped to unit range,
            both of shape ``(height, width)``.
        """
        height, width = image_size
        depth_channel = np.zeros((height, width), dtype=np.float32)
        intensity_channel = np.zeros((height, width), dtype=np.float32)

        valid = (
            (pixels[:, 0] >= 0)
            & (pixels[:, 0] <= width - 1)
            & (pixels[:, 1] >= 0)
            & (pixels[:, 1] <= height - 1)
            & (depths > 0.0)
            & (depths < self.max_depth)
        )
        if not np.any(valid):
            return depth_channel, intensity_channel

        pixels = pixels[valid]
        depths = depths[valid]
        intensities = intensities[valid]

        row_offsets, col_offsets = np.mgrid[
            -self.dilation_size : self.dilation_size + 1,
            -self.dilation_size : self.dilation_size + 1,
        ]
        patch_rows = pixels[:, 1].astype(np.int32)[:, None] + row_offsets.flatten()[None, :]
        patch_cols = pixels[:, 0].astype(np.int32)[:, None] + col_offsets.flatten()[None, :]
        in_bounds = (
            (patch_rows >= 0) & (patch_rows < height) & (patch_cols >= 0) & (patch_cols < width)
        )

        patch_depths = np.broadcast_to(depths[:, None], patch_rows.shape)[in_bounds]
        patch_intensities = np.broadcast_to(intensities[:, None], patch_rows.shape)[in_bounds]
        patch_rows = patch_rows[in_bounds]
        patch_cols = patch_cols[in_bounds]

        # Paint far to near so the closest point ends up on top.
        order = np.argsort(patch_depths)[::-1]
        depth_channel[patch_rows[order], patch_cols[order]] = patch_depths[order] / self.max_depth
        intensity_channel[patch_rows[order], patch_cols[order]] = np.clip(
            patch_intensities[order], 0.0, 1.0
        )
        return depth_channel, intensity_channel


class Affine(ImageSpaceTransform):
    """Random affine distortion of the camera images.

    Displaces the three reference corners of every image by up to ``max_distortion`` of the
    image size and warps the pixels with the resulting affine. A zoom is folded in so the
    warped image covers the whole viewport without black borders. Every camera draws its own
    affine. The pixel affine is composed into ``augmented_camera_intrinsics``,
    ``image_augmentation_matrices`` and ``lidar2images`` like the other image-space
    transforms, so the lidar projections keep following the pixels.

    Required keys:
        - camera_image_data: images with their calibration.

    Generated keys:
        - camera_image_data.images: warped images (when applied).
        - camera_image_data.augmented_camera_intrinsics / image_augmentation_matrices /
          lidar2images: composed with the affine (when applied).
    """

    _required_keys = ["camera_image_data"]

    def __init__(self, *, probability: float | None = 0.5, max_distortion: float = 0.1) -> None:
        """Initialize the Affine transform.

        Args:
            probability: Probability of applying the augmentation, ``None`` to always run.
            max_distortion: Maximum corner displacement as a fraction of the image size.
        """
        super().__init__(probability=probability)
        self.max_distortion = max_distortion

    def transform(self, model_gt_sample: ModelGTSample) -> ModelGTSample:
        """Warp every camera image with its own random affine.

        Args:
            model_gt_sample: Sample holding ``camera_image_data``.

        Returns:
            Updated ModelGTSample with the warped images and the composed calibration.
        """
        assert model_gt_sample.camera_image_data is not None
        camera_image_data = model_gt_sample.camera_image_data
        images = camera_image_data.images
        num_cameras, _, height, width = images.shape

        pixel_affines = np.stack(
            [self.sample_affine_matrix(int(height), int(width)) for _ in range(num_cameras)]
        )
        warped_images = []
        for index in range(num_cameras):
            image_hwc = images[index].permute(1, 2, 0).contiguous().cpu().numpy()
            warped_hwc = cv2.warpAffine(
                image_hwc,
                pixel_affines[index, :2],
                (int(width), int(height)),
                borderMode=cv2.BORDER_CONSTANT,
            ).reshape(int(height), int(width), -1)
            warped_images.append(torch.from_numpy(warped_hwc).permute(2, 0, 1))

        return model_gt_sample._replace(
            camera_image_data=self.apply_image_space_transform(
                camera_image_data,
                torch.stack(warped_images, dim=0).to(images),
                torch.from_numpy(pixel_affines).to(torch.float32),
            )
        )

    def sample_affine_matrix(self, height: int, width: int) -> npt.NDArray[np.float64]:
        """Draw one random 3x3 pixel affine covering the whole viewport.

        Args:
            height: Image height in pixels.
            width: Image width in pixels.

        Returns:
            3x3 affine mapping a pixel of the input image onto its location in the warped
            image.
        """
        max_offset = np.array([[self.max_distortion * width, self.max_distortion * height]] * 3)
        source_points = np.float32([[0, 0], [width - 1, 0], [0, height - 1]])
        target_points = source_points + np.random.uniform(-max_offset, max_offset).astype(
            np.float32
        )
        affine_2x3 = cv2.getAffineTransform(source_points, target_points)

        # Map the viewport corners back into the source image to find the zoom that keeps
        # them inside the image bounds.
        inverse_affine = cv2.invertAffineTransform(affine_2x3)
        corners = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float64)
        source_corners = np.hstack([corners, np.ones((4, 1))]) @ inverse_affine.T
        center_x, center_y = width / 2.0, height / 2.0
        scale = max(
            1.0,
            np.max(np.abs(source_corners[:, 0] - center_x)) / center_x,
            np.max(np.abs(source_corners[:, 1] - center_y)) / center_y,
        )
        zoom = np.array(
            [[scale, 0.0, center_x * (1 - scale)], [0.0, scale, center_y * (1 - scale)], [0, 0, 1]],
            dtype=np.float64,
        )

        affine_3x3 = np.eye(3, dtype=np.float64)
        affine_3x3[:2, :3] = affine_2x3
        return affine_3x3 @ zoom


class SaveFusionPreview(BaseTransform):
    """Save preview images of the projected points over the camera images.

    Writes two overlays per camera: the RGB image with the depth points and with the
    intensity points, each colorized with a colormap. The files are named after the camera
    and its timestamp, suffixed with the calibration status when the misalignment
    augmentation ran.

    Required keys:
        - camera_image_data: images in unit range with the ``depth_maps`` written by
          ``LidarCameraFusion``, ``calibration_statuses`` when the misalignment augmentation
          ran.

    Generated keys:
        - None (pass-through transform, only writes files to disk).
    """

    _required_keys = ["camera_image_data"]

    def __init__(
        self,
        *,
        probability: float | None = 1.0,
        out_dir: str | Path = "",
        max_depth: float = 128.0,
        alpha: float = 0.5,
        depth_colormap: str = "turbo",
        intensity_colormap: str = "jet",
    ) -> None:
        """Initialize the SaveFusionPreview transform.

        Args:
            probability: Probability of saving the previews of a sample, ``None`` to always
                save.
            out_dir: Output directory of the preview images, created if missing.
            max_depth: ``max_depth`` used by ``LidarCameraFusion``, to recover the depth.
            alpha: Blending factor of the overlay (0.0 = RGB only, 1.0 = points only).
            depth_colormap: Matplotlib colormap of the depth overlay.
            intensity_colormap: Matplotlib colormap of the intensity overlay.
        """
        super().__init__(probability=probability)
        self.out_dir = Path(out_dir)
        self.max_depth = max_depth
        self.alpha = alpha
        self.depth_cmap = plt.get_cmap(depth_colormap)
        self.intensity_cmap = plt.get_cmap(intensity_colormap)

        self.out_dir.mkdir(parents=True, exist_ok=True)

    def transform(self, model_gt_sample: ModelGTSample) -> ModelGTSample:
        """Save the depth and intensity previews of every camera.

        Args:
            model_gt_sample: Sample holding the ``camera_image_data`` with its depth maps.

        Returns:
            The unmodified ModelGTSample.

        Raises:
            ValueError: If the images carry no depth maps.
        """
        assert model_gt_sample.camera_image_data is not None
        camera_image_data = model_gt_sample.camera_image_data
        if camera_image_data.depth_maps is None:
            raise ValueError(
                f"{self.__class__.__name__} runs after LidarCameraFusion, the images carry "
                "no depth maps"
            )

        statuses = camera_image_data.calibration_statuses
        for index, camera_name in enumerate(camera_image_data.camera_names):
            timestamp = float(camera_image_data.timestamps[index])
            self.save_preview(
                image=camera_image_data.images[index].detach().cpu().numpy(),
                depth_map=camera_image_data.depth_maps[index].detach().cpu().numpy(),
                base_name=f"{camera_name}_{timestamp:.6f}",
                calibration_status=None if statuses is None else int(statuses[index].item()),
            )
        return model_gt_sample

    def save_preview(
        self,
        image: npt.NDArray[np.float32],
        depth_map: npt.NDArray[np.float32],
        base_name: str,
        calibration_status: int | None,
    ) -> None:
        """Write the depth and intensity overlays of one camera.

        Args:
            image: RGB image of shape ``(3, height, width)`` in unit range.
            depth_map: Depth and intensity channels of shape ``(2, height, width)``.
            base_name: Filename stem of the previews.
            calibration_status: ``CalibrationStatus`` value appended to the filename, ``None``
                for no suffix.
        """
        rgb, depth, intensity = self.recover_channels(image, depth_map)

        if calibration_status is None:
            status_suffix = ""
        elif calibration_status == CalibrationStatus.CALIBRATED.value:
            status_suffix = "_calibrated"
        else:
            status_suffix = "_miscalibrated"

        depth_overlay = self.create_overlay(rgb, depth, self.depth_cmap, self.alpha)
        intensity_overlay = self.create_overlay(rgb, intensity, self.intensity_cmap, self.alpha)

        depth_path = self.out_dir / f"{base_name}{status_suffix}_depth.png"
        intensity_path = self.out_dir / f"{base_name}{status_suffix}_intensity.png"
        # cv2.imwrite expects BGR.
        cv2.imwrite(str(depth_path), cv2.cvtColor(depth_overlay, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(intensity_path), cv2.cvtColor(intensity_overlay, cv2.COLOR_RGB2BGR))

    def recover_channels(
        self, image: npt.NDArray[np.float32], depth_map: npt.NDArray[np.float32]
    ) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.float32], npt.NDArray[np.float32]]:
        """Recover the RGB image, the depth in meters and the intensity.

        Args:
            image: RGB image of shape ``(3, height, width)`` in unit range.
            depth_map: Depth and intensity channels of shape ``(2, height, width)`` as
                written by ``LidarCameraFusion``.

        Returns:
            RGB image ``(height, width, 3)`` uint8, depth ``(height, width)`` in meters and
            intensity ``(height, width)`` in unit range.
        """
        rgb = np.clip(image[:3] * 255.0, 0, 255).astype(np.uint8).transpose(1, 2, 0)
        depth = (depth_map[0] * self.max_depth).astype(np.float32)
        intensity = depth_map[1].astype(np.float32)
        return np.ascontiguousarray(rgb), depth, intensity

    @staticmethod
    def create_overlay(
        rgb: npt.NDArray[np.uint8],
        values: npt.NDArray[np.float32],
        cmap: plt.Colormap,
        alpha: float,
    ) -> npt.NDArray[np.uint8]:
        """Blend colorized point values over an RGB image.

        Args:
            rgb: RGB image ``(height, width, 3)`` uint8.
            values: Values ``(height, width)`` to colorize, zero where no point landed.
            cmap: Matplotlib colormap.
            alpha: Blending factor of the points.

        Returns:
            Blended RGB image ``(height, width, 3)`` uint8.
        """
        mask = values > 0
        max_value = values.max() if values.max() > 0 else 1.0
        colored = (cmap(values / max_value)[:, :, :3] * 255).astype(np.uint8)

        result = rgb.copy()
        result[mask] = (
            (1 - alpha) * rgb[mask].astype(np.float32) + alpha * colored[mask].astype(np.float32)
        ).astype(np.uint8)
        return result
