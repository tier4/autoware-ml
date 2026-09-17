"""
It is a base class for all image data structures, providing common attributes and methods that can be used by derived classes.
"""

from __future__ import annotations

from typing import Sequence

from jaxtyping import Float32, Int64
from pydantic import BaseModel, ConfigDict
import torch


class BaseImages(BaseModel):
    """
    Immutable container for the camera images of a sample and their calibration.

    Note that it only supports tensor data type for now. Every field is frozen, so a
    transform never mutates an instance in place: it builds the updated container with
    ``model_copy(update=...)`` and hands it back to the caller.

    Attributes:
        images: Images in Tensor to represent images for a sample.
        depth_maps: Tensor to represent depth value (distance to cameras) of pixels
        for each image. None when the sample carries no depth.
        timestamps: Tensor represents the timestamps for each images.
        camera_intrinsics: Tensor represents camera intrinsics for each camera.
        camera_names: Sequence for camera names to represent each image.
        lidar2images: Transformation matrix for lidar to each image.
        lidar2cams: Transformation matrix for lidar to each camera.
        distortion_models: Distortion model name per camera following ROS convention
        (e.g. ``"plumb_bob"``, ``"rational_polynomial"``). Empty string for
        pre-undistorted images.
        distortion_coefficients: Distortion coefficients per camera following the OpenCV
        convention ``(k1, k2, p1, p2[, k3[, ...]])``. The length varies by distortion
        model (4, 5, 8, 12 or 14), hence a sequence of 1D tensors rather than a single
        stacked tensor. An empty tensor for pre-undistorted images.
        augmented_camera_intrinsics: Updated intrinsic matrix after image-space transforms.
        The loading transforms initialize it with a copy of ``camera_intrinsics``.
        image_augmentation_matrices: Composition of the 2D affines applied to the pixels
        by the image-space augmentations, mapping a pixel of the raw loaded image onto
        its location in ``images``. It is stored as a 4x4 homogeneous matrix: the 2D
        rotation and scale sit in the top-left 2x2 block, the pixel translation in the
        first two rows of the last column, and the remaining diagonal is one. Slicing
        ``[:3, :3]`` therefore yields the rotation padded with a unit depth axis and
        ``[:3, 3]`` the translation with a zero depth offset, which is the layout the
        view transforms consume. The loading transforms initialize it with the identity.
        Note that a non-affine image transform, such as the undistortion applied by
        ``UndistortImage``, is not composed into it, so it describes the augmentations
        alone rather than the full raw-image-to-augmented-image mapping.
        noises: Per-camera 4x4 homogeneous perturbation applied to ``lidar2cams`` by the
        calibration-misalignment augmentation, ``None`` when no perturbation was applied.
        calibration_statuses: Per-camera ``CalibrationStatus`` value written by the
        calibration-misalignment augmentation, ``None`` until it has run.
    """

    # ``revalidate_instances`` makes ``model_validate`` re-run the field validation on an
    # instance instead of handing it back untouched, which is what lets the transforms
    # validate the containers they build with ``model_copy``, as that call skips it.
    model_config = ConfigDict(
        frozen=True,
        strict=True,
        arbitrary_types_allowed=True,
        revalidate_instances="always",
    )

    images: Float32[torch.Tensor, "num_cameras num_channels height width"]
    depth_maps: Float32[torch.Tensor, "num_cameras 1 height width"] | None = None
    timestamps: Float32[torch.Tensor, " num_cameras"]
    camera_intrinsics: Float32[torch.Tensor, "num_cameras 3 3"]
    camera_names: Sequence[str]
    lidar2images: Float32[torch.Tensor, "num_cameras 4 4"]
    lidar2cams: Float32[torch.Tensor, "num_cameras 4 4"]
    distortion_models: Sequence[str]
    distortion_coefficients: Sequence[Float32[torch.Tensor, " num_coefficients"]]
    augmented_camera_intrinsics: Float32[torch.Tensor, "num_cameras 3 3"]
    image_augmentation_matrices: Float32[torch.Tensor, "num_cameras 4 4"]
    noises: Float32[torch.Tensor, "num_cameras 4 4"] | None = None
    calibration_statuses: Int64[torch.Tensor, " num_cameras"] | None = None

    @staticmethod
    def identity_image_augmentation_matrices(
        camera_intrinsics: Float32[torch.Tensor, "num_cameras 3 3"],
    ) -> Float32[torch.Tensor, "num_cameras 4 4"]:
        """
        Build the per-camera identity the image augmentation matrices start from.

        Args:
            camera_intrinsics: Intrinsics whose leading dimension, dtype, and device the
                identity matrices are built for.

        Returns:
            The 4x4 identity repeated once per camera.
        """
        return torch.eye(4, dtype=camera_intrinsics.dtype, device=camera_intrinsics.device).repeat(
            camera_intrinsics.shape[0], 1, 1
        )

    @staticmethod
    def homogeneous_image_augmentation_matrices(
        pixel_affines: Float32[torch.Tensor, "num_cameras 3 3"],
    ) -> Float32[torch.Tensor, "num_cameras 4 4"]:
        """
        Lift per-camera 2D pixel affines into the 4x4 layout of ``image_augmentation_matrices``.

        Args:
            pixel_affines: 3x3 affines ``[[R, t], [0, 1]]`` acting on homogeneous pixels.

        Returns:
            4x4 matrices holding ``R`` in the top-left 2x2 block, ``t`` in the first two rows
            of the last column, and one on the remaining diagonal.
        """
        homogeneous = torch.eye(4, dtype=pixel_affines.dtype, device=pixel_affines.device).repeat(
            pixel_affines.shape[0], 1, 1
        )
        homogeneous[:, :2, :2] = pixel_affines[:, :2, :2]
        homogeneous[:, :2, 3] = pixel_affines[:, :2, 2]
        return homogeneous

    def image_augmentation_pixel_affines(self) -> Float32[torch.Tensor, "num_cameras 3 3"]:
        """
        Read ``image_augmentation_matrices`` back as 3x3 affines acting on homogeneous pixels.

        Returns:
            Per-camera 3x3 affines ``[[R, t], [0, 1]]`` such that ``affine @ (u, v, 1)`` is the
            location in ``images`` of the raw pixel ``(u, v)``.
        """
        pixel_affines = torch.eye(
            3,
            dtype=self.image_augmentation_matrices.dtype,
            device=self.image_augmentation_matrices.device,
        ).repeat(self.image_augmentation_matrices.shape[0], 1, 1)
        pixel_affines[:, :2, :2] = self.image_augmentation_matrices[:, :2, :2]
        pixel_affines[:, :2, 2] = self.image_augmentation_matrices[:, :2, 3]
        return pixel_affines

    def update_lidar_transformation_matrices(
        self, augmentation_inverse: Float32[torch.Tensor, "4 4"]
    ) -> BaseImages:
        """
        Keep lidar-camera projection consistent after a lidar-space transform.

        Applies ``augmentation_inverse`` (inverse of the 4x4 point-space augmentation) to
        ``lidar2cams`` and recomputes ``lidar2images`` from ``augmented_camera_intrinsics``.

        Args:
            augmentation_inverse: Inverse of the 4x4 point-space augmentation.

        Returns:
            A new BaseImages holding the updated ``lidar2cams`` and ``lidar2images``. The
            instance it is built from is left untouched.

        Raises:
            ValidationError: If the updated matrices do not match the field annotations.
        """
        # (num_cameras, 4, 4) @ (4, 4)
        augmented_lidar2cam = self.lidar2cams @ augmentation_inverse

        # (num_cameras, 4, 4), the intrinsics padded to homogeneous coordinates.
        camera_intrinsics = torch.eye(
            4,
            device=self.augmented_camera_intrinsics.device,
            dtype=self.augmented_camera_intrinsics.dtype,
        ).repeat(self.augmented_camera_intrinsics.shape[0], 1, 1)
        camera_intrinsics[:, :3, :3] = self.augmented_camera_intrinsics

        # model_copy does not validate what it is given, so the copy is validated explicitly.
        return BaseImages.model_validate(
            self.model_copy(
                update={
                    "lidar2cams": augmented_lidar2cam,
                    # augmented_intrinsic @ augmented_lidar2cam -> lidar2img
                    "lidar2images": camera_intrinsics @ augmented_lidar2cam,
                }
            )
        )
