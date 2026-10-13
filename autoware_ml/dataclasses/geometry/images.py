from __future__ import annotations

from typing import Sequence, NamedTuple

from jaxtyping import Float32, Int64
import torch
from torch import Tensor

from autoware_ml.geometry.cameras.base_images import BaseImages


def _stack_present_in_all(tensors: Sequence[Tensor | None], name: str) -> Tensor | None:
    """
    Stack an optional per sample tensor that every sample of a batch carries, or none does.

    Args:
      tensors: The tensor of every sample, None when the sample carries none.
      name: Name of the tensor for the error message.

    Returns:
      Tensor | None: The stacked tensors, None when no sample carries one.

    Raises:
      ValueError: If only some of the samples carry the tensor.
    """
    present = [tensor for tensor in tensors if tensor is not None]
    if not present:
        return None
    if len(present) != len(tensors):
        raise ValueError(
            f"All samples must either carry {name} or none of them, got {len(present)} out of "
            f"{len(tensors)} samples with {name}."
        )
    return torch.stack(present)


class ImageGTBatch(NamedTuple):
    """Named tuple to represent pointcloud features in a batch size with their batch indices."""

    images: Float32[Tensor, "batch_size num_cameras num_channels height width"]
    depth_maps: Float32[Tensor, "batch_size num_cameras num_depth_channels height width"] | None
    # Intrinsics after the image space augmentation, the ones lidar2images is built from
    camera_intrinsics: Float32[Tensor, "batch_size num_cameras 3 3"]
    image_augmentation_matrices: Float32[Tensor, "batch_size num_cameras 4 4"]
    lidar2images: Float32[Tensor, "batch_size num_cameras 4 4"]
    lidar2cams: Float32[Tensor, "batch_size num_cameras 4 4"]
    # Calibration status of every camera, None until the misalignment augmentation has run
    calibration_statuses: Int64[Tensor, "batch_size num_cameras"] | None

    @staticmethod
    def collate_gt_samples(
        images_gt_samples: Sequence[BaseImages],
    ) -> ImageGTBatch | None:
        """
        Collate a sequence of images (BaseImages) into a single ImageGTBatch.

        Args:
          images_gt_samples: Sequence of images (BaseImages) to be collated.

        Returns:
          ImageGTBatch: Collated images GT batch, or None if the sequence is empty.

        Raises:
          ValueError: If only some of the samples carry depth images or calibration statuses.
        """
        if len(images_gt_samples) == 0:
            return None

        # Stack the per-camera tensors of every sample along a new leading batch dimension,
        # so every field is laid out as (batch_size, num_cameras, ...).
        images = torch.stack([sample.images for sample in images_gt_samples])
        lidar2images = torch.stack([sample.lidar2images for sample in images_gt_samples])
        lidar2cams = torch.stack([sample.lidar2cams for sample in images_gt_samples])
        camera_intrinsics = torch.stack(
            [sample.augmented_camera_intrinsics for sample in images_gt_samples]
        )
        image_augmentation_matrices = torch.stack(
            [sample.image_augmentation_matrices for sample in images_gt_samples]
        )

        calibration_statuses = _stack_present_in_all(
            [sample.calibration_statuses for sample in images_gt_samples], "calibration statuses"
        )
        depth_maps = _stack_present_in_all(
            [sample.depth_maps for sample in images_gt_samples], "depth images"
        )

        return ImageGTBatch(
            images=images,
            depth_maps=depth_maps,
            camera_intrinsics=camera_intrinsics,
            image_augmentation_matrices=image_augmentation_matrices,
            lidar2cams=lidar2cams,
            lidar2images=lidar2images,
            calibration_statuses=calibration_statuses,
        )

    def fused_images(
        self,
    ) -> Float32[Tensor, "batch_size*num_cameras num_fused_channels height width"]:
        """
        Stack the depth maps onto the images, one fused image per camera of every sample.

        Returns:
          Float32[Tensor, "batch_size*num_cameras num_fused_channels height width"]: The images
            with their depth channels appended, the sample and the camera dimensions merged.

        Raises:
          ValueError: If the batch carries no depth maps.
        """
        if self.depth_maps is None:
            raise ValueError("Fused images need the depth maps of the images.")
        return torch.cat([self.images, self.depth_maps], dim=2).flatten(0, 1)

    def to_device(self, device: torch.device) -> ImageGTBatch:
        """
        Move the ImageGtBatch to the specified device.

        Args:
          device: The target device to move the batch to.

        Returns:
          ImageGtBatch: The batch moved to the specified device.
        """
        return ImageGTBatch(
            images=self.images.to(device),
            depth_maps=(self.depth_maps.to(device) if self.depth_maps is not None else None),
            camera_intrinsics=self.camera_intrinsics.to(device),
            image_augmentation_matrices=self.image_augmentation_matrices.to(device),
            lidar2cams=self.lidar2cams.to(device),
            lidar2images=self.lidar2images.to(device),
            calibration_statuses=(
                self.calibration_statuses.to(device)
                if self.calibration_statuses is not None
                else None
            ),
        )


class ImageSample(NamedTuple):
    """
    Named tuple to represent a single row of image data, which contains the dataset record for the
    image task.
    """

    image_path: str
    camera_name: str
    timestamp: float
    # Transformation matrix for camera_intrinsics
    camera_intrinsic: Float32[Tensor, "3 3"]
    # Transformation matrix for lidar to camera
    lidar2cam: Float32[Tensor, "4 4"]
    # Transformation matrix for lidar to image
    lidar2image: Float32[Tensor, "4 4"]
    distortion_model: str
    # Distortion coefficients following the OpenCV convention ``(k1, k2, p1, p2[, k3[, ...]])``.
    # The length varies by distortion model (4, 5, 8, 12 or 14), empty for undistorted images.
    distortion_coefficients: Float32[Tensor, " num_coefficients"]
