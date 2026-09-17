from __future__ import annotations

from typing import Sequence, NamedTuple

from jaxtyping import Float32
import torch
from torch import Tensor

from autoware_ml.geometry.cameras.base_images import BaseImages


class ImageGTBatch(NamedTuple):
    """Named tuple to represent pointcloud features in a batch size with their batch indices."""

    images: Float32[Tensor, "batch_size num_cameras num_channels height width"]
    depth_maps: Float32[Tensor, "batch_size num_cameras 1 height width"] | None
    camera_intrinsics: Float32[Tensor, "batch_size num_cameras 3 3"]
    image_augmentation_matrices: Float32[Tensor, "batch_size num_cameras 4 4"]
    lidar2images: Float32[Tensor, "batch_size num_cameras 4 4"]
    lidar2cams: Float32[Tensor, "batch_size num_cameras 4 4"]

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
          ValueError: If only a part of the samples carries depth images.
        """
        if len(images_gt_samples) == 0:
            return None

        # Concatenate all images from the sequence of BaseImages
        images = torch.cat([sample.images for sample in images_gt_samples], dim=0)
        lidar2images = torch.cat([sample.lidar2images for sample in images_gt_samples], dim=0)
        lidar2cams = torch.cat([sample.lidar2cams for sample in images_gt_samples], dim=0)
        camera_intrinsics = torch.cat(
            [sample.camera_intrinsics for sample in images_gt_samples], dim=0
        )
        image_augmentation_matrices = torch.cat(
            [sample.image_augmentation_matrices for sample in images_gt_samples], dim=0
        )

        # Depth images are optional, but either every sample of the batch carries them or
        # none does, otherwise the batch cannot be built.
        samples_with_depth = [
            sample.depth_maps for sample in images_gt_samples if sample.depth_maps is not None
        ]
        if len(samples_with_depth) == 0:
            depth_maps = None
        elif len(samples_with_depth) == len(images_gt_samples):
            depth_maps = torch.stack(samples_with_depth)
        else:
            raise ValueError(
                "All samples must either carry depth images or none of them, got "
                f"{len(samples_with_depth)} out of {len(images_gt_samples)} samples with depth."
            )

        return ImageGTBatch(
            images=images,
            depth_maps=depth_maps,
            camera_intrinsics=camera_intrinsics,
            image_augmentation_matrices=image_augmentation_matrices,
            lidar2cams=lidar2cams,
            lidar2images=lidar2images,
        )

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
