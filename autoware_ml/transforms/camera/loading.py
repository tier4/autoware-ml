"""Camera loading transforms."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from autoware_ml.transforms.base import BaseTransform


class LoadImageFromFile(BaseTransform):
    """Load one RGB image from a metadata path."""

    _required_keys = ["img_path"]

    def __init__(self, *, to_float32: bool = False, color_type: str = "rgb") -> None:
        """Initialize the LoadImageFromFile transform.

        Args:
            to_float32: Whether to cast image pixels to ``float32``.
            color_type: Output color format, ``"rgb"`` or ``"bgr"``.
        """
        self.to_float32 = to_float32
        self.color_type = color_type

    def transform(self, input_dict: dict[str, Any]) -> dict[str, Any]:
        """Load an image from the configured path.

        Args:
            input_dict: Sample metadata containing ``img_path``.

        Returns:
            Updated sample dictionary with ``img``.
        """
        image = cv2.imread(input_dict["img_path"], cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Image not found: {input_dict['img_path']}")
        if self.color_type.lower() == "rgb":
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.to_float32:
            image = image.astype(np.float32)
        return {"img": image}


class LoadMultiViewImagesFromFiles(BaseTransform):
    """Load synchronized multiview images and camera matrices."""

    _required_keys = ["images", "camera_order"]

    def __init__(self, *, to_float32: bool = True, normalize_to_unit: bool = True) -> None:
        """Initialize the LoadMultiViewImagesFromFiles transform.

        Args:
            to_float32: Whether to cast images to ``float32``.
            normalize_to_unit: Whether to divide pixel values by ``255``.
        """
        self.to_float32 = to_float32
        self.normalize_to_unit = normalize_to_unit

    def transform(self, input_dict: dict[str, Any]) -> dict[str, Any]:
        """Load images and camera matrices for all configured views.

        Args:
            input_dict: Sample metadata containing multiview image info.

        Returns:
            Updated sample dictionary with image and calibration tensors.
        """
        images = []
        intrinsics = []
        lidar2cam = []
        lidar2img = []
        camera_names = []
        for camera_name in input_dict["camera_order"]:
            camera_info = input_dict["images"].get(camera_name)
            if camera_info is None:
                raise ValueError(
                    f"Camera '{camera_name}' declared in camera_order but missing from sample."
                )
            if camera_info.get("img_path") is None:
                raise ValueError(f"Camera '{camera_name}' has no img_path in sample.")

            image = cv2.imread(camera_info["img_path"], cv2.IMREAD_COLOR)
            if image is None:
                raise FileNotFoundError(f"Image not found: {camera_info['img_path']}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            if self.to_float32:
                image = image.astype(np.float32)
            if self.normalize_to_unit:
                image = image / 255.0
            images.append(np.transpose(image, (2, 0, 1)))

            camera_matrix = np.eye(4, dtype=np.float32)
            camera_matrix[:3, :3] = np.asarray(camera_info["cam2img"], dtype=np.float32)
            intrinsics.append(camera_matrix)

            camera_transform = np.asarray(camera_info["lidar2cam"], dtype=np.float32)
            lidar2cam.append(camera_transform)
            lidar2img.append(camera_matrix @ camera_transform)
            camera_names.append(camera_name)

        return {
            "img": np.stack(images, axis=0),
            "camera_intrinsics": np.stack(intrinsics, axis=0),
            "lidar2cam": np.stack(lidar2cam, axis=0),
            "lidar2img": np.stack(lidar2img, axis=0),
            "camera_names": camera_names,
        }
