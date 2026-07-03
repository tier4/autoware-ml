"""Camera image masking transforms."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
import numpy.typing as npt

from autoware_ml.transforms.base import BaseTransform
from autoware_ml.transforms.camera.utils import as_hwc_image_list, restore_image_container


class GridMask(BaseTransform):
    """Apply grid masking augmentation to one image or a list of images."""

    _required_keys = ["img"]

    def __init__(self, *, p: float = 0.7, ratio: float = 0.5, rotate: int = 1) -> None:
        """Initialize the GridMask transform.

        Args:
            p: Probability of applying the transform.
            ratio: Fraction of each grid period that is masked out.
            rotate: Maximum absolute rotation in degrees applied to the mask.
        """
        self.p = p
        self.ratio = ratio
        self.rotate = rotate

    def transform(self, input_dict: dict[str, Any]) -> dict[str, Any]:
        """Mask images with a regular grid pattern."""
        images, format_info = as_hwc_image_list(input_dict["img"])
        masked = [self._grid_mask(image) for image in images]
        input_dict["img"] = restore_image_container(input_dict["img"], masked, format_info)
        return input_dict

    def _grid_mask(self, image: npt.NDArray) -> npt.NDArray:
        """Apply the grid mask to a single image."""
        height, width = image.shape[:2]
        period = np.random.randint(32, max(33, min(height, width)))
        cut = max(1, int(period * self.ratio))
        mask = np.ones((height, width), dtype=np.float32)
        offset_x = np.random.randint(period)
        offset_y = np.random.randint(period)
        for x in range(offset_x, width, period):
            mask[:, x : x + cut] = 0
        for y in range(offset_y, height, period):
            mask[y : y + cut, :] = 0
        if self.rotate > 0:
            angle = np.random.uniform(-self.rotate, self.rotate)
            rotation = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
            mask = cv2.warpAffine(mask, rotation, (width, height))
        if image.ndim == 3:
            mask = mask[..., None]
        return (image.astype(np.float32) * mask).astype(image.dtype)
