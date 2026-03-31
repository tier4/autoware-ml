"""Camera loading transforms."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from autoware_ml.transforms.base import BaseTransform


class LoadImageFromFile(BaseTransform):
    """Load one RGB image from a metadata path."""

    _required_keys = ["img_path"]

    def __init__(self, to_float32: bool = False, color_type: str = "rgb") -> None:
        """Initialize the image loader.

        Args:
            to_float32: Whether to cast image pixels to ``float32``.
            color_type: Output color format, ``rgb`` or ``bgr``.
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
