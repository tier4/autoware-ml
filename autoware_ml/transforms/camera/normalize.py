"""Camera image normalization transforms."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from autoware_ml.transforms.base import BaseTransform
from autoware_ml.transforms.camera.utils import as_hwc_image_list, restore_image_container


class NormalizeMultiviewImage(BaseTransform):
    """Normalize multiview images channel-wise."""

    _required_keys = ["img"]

    def __init__(self, *, mean: Sequence[float], std: Sequence[float], to_rgb: bool = True) -> None:
        """Initialize the NormalizeMultiviewImage transform.

        Args:
            mean: Per-channel mean subtracted from each image.
            std: Per-channel standard deviation used for scaling.
            to_rgb: Whether to reverse 3-channel images before normalization.
        """
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)
        self.to_rgb = to_rgb

    def transform(self, input_dict: dict[str, Any]) -> dict[str, Any]:
        """Normalize one or more images."""
        images, format_info = as_hwc_image_list(input_dict["img"])
        normalized = []
        for image in images:
            image = image.astype(np.float32)
            if self.to_rgb and image.shape[-1] == 3:
                image = image[..., ::-1]
            normalized.append((image - self.mean) / self.std)
        input_dict["img"] = restore_image_container(input_dict["img"], normalized, format_info)
        input_dict["img_norm_cfg"] = {"mean": self.mean, "std": self.std, "to_rgb": self.to_rgb}
        return input_dict
