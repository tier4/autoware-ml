"""Private helpers shared by camera transform modules."""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt


def is_chw_image(image: npt.NDArray) -> bool:
    """Return whether an image uses channel-first layout."""
    return image.ndim == 3 and image.shape[0] in (1, 3, 4) and image.shape[-1] not in (1, 3, 4)


def as_hwc_image_list(images: Any) -> tuple[list[npt.NDArray], dict[str, Any]]:
    """Normalize image containers to a list of HWC images."""
    if isinstance(images, list):
        format_info = {
            "container": "list",
            "layout": "chw" if images and is_chw_image(images[0]) else "hwc",
        }
        image_list = images
    elif isinstance(images, np.ndarray) and images.ndim == 4:
        format_info = {
            "container": "stack",
            "layout": "chw" if images.shape[1] in (1, 3, 4) else "hwc",
        }
        image_list = [images[index] for index in range(images.shape[0])]
    else:
        format_info = {"container": "single", "layout": "chw" if is_chw_image(images) else "hwc"}
        image_list = [images]

    hwc_images = [
        np.transpose(image, (1, 2, 0)) if format_info["layout"] == "chw" else image
        for image in image_list
    ]
    return hwc_images, format_info


def restore_image_container(
    template: Any, images: list[npt.NDArray], format_info: dict[str, Any]
) -> Any:
    """Restore a list of HWC images to the original container type."""
    restored = [
        np.transpose(image, (2, 0, 1)) if format_info["layout"] == "chw" else image
        for image in images
    ]
    if format_info["container"] == "list":
        return restored
    if format_info["container"] == "stack":
        return np.stack(restored, axis=0)
    del template
    return restored[0]
