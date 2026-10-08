# Copyright 2026 TIER IV, Inc.
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

"""Color helpers for visualization backends."""

from __future__ import annotations

import colorsys

import numpy as np
import numpy.typing as npt


def build_label_palette(num_classes: int, alpha: int = 255) -> npt.NDArray[np.uint8]:
    """Build a deterministic RGBA palette for semantic and detection labels."""
    if num_classes <= 0:
        return np.zeros((0, 4), dtype=np.uint8)

    palette = np.zeros((num_classes, 4), dtype=np.uint8)
    for index in range(num_classes):
        hue = (index * 0.61803398875) % 1.0
        red, green, blue = colorsys.hsv_to_rgb(hue, 0.75, 0.95)
        palette[index] = np.array(
            [int(red * 255), int(green * 255), int(blue * 255), alpha],
            dtype=np.uint8,
        )
    return palette


def labels_to_colors(
    labels: npt.NDArray[np.int64],
    palette: npt.NDArray[np.uint8],
    ignore_index: int | None = None,
    ignore_color: tuple[int, int, int, int] = (128, 128, 128, 255),
) -> npt.NDArray[np.uint8]:
    """Map integer labels to RGBA colors."""
    if labels.ndim != 1:
        raise ValueError(f"labels must be 1D, got shape {labels.shape}")

    colors = np.zeros((labels.shape[0], 4), dtype=np.uint8)
    valid_mask = labels >= 0
    if ignore_index is not None:
        ignore_mask = labels == ignore_index
        colors[ignore_mask] = np.array(ignore_color, dtype=np.uint8)
        valid_mask &= ~ignore_mask

    if valid_mask.any():
        max_label = int(labels[valid_mask].max())
        if max_label >= palette.shape[0]:
            raise ValueError(
                f"palette only covers labels up to {palette.shape[0] - 1}, got label {max_label}"
            )
        colors[valid_mask] = palette[labels[valid_mask]]

    invalid_mask = ~valid_mask
    colors[invalid_mask] = np.array(ignore_color, dtype=np.uint8)
    return colors


def scalar_to_heatmap_colors(
    values: npt.NDArray[np.float32], alpha: int = 255
) -> npt.NDArray[np.uint8]:
    """Map scalar values in [0, 1] to RGBA, blue (low) -> cyan -> green -> yellow -> red (high)."""
    if values.ndim != 1:
        raise ValueError(f"values must be 1D, got shape {values.shape}")
    if values.size == 0:
        return np.zeros((0, 4), dtype=np.uint8)
    v = np.clip(values, 0.0, 1.0).astype(np.float32)
    keypoints = np.array(
        [
            [0.0, 0.0, 255.0],
            [0.0, 255.0, 255.0],
            [0.0, 255.0, 0.0],
            [255.0, 255.0, 0.0],
            [255.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    positions = np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=np.float32)
    segment = np.searchsorted(positions, v, side="right").clip(1, len(positions) - 1) - 1
    t = ((v - positions[segment]) / (positions[segment + 1] - positions[segment] + 1e-8))[
        :, np.newaxis
    ]
    rgb = keypoints[segment] * (1.0 - t) + keypoints[segment + 1] * t
    colors = np.zeros((v.shape[0], 4), dtype=np.uint8)
    colors[:, :3] = rgb.clip(0, 255).astype(np.uint8)
    colors[:, 3] = alpha
    return colors


def depths_to_colors(depths: npt.NDArray[np.float32], alpha: int = 255) -> npt.NDArray[np.uint8]:
    """Map depth values to a deterministic near-to-far RGBA gradient."""
    if depths.ndim != 1:
        raise ValueError(f"depths must be 1D, got shape {depths.shape}")
    if depths.size == 0:
        return np.zeros((0, 4), dtype=np.uint8)

    min_depth = float(depths.min())
    max_depth = float(depths.max())
    if max_depth - min_depth < 1e-6:
        normalized = np.zeros_like(depths, dtype=np.float32)
    else:
        normalized = (depths - min_depth) / (max_depth - min_depth)

    colors = np.zeros((depths.shape[0], 4), dtype=np.uint8)
    for index, value in enumerate(normalized):
        hue = (2.0 / 3.0) * (1.0 - float(value))
        red, green, blue = colorsys.hsv_to_rgb(hue, 0.85, 0.95)
        colors[index] = np.array(
            [int(red * 255), int(green * 255), int(blue * 255), alpha],
            dtype=np.uint8,
        )
    return colors
