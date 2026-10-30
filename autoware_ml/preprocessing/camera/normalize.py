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

"""Batched per-channel image normalization for the GPU preprocessing pipeline."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from autoware_ml.dataclasses.models.model_batch_inputs import ModelBatchInputs
from autoware_ml.preprocessing.data_preprocessor_modules import DataPreprocessorModule


class BatchNormalizeImages(DataPreprocessorModule):
    """Standardize every channel of the collated ``(B, N, C, H, W)`` images.

    The images and the statistics are in [0, 1], as ``LoadImagesFromFile(normalize_to_unit=True)``
    serves the images; the ImageNet statistics in that range are ``[0.485, 0.456, 0.406]`` and
    ``[0.229, 0.224, 0.225]``. The layer normalizes in training and in evaluation alike. Placed
    first in ``data_preprocessor.preprocessor_modules``, it shows the later modules what the
    runtime feeds a deployed graph that takes normalized images.
    """

    def __init__(self, mean: Sequence[float], std: Sequence[float]) -> None:
        """Initialize the normalization.

        Args:
            mean: Per-channel mean of the images in [0, 1].
            std: Per-channel standard deviation of the images in [0, 1].

        Raises:
            ValueError: If the statistics disagree on the channel count, or a mean lies outside
                [0, 1] or a standard deviation outside (0, 1], as the 0-255 statistics of the
                mmdetection configs do.
        """
        if len(mean) != len(std):
            raise ValueError(f"mean and std must describe the same channels, got {mean}, {std}.")
        if not all(0.0 <= value <= 1.0 for value in mean) or not all(
            0.0 < value <= 1.0 for value in std
        ):
            raise ValueError(
                "The statistics describe unit-range images: every mean must lie in [0, 1] and "
                f"every standard deviation in (0, 1], got {mean}, {std}."
            )
        self.mean = torch.tensor(mean, dtype=torch.float32).view(1, 1, -1, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).view(1, 1, -1, 1, 1)
        self._statistics: dict[
            tuple[torch.device, torch.dtype], tuple[torch.Tensor, torch.Tensor]
        ] = {}

    def _statistics_like(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the statistics on the device and in the dtype of the images, copied once."""
        key = (images.device, images.dtype)
        if key not in self._statistics:
            self._statistics[key] = (
                self.mean.to(device=images.device, dtype=images.dtype),
                self.std.to(device=images.device, dtype=images.dtype),
            )
        return self._statistics[key]

    def __call__(self, batch_inputs: ModelBatchInputs, *, is_training: bool) -> ModelBatchInputs:
        """Normalize the images of the batch.

        Args:
            batch_inputs: Model inputs holding the ``(B, N, C, H, W)`` images of the batch.
            is_training: Whether the owning model runs in training mode. Unused: the images are
                normalized in both modes.

        Returns:
            The model inputs with the normalized images.

        Raises:
            ValueError: If the model inputs carry no images, or images of another channel count
                than the statistics.
        """
        image_data = batch_inputs.image_data
        if image_data is None:
            raise ValueError("BatchNormalizeImages needs the images of the batch.")
        images = image_data.images
        if images.shape[2] != self.mean.numel():
            raise ValueError(
                f"The statistics describe {self.mean.numel()} channels, the images have "
                f"{images.shape[2]}."
            )
        mean, std = self._statistics_like(images)
        return batch_inputs.replace(image_data=image_data._replace(images=(images - mean) / std))
