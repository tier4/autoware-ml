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


# Largest pixel value accepted in unit-range images. Unit-range images exceed 1 only by the
# overshoot of an image interpolation, while [0, 255] images exceed it by far.
_UNIT_RANGE_LIMIT = 1.5


class BatchNormalizeImages(DataPreprocessorModule):
    """Standardize every channel of every image of a collated batch.

    The images arrive in [0, 1], as ``LoadImagesFromFile`` serves them with
    ``normalize_to_unit=True``, so the statistics are given in that range as well, for example
    the ImageNet mean ``[0.485, 0.456, 0.406]`` and standard deviation ``[0.229, 0.224, 0.225]``
    of RGB images. The models, the camera-lidar fusion and these statistics read unit-range
    pixels, and unit-range statistics applied to ``[0, 255]`` pixels would fail silently, so
    the layer refuses images that are not unit-range: a pixel of the first batch it sees above
    1.5, which leaves room for the overshoot of an interpolation, raises an error naming
    ``normalize_to_unit``. Only the first batch is checked, which costs one reduction and one
    host synchronization per run rather than one per step.

    A model whose deployed graph takes normalized images, normalized by the runtime outside the
    graph, normalizes here, before any image augmentation of the preprocessing pipeline, so the
    layers after it see what the runtime feeds the graph. The normalization runs in training and
    in evaluation alike.
    """

    def __init__(self, mean: Sequence[float], std: Sequence[float]) -> None:
        """Initialize the normalization.

        Args:
            mean: Per-channel mean of the images in [0, 1].
            std: Per-channel standard deviation of the images in [0, 1].

        Raises:
            ValueError: If the statistics disagree on the channel count or a standard
                deviation is not positive.
        """
        if len(mean) != len(std):
            raise ValueError(f"mean and std must describe the same channels, got {mean}, {std}.")
        if any(value <= 0.0 for value in std):
            raise ValueError(f"Every standard deviation must be positive, got {std}.")
        self.mean = torch.tensor(mean, dtype=torch.float32)
        self.std = torch.tensor(std, dtype=torch.float32)
        self._unit_range_checked = False

    def __call__(self, batch_inputs: ModelBatchInputs, *, is_training: bool) -> ModelBatchInputs:
        """Normalize the images of the batch.

        Args:
            batch_inputs: Model inputs holding the ``(B, N, C, H, W)`` images of the batch.
            is_training: Unused, the images are normalized in every phase.

        Returns:
            The model inputs with the normalized images.

        Raises:
            ValueError: If the model inputs carry no images, images of another channel count
                than the statistics, or, in the first batch, images that are not unit-range.
        """
        del is_training
        image_data = batch_inputs.image_data
        if image_data is None:
            raise ValueError("BatchNormalizeImages needs the images of the batch.")
        images = image_data.images
        if images.shape[2] != self.mean.numel():
            raise ValueError(
                f"The statistics describe {self.mean.numel()} channels, the images have "
                f"{images.shape[2]}."
            )
        if not self._unit_range_checked:
            # One reduction and one host synchronization per run (see the class docstring).
            max_pixel = float(images.amax())
            if max_pixel > _UNIT_RANGE_LIMIT:
                raise ValueError(
                    "BatchNormalizeImages takes unit-range images and statistics, but the first "
                    f"batch holds a pixel of {max_pixel:g}. Load the images with "
                    "LoadImagesFromFile(normalize_to_unit=True)."
                )
            self._unit_range_checked = True
        shape = (1, 1, -1, 1, 1)
        mean = self.mean.to(device=images.device, dtype=images.dtype).view(shape)
        std = self.std.to(device=images.device, dtype=images.dtype).view(shape)
        return batch_inputs.replace(image_data=image_data._replace(images=(images - mean) / std))
