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
    ``normalize_to_unit=True``, and so do the statistics, for example the ImageNet mean
    ``[0.485, 0.456, 0.406]`` and standard deviation ``[0.229, 0.224, 0.225]`` of RGB images.
    Statistics or pixels in [0, 255] would be standardized without an error into values far from
    the ones the models read, so the layer refuses both: statistics outside the unit range when
    it is built, and a pixel above 1.5 (room for the overshoot of an interpolation) in its first
    training batch and its first evaluation batch, one host synchronization each. The training
    and evaluation pipelines load their images apart, and a fit run starts with validation.

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
            ValueError: If the statistics disagree on the channel count, or a mean lies outside
                [0, 1] or a standard deviation outside (0, 1].
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
        self.mean = torch.tensor(mean, dtype=torch.float32)
        self.std = torch.tensor(std, dtype=torch.float32)
        # The is_training values whose first batch passed the range check.
        self._checked_modes: set[bool] = set()

    def __call__(self, batch_inputs: ModelBatchInputs, *, is_training: bool) -> ModelBatchInputs:
        """Normalize the images of the batch.

        Args:
            batch_inputs: Model inputs holding the ``(B, N, C, H, W)`` images of the batch.
            is_training: Whether the owning model runs in training mode. The images are
                normalized in both modes; the first batch of each is range-checked.

        Returns:
            The model inputs with the normalized images.

        Raises:
            ValueError: If the model inputs carry no images, images of another channel count
                than the statistics, or, in the first batch of a mode, images that are not
                unit-range.
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
        if is_training not in self._checked_modes:
            # One reduction and one host synchronization per mode, see the class docstring.
            max_pixel = float(images.amax())
            if max_pixel > _UNIT_RANGE_LIMIT:
                mode = "training" if is_training else "evaluation"
                raise ValueError(
                    "BatchNormalizeImages takes unit-range images and statistics, but the first "
                    f"{mode} batch holds a pixel of {max_pixel:g}. Load the images with "
                    "LoadImagesFromFile(normalize_to_unit=True)."
                )
            self._checked_modes.add(is_training)
        shape = (1, 1, -1, 1, 1)
        # A blocking copy to the device would wait for the work queued on it, every step.
        mean = self.mean.to(device=images.device, dtype=images.dtype, non_blocking=True).view(shape)
        std = self.std.to(device=images.device, dtype=images.dtype, non_blocking=True).view(shape)
        return batch_inputs.replace(image_data=image_data._replace(images=(images - mean) / std))
