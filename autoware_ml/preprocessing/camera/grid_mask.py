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

"""Batched grid-mask image augmentation for the GPU preprocessing pipeline."""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image

from autoware_ml.dataclasses.models.model_batch_inputs import ModelBatchInputs


class BatchGridMask:
    """Mask a rotated regular grid out of every image in a collated batch.

    The mask follows the StreamPETR implementation. The layer runs after batch transfer,
    reads the images of the model inputs and replaces them with the masked images.

    * One random grid is sampled per call and shared by every image in the batch.
    * The grid period is sampled from ``[2, H)``.
    * ``mode=1`` (default) zeroes the grid cells and keeps the bands.
    * Rotation uses integer degrees in ``[0, rotate)`` via PIL, so the default
      ``rotate=1`` never rotates.
    * ``offset`` can fill masked pixels with uniform noise instead of zeros.

    The augmentation is training-only and is a no-op when ``is_training`` is
    false or the probability gate does not fire.
    """

    def __init__(
        self,
        use_h: bool = True,
        use_w: bool = True,
        rotate: int = 1,
        offset: bool = False,
        ratio: float = 0.5,
        mode: int = 1,
        prob: float = 0.7,
    ) -> None:
        """Initialize the batched grid-mask augmentation.

        Args:
            use_h: Whether to mask horizontal grid bands.
            use_w: Whether to mask vertical grid bands.
            rotate: Upper bound (degrees) of the random grid rotation; ``0`` disables it.
            offset: Whether to add random noise inside masked cells.
            ratio: Band-width ratio of one grid period.
            mode: ``0`` masks the grid cells, ``1`` masks their complement.
            prob: Probability of applying the mask per call.
        """
        self.use_h = use_h
        self.use_w = use_w
        self.rotate = rotate
        self.offset = offset
        self.ratio = ratio
        self.mode = mode
        self.prob = prob

    def __call__(self, batch_inputs: ModelBatchInputs, *, is_training: bool) -> ModelBatchInputs:
        """Apply one shared grid mask to the images of the batch.

        Args:
            batch_inputs: Model inputs holding the ``(B, N, C, H, W)`` images of the batch.
            is_training: Whether the owning model is in training mode. The mask
                is only applied during training.

        Returns:
            The model inputs with the masked images, unchanged when the augmentation is skipped.

        Raises:
            ValueError: If the model inputs carry no images.
        """
        if not is_training or np.random.rand() > self.prob:
            return batch_inputs
        image_data = batch_inputs.image_data
        if image_data is None:
            raise ValueError("BatchGridMask needs the images of the batch.")
        images = image_data.images
        masked = self.mask_images(images.flatten(0, 1)).reshape(images.shape)
        return batch_inputs.replace(image_data=image_data._replace(images=masked))

    def mask_images(self, x: torch.Tensor) -> torch.Tensor:
        """Apply one shared grid mask to a ``(N, C, H, W)`` image batch.

        This is the upstream StreamPETR algorithm, unchanged. Unlike
        :meth:`__call__` it does not check the probability gate or training mode.
        """
        n, c, h, w = x.size()
        x = x.reshape(-1, h, w)
        padded_h = int(1.5 * h)
        padded_w = int(1.5 * w)
        d = np.random.randint(2, h)
        band = min(max(int(d * self.ratio + 0.5), 1), d - 1)
        mask = np.ones((padded_h, padded_w), np.float32)
        st_h = np.random.randint(d)
        st_w = np.random.randint(d)
        if self.use_h:
            for i in range(padded_h // d):
                s = d * i + st_h
                t = min(s + band, padded_h)
                mask[s:t, :] *= 0
        if self.use_w:
            for i in range(padded_w // d):
                s = d * i + st_w
                t = min(s + band, padded_w)
                mask[:, s:t] *= 0

        r = np.random.randint(self.rotate) if self.rotate > 0 else 0
        mask_image = Image.fromarray(np.uint8(mask)).rotate(r)
        mask = np.asarray(mask_image)
        mask = mask[
            (padded_h - h) // 2 : (padded_h - h) // 2 + h,
            (padded_w - w) // 2 : (padded_w - w) // 2 + w,
        ]

        mask_tensor = torch.from_numpy(mask.copy()).to(device=x.device, dtype=x.dtype)
        if self.mode == 1:
            mask_tensor = 1 - mask_tensor
        mask_tensor = mask_tensor.expand_as(x)
        if self.offset:
            offset = torch.from_numpy(2 * (np.random.rand(h, w) - 0.5)).to(
                device=x.device, dtype=x.dtype
            )
            x = x * mask_tensor + offset * (1 - mask_tensor)
        else:
            x = x * mask_tensor

        return x.reshape(n, c, h, w)
