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

from typing import Any

import numpy as np
import torch
from PIL import Image

# TODO(vividf): Unify with autoware_ml.transforms.camera.masking.GridMask (CPU,
# per-image). Keep this verbatim port until the upstream StreamPETR baseline is
# reproduced, then either consolidate on one implementation or share the mask
# generation between the two.


class BatchGridMask:
    """Mask a rotated regular grid out of every image in a collated batch.

    This is the batched, on-device variant of GridMask ported verbatim from the
    upstream StreamPETR implementation. It is a
    :class:`~autoware_ml.preprocessing.base.DataPreprocessing` layer: it runs
    after batch transfer, reads ``img`` from the batch dictionary and returns the
    masked images under the same key and in the same container type (a list of
    ``(N, C, H, W)`` tensors or one ``(B, N, C, H, W)`` tensor).

    It intentionally differs from the CPU transform
    :class:`autoware_ml.transforms.camera.masking.GridMask`:

    * one random grid is sampled per call and shared by every image in the
      batch, instead of one grid per image;
    * the grid period is sampled from ``[2, H)`` instead of ``[32, min(H, W))``;
    * ``mode=1`` (default) zeroes the grid cells and keeps the bands, the
      complement of the CPU transform;
    * rotation uses integer degrees in ``[0, rotate)`` via PIL, so the default
      ``rotate=1`` never rotates;
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

    def __call__(self, batch_inputs_dict: dict[str, Any], *, is_training: bool) -> dict[str, Any]:
        """Apply one shared grid mask to the batch's ``img`` entry.

        Args:
            batch_inputs_dict: Collated batch on the target device. ``img`` is
                either a list of per-sample ``(N, C, H, W)`` tensors or one
                ``(B, N, C, H, W)`` tensor.
            is_training: Whether the owning model is in training mode. The mask
                is only applied during training.

        Returns:
            ``{"img": masked}`` in the same container type as the input, or an
            empty dict when the augmentation is skipped.
        """
        if not is_training or np.random.rand() > self.prob:
            return {}
        images = batch_inputs_dict["img"]
        if isinstance(images, (list, tuple)):
            stacked = torch.stack(list(images), dim=0)
        else:
            stacked = images
        batch_size, num_cams = stacked.shape[:2]
        masked = self.mask_images(stacked.flatten(0, 1)).reshape(
            batch_size, num_cams, *stacked.shape[2:]
        )
        if isinstance(images, (list, tuple)):
            return {"img": list(masked.unbind(0))}
        return {"img": masked}

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
