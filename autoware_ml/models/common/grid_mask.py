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

"""Grid-mask image augmentation applied on batched image tensors."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from PIL import Image


class GridMask(nn.Module):
    """Mask a rotated regular grid out of every image in the batch.

    One random grid pattern is sampled per call and applied to all images,
    zeroing (``mode=0``) or keeping (``mode=1``) the grid cells. The module is
    a training-time augmentation and passes inputs through unchanged in eval
    mode or when the probability gate does not fire.
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
        """Initialize the grid-mask augmentation.

        Args:
            use_h: Whether to mask horizontal grid bands.
            use_w: Whether to mask vertical grid bands.
            rotate: Upper bound (degrees) of the random grid rotation.
            offset: Whether to add random noise inside masked cells.
            ratio: Band-width ratio of one grid period.
            mode: ``0`` masks the grid cells, ``1`` masks their complement.
            prob: Probability of applying the mask per call.
        """
        super().__init__()
        self.use_h = use_h
        self.use_w = use_w
        self.rotate = rotate
        self.offset = offset
        self.ratio = ratio
        self.mode = mode
        self.prob = prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply one shared grid mask to a ``(N, C, H, W)`` image batch."""
        if np.random.rand() > self.prob or not self.training:
            return x
        n, c, h, w = x.size()
        x = x.view(-1, h, w)
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

        r = np.random.randint(self.rotate)
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

        return x.view(n, c, h, w)
