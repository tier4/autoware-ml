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

from typing import Any, Dict

import cv2
import numpy as np

from autoware_ml.transforms.base import BaseTransform


class PhotometricDistortion(BaseTransform):
    """Apply random brightness, contrast, saturation, and hue to RGB channels.

    Operates on img (H, W, 3). Assumes uint8 [0, 255] input in BGR format.

    Required keys:
        - img: (H, W, 3) uint8 BGR image.

    Optional keys:
        - None

    Generated keys:
        - img: Modified in-place with photometric distortions (when applied).

    Args:
        p: Probability of applying augmentation.
        brightness: Max brightness deviation [0, 1].
        contrast: Max contrast deviation [0, 1].
        saturation: Max saturation deviation [0, 1].
        hue: Max hue deviation [0, 0.5].
    """

    _required_keys = ["img"]

    def __init__(
        self,
        p: float = 0.5,
        brightness: float = 0.0,
        contrast: float = 0.0,
        saturation: float = 0.0,
        hue: float = 0.0,
    ):
        super().__init__()
        self.p = p
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def transform(self, input_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Apply photometric distortion to RGB channels of image."""
        # img is (H, W, 3) uint8
        img = input_dict["img"]

        # Convert to HSV
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)

        # Apply distortions
        if self.brightness > 0:
            hsv[..., 2] *= np.random.uniform(1 - self.brightness, 1 + self.brightness)

        if self.saturation > 0:
            hsv[..., 1] *= np.random.uniform(1 - self.saturation, 1 + self.saturation)

        if self.contrast > 0:
            # Simple contrast: scale V around 127.5
            factor = np.random.uniform(1 - self.contrast, 1 + self.contrast)
            hsv[..., 2] = (hsv[..., 2] - 127.5) * factor + 127.5

        if self.hue > 0:
            hsv[..., 0] += np.random.uniform(-self.hue, self.hue) * 179.0
            hsv[..., 0] = np.mod(hsv[..., 0], 180.0)

        # Clip and convert back
        hsv = np.clip(hsv, 0, 255).astype(np.uint8)
        img_aug = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        input_dict["img"] = img_aug
        return input_dict
