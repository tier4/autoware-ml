# Copyright 2025 TIER IV, Inc.
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

from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from autoware_ml.transforms.base import BaseTransform


class PermuteAxes(BaseTransform):
    """Permute axes of arrays/tensors.

    Reorders dimensions according to the specified axes order.
    For example, axes=(2, 0, 1) converts (H, W, C) to (C, H, W).

    Required keys:
        - All keys specified in input_keys parameter. Each must be a numpy array
          or torch tensor with number of dimensions matching len(axes).

    Optional keys:
        - None

    Generated keys:
        - All keys in input_keys are modified in-place with permuted dimensions.

    Args:
        input_keys: List of keys to permute from the input dictionary.
        axes: Tuple specifying the new order of axes.
    """

    def __init__(self, input_keys: List[str], axes: Tuple[int, ...]):
        super().__init__()
        self.input_keys = input_keys
        self.axes = axes
        # Dynamic required keys based on input_keys parameter
        self._required_keys = list(input_keys)

    def transform(self, input_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Permute axes of specified keys.

        Args:
            input_dict: Dictionary with arrays/tensors.

        Returns:
            Dictionary with permuted arrays/tensors.
        """
        for key in self.input_keys:
            data = input_dict[key]
            if len(data.shape) != len(self.axes):
                raise ValueError(
                    f"PermuteAxes: Number of dimensions {data.shape} does not match "
                    f"number of axes {self.axes} for key '{key}'"
                )
            if isinstance(data, torch.Tensor):
                input_dict[key] = data.permute(*self.axes)
            elif isinstance(data, np.ndarray):
                input_dict[key] = np.transpose(data, self.axes)

        return input_dict
