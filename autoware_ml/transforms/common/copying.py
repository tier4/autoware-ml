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

"""Copy transforms shared across tasks."""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import torch

from autoware_ml.transforms.base import BaseTransform


class Copy(BaseTransform):
    """Copy selected fields to new keys."""

    def __init__(self, keys_dict: dict[str, str]) -> None:
        """Initialize the copy transform.

        Args:
            keys_dict: Mapping from source key to destination key.
        """
        self.keys_dict = keys_dict
        self._required_keys = list(self.keys_dict.keys())

    def transform(self, input_dict: dict[str, Any]) -> dict[str, Any]:
        """Copy configured fields into new entries."""
        output: dict[str, Any] = {}
        for source_key, target_key in self.keys_dict.items():
            value = input_dict[source_key]
            if isinstance(value, np.ndarray):
                output[target_key] = value.copy()
            elif isinstance(value, torch.Tensor):
                output[target_key] = value.clone().detach()
            else:
                output[target_key] = copy.deepcopy(value)
        return output
