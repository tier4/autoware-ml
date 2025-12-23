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

"""Data preprocessing module for GPU-accelerated batch transformations."""

from typing import Any, Dict, List, Optional

import torch.nn as nn


class DataPreprocessing(nn.Module):
    """Apply a sequence of preprocessing layers to a collated batch.

    This module runs on the GPU after batch transfer, enabling hardware-accelerated
    preprocessing operations like normalization, augmentation, and format conversion.

    The pipeline follows a dict-in/dict-out pattern where each layer receives the
    current batch dictionary and returns updates to merge into it.

    Args:
        pipeline: List of nn.Module layers to apply sequentially.
            Each layer should accept Dict[str, Any] and return Dict[str, Any].

    Example:
        ```python
        preprocessing = DataPreprocessing(
            pipeline=[
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                RandomFlip(p=0.5),
            ]
        )
        batch = preprocessing(batch)  # Applied on GPU
        ```
    """

    def __init__(self, pipeline: Optional[List[nn.Module]] = None) -> None:
        """Initialize preprocessing with optional layers.

        Args:
            pipeline: List of nn.Module layers to apply sequentially.
        """
        super().__init__()
        self.pipeline = nn.ModuleList(pipeline or [])

    def __call__(self, batch_inputs_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Apply preprocessing layers after the batch is already on device.

        Args:
            batch_inputs_dict: Collated batch dictionary on the target device.

        Returns:
            Updated batch dictionary with preprocessing applied.
        """
        for layer in self.pipeline:
            batch_inputs_dict |= layer(batch_inputs_dict)

        return batch_inputs_dict
