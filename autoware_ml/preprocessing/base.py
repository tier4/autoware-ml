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

"""Base classes for GPU-oriented batch preprocessing pipelines.

This module defines the shared preprocessing interface used between dataloaders
and model forward passes.
"""

from collections.abc import Sequence
from typing import Any


class DataPreprocessing:
    """Apply a sequence of preprocessing layers to a collated batch.

    This runtime pipeline runs after batch transfer, enabling hardware-accelerated
    preprocessing operations like voxelization, projection, and format conversion
    without registering the pipeline as part of the neural network.

    The pipeline follows a dict-in/dict-out pattern where each layer receives the
    current batch dictionary and returns updates to merge into it.

    Args:
        pipeline: List of callable layers to apply sequentially.
            Each layer should accept ``dict[str, Any]`` and return ``dict[str, Any]``.

    Example:
        ```python
        preprocessing = DataPreprocessing(
            pipeline=[
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                RandomFlip3D(p=0.5),
            ]
        )
        batch = preprocessing(batch)  # Applied on GPU
        ```
    """

    def __init__(self, pipeline: Sequence[Any] = ()) -> None:
        """Initialize preprocessing with optional layers.

        Args:
            pipeline: List of callable layers to apply sequentially.
        """
        self.pipeline = list(pipeline)

    def __call__(self, batch_inputs_dict: dict[str, Any]) -> dict[str, Any]:
        """Apply preprocessing layers after the batch is already on device.

        The input dictionary is mutated in place; the same object is also
        returned for chaining convenience.

        Args:
            batch_inputs_dict: Collated batch dictionary on the target device.
                Mutated in place: each layer's returned mapping is merged into
                this dict.

        Returns:
            The same ``batch_inputs_dict`` with preprocessing applied.
        """
        for layer in self.pipeline:
            batch_inputs_dict |= layer(batch_inputs_dict)

        return batch_inputs_dict
