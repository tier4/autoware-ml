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

from autoware_ml.dataclasses.batch.sample_batch import ModelGTBatch
from autoware_ml.preprocessing.batch_adapter import ModelGTBatchAdapter


class DataPreprocessing:
    """Apply a sequence of preprocessing layers to a collated batch.

    This runtime pipeline runs after batch transfer, enabling hardware-accelerated
    preprocessing operations like voxelization, projection, and format conversion
    without registering the pipeline as part of the neural network.

    A ModelGTBatchAdapter first turns the typed batch into a dictionary of named tensors. Each
    layer then receives that dictionary and returns updates to merge into it.

    Args:
        pipeline: List of callable layers to apply sequentially. Each layer
            should accept ``(dict[str, Any], *, is_training: bool)`` and return
            ``dict[str, Any]``.

    Example:
        ```python
        preprocessing = DataPreprocessing(
            pipeline=[
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        batch_inputs_dict = preprocessing(batch, is_training=True)
        ```
    """

    def __init__(self, pipeline: Sequence[Any] = ()) -> None:
        """Initialize preprocessing with optional layers.

        Args:
            pipeline: List of callable layers to apply sequentially.
        """
        self.pipeline = list(pipeline)
        self.batch_adapter = ModelGTBatchAdapter()

    def __call__(self, batch: ModelGTBatch, *, is_training: bool) -> dict[str, Any]:
        """Apply preprocessing layers after the batch is already on device.

        Args:
            batch: Collated typed batch on the target device.
            is_training: Whether the model is in training mode. Passed to every layer, so
                a layer that behaves differently in training does not read module state.

        Returns:
            The model inputs of the batch with preprocessing applied.
        """
        batch_inputs_dict = self.batch_adapter(batch)
        for layer in self.pipeline:
            batch_inputs_dict |= layer(batch_inputs_dict, is_training=is_training)

        return batch_inputs_dict
