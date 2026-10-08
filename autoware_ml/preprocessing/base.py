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
from autoware_ml.dataclasses.models.model_batch_inputs import ModelBatchInputs


class DataPreprocessing:
    """Apply a sequence of preprocessing layers to a collated batch.

    This runtime pipeline runs after batch transfer, enabling hardware-accelerated
    preprocessing operations like voxelization, projection, and format conversion
    without registering the pipeline as part of the neural network.

    The collated batch starts the model inputs. Each layer receives the model inputs and
    returns them with the features it computes added or the data it changes replaced.

    Args:
        pipeline: List of callable layers to apply sequentially. Each layer
            should accept ``(ModelBatchInputs, *, is_training: bool)`` and return
            ``ModelBatchInputs``.

    Example:
        ```python
        preprocessing = DataPreprocessing(
            pipeline=[
                PointPillarPreprocessor(...),
            ]
        )
        batch_inputs = preprocessing(batch, is_training=True)
        ```
    """

    def __init__(self, pipeline: Sequence[Any] = ()) -> None:
        """Initialize preprocessing with optional layers.

        Args:
            pipeline: List of callable layers to apply sequentially.
        """
        self.pipeline = list(pipeline)

    def __call__(self, batch: ModelGTBatch, *, is_training: bool) -> ModelBatchInputs:
        """Apply preprocessing layers after the batch is already on device.

        Args:
            batch: Collated typed batch on the target device.
            is_training: Whether the model is in training mode. Passed to every layer, so
                a layer that behaves differently in training does not read module state.

        Returns:
            The model inputs of the batch with preprocessing applied.
        """
        batch_inputs = ModelBatchInputs.from_gt_batch(batch)
        for layer in self.pipeline:
            batch_inputs = layer(batch_inputs, is_training=is_training)
        return batch_inputs
