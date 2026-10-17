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

from abc import ABC, abstractmethod

from autoware_ml.dataclasses.models.model_batch_inputs import ModelBatchInputs


class DataPreprocessorModule(ABC):
    """Interface for data preprocessor module."""

    @abstractmethod
    def __call__(
        self,
        batch: ModelBatchInputs,
        *,
        is_training: bool,
    ) -> ModelBatchInputs:
        """
        Process batch data and convert to multi_task_batch_inputs for downstream tasks.

        Args:
            batch (ModelBatchInputs): The input features after processing.
            is_training (bool): Flag indicating whether the model is in training mode.

        Returns:
            ModelBatchInputs: The processed input features ready for downstream tasks.
        """
        raise NotImplementedError
