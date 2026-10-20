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

"""Runtime preprocessing pipeline that turns a collated batch into model inputs."""

from __future__ import annotations

from collections.abc import Sequence

from autoware_ml.dataclasses.batch.sample_batch import ModelGTBatch
from autoware_ml.dataclasses.models.model_batch_inputs import ModelBatchInputs
from autoware_ml.preprocessing.data_preprocessor_modules import DataPreprocessorModule


class DataPreprocessor:
    """Apply the runtime preprocessing modules to a batch before it reaches the model.

    The preprocessor is not a registered submodule of the model, so the mode of the owning
    model reaches the modules only through the explicit ``is_training`` argument.

    Args:
        preprocessor_modules: Modules applied in order to the batch inputs. Every module must
            implement :class:`DataPreprocessorModule`.

    Raises:
        TypeError: If a module does not implement :class:`DataPreprocessorModule`.
    """

    def __init__(self, preprocessor_modules: Sequence[DataPreprocessorModule]) -> None:
        for index, module in enumerate(preprocessor_modules):
            if not isinstance(module, DataPreprocessorModule):
                raise TypeError(
                    f"preprocessor_modules[{index}] must be a DataPreprocessorModule, got "
                    f"{type(module).__name__}."
                )
        self.preprocessor_modules: tuple[DataPreprocessorModule, ...] = tuple(preprocessor_modules)

    def __call__(self, batch: ModelGTBatch, *, is_training: bool) -> ModelBatchInputs:
        """Run every module on the batch, each reading the inputs the previous one returned.

        Args:
            batch: Collated batch on the target device.
            is_training: Whether the owning model runs in training mode.

        Returns:
            ModelBatchInputs: The model inputs after every preprocessor module.
        """
        batch_inputs = ModelBatchInputs.from_gt_batch(batch)
        for module in self.preprocessor_modules:
            batch_inputs = module(batch_inputs, is_training=is_training)
        return batch_inputs

    def __repr__(self) -> str:
        """Name the preprocessor by the modules it runs, in order."""
        modules = ", ".join(type(module).__name__ for module in self.preprocessor_modules)
        return f"{type(self).__name__}(preprocessor_modules=[{modules}])"
