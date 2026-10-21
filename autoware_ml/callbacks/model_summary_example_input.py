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

"""Example input for the Lightning model summary, taken from the training split."""

from __future__ import annotations

from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities.exceptions import MisconfigurationException

from autoware_ml.dataclasses.batch.sample_batch import ModelGTBatch
from autoware_ml.datamodule.data_module import DataModule
from autoware_ml.types.dataset import SplitType


class ModelSummaryExampleInput(Callback):
    """Expose the first training sample as the module's example input at fit time.

    Lightning's model summary reports the FLOPs of a forward pass only when the module carries
    an ``example_input_array``. This callback collates the first sample of the training split
    into a single-sample batch and sets it on the module in ``setup``, which Lightning runs
    after the datamodule's ``setup`` and before the summary's ``on_fit_start``. The summary
    then moves the batch to the device and runs it through ``on_after_batch_transfer`` like a
    training batch, so the reported FLOPs cover the runtime preprocessing and the model for
    one sample.

    The summary runs that forward pass on every rank at fit start. Remove the callback from the
    configuration to skip it. A module that already carries an example input keeps it.

    Raises:
        MisconfigurationException: In ``setup`` for the fit stage when the trainer has no
            :class:`DataModule`, the datamodule serves no training split, or the training split
            is empty.
    """

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        """Set the example input of the module for the fit stage.

        Args:
            trainer: The trainer, whose datamodule provides the training split.
            pl_module: The module the summary will describe.
            stage: Lightning stage, one of ``fit``, ``validate``, ``test`` or ``predict``.
        """
        if stage != "fit" or pl_module.example_input_array is not None:
            return
        pl_module.example_input_array = self.build_example_input_batch(trainer)

    @staticmethod
    def build_example_input_batch(trainer: Trainer) -> ModelGTBatch:
        """Collate the first training sample into a single-sample batch.

        Args:
            trainer: The trainer whose datamodule serves the training split.

        Returns:
            A batch holding the first training sample.

        Raises:
            MisconfigurationException: See the class docstring.
        """
        datamodule = trainer.datamodule
        if not isinstance(datamodule, DataModule):
            raise MisconfigurationException(
                f"{ModelSummaryExampleInput.__name__} needs a {DataModule.__name__} on the "
                f"trainer to take the example input from, got {type(datamodule).__name__}."
            )
        dataset = datamodule.datasets.get(SplitType.TRAIN)
        if dataset is None:
            raise MisconfigurationException(
                f"{ModelSummaryExampleInput.__name__} needs a training split to take the "
                "example input from, but the datamodule serves none."
            )
        if len(dataset) == 0:
            raise MisconfigurationException(
                f"{ModelSummaryExampleInput.__name__} needs a training sample to use as the "
                "example input, but the training split is empty."
            )
        return dataset.collate_fn([dataset[0]])
