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

"""Re-estimate the BatchNorm statistics on training batches before each evaluation."""

from __future__ import annotations

import logging

from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities.exceptions import MisconfigurationException
import torch
from torch.nn.modules.batchnorm import _BatchNorm

logger = logging.getLogger(__name__)


class PreciseBatchNorm(Callback):
    """Recompute every BatchNorm's running statistics before validation and test.

    With a small momentum the running statistics trail the weights, which move fast around the
    peak learning rate, so the eval-mode model can drift far from the train-mode one while the
    training loss keeps falling. Before each validation and test the callback resets the
    running statistics of every BatchNorm and recomputes them as the plain average over
    ``num_batches`` training batches, with the rest of the model in eval mode. The checkpoint
    saved after the validation carries the recomputed statistics. Under a distributed
    strategy every rank averages its own batches.

    Args:
        num_batches: Training batches the statistics are averaged over.

    Raises:
        ValueError: If ``num_batches`` is not positive.
    """

    def __init__(self, num_batches: int = 100) -> None:
        if num_batches <= 0:
            raise ValueError(f"num_batches must be positive, got {num_batches}.")
        self.num_batches = int(num_batches)

    def on_validation_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Recompute the statistics before a validation epoch, not before the sanity check.

        Args:
            trainer: The trainer, whose datamodule serves the training batches.
            pl_module: The module whose BatchNorm layers are recalibrated.
        """
        if trainer.sanity_checking:
            return
        self.recalibrate(trainer, pl_module)

    def on_test_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Recompute the statistics before a test epoch.

        Args:
            trainer: The trainer, whose datamodule serves the training batches.
            pl_module: The module whose BatchNorm layers are recalibrated.
        """
        self.recalibrate(trainer, pl_module)

    def recalibrate(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Average the BatchNorm statistics over training batches.

        Args:
            trainer: The trainer, whose datamodule serves the training batches.
            pl_module: The module whose BatchNorm layers are recalibrated.

        Raises:
            MisconfigurationException: If the trainer has no datamodule to take the batches
                from.
        """
        batchnorms = [module for module in pl_module.modules() if isinstance(module, _BatchNorm)]
        if not batchnorms:
            return
        datamodule = trainer.datamodule
        if datamodule is None:
            raise MisconfigurationException(
                f"{type(self).__name__} needs a datamodule on the trainer to take the training "
                "batches from."
            )
        momenta = [module.momentum for module in batchnorms]
        was_training = pl_module.training
        pl_module.eval()
        for module in batchnorms:
            module.reset_running_stats()
            module.momentum = None  # cumulative average over the batches seen
            module.train()
        seen = 0
        with torch.no_grad():
            for batch in datamodule.train_dataloader():
                if seen >= self.num_batches:
                    break
                batch = pl_module.transfer_batch_to_device(batch, pl_module.device, 0)
                pl_module(pl_module.on_after_batch_transfer(batch, 0))
                seen += 1
        for module, momentum in zip(batchnorms, momenta):
            module.momentum = momentum
        pl_module.train(was_training)
        logger.info(
            "Recomputed the statistics of %d BatchNorm layers over %d training batches.",
            len(batchnorms),
            seen,
        )
