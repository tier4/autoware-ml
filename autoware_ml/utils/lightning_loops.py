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

"""Lightning loop overrides shared by the training entrypoints."""

from __future__ import annotations

from typing import Any

import lightning as L
from lightning.pytorch.loops import _TrainingEpochLoop


class EpochEndValidationLoop(_TrainingEpochLoop):
    """Training epoch loop that also validates on the true last batch of an epoch.

    Lightning derives ``val_check_batch`` once from the dataloader length seen at
    setup time, and with ``val_check_interval=1.0`` only runs validation when
    ``(batch_idx + 1) % val_check_batch == 0``. Samplers whose epoch length
    varies (e.g. ``GroupStreamingSampler``, whose per-epoch scene permutation
    changes how many tail frames are trimmed) make shorter epochs never hit that
    modulo, so validation is *silently skipped* and stale ``callback_metrics``
    feed the checkpoint monitor. This subclass additionally triggers validation
    whenever the epoch's actual last batch is reached and the epoch is due for
    validation per ``check_val_every_n_epoch``. Trainers configured for
    step-based validation (``check_val_every_n_epoch=None``) are left
    untouched: for them ``_should_check_val_epoch`` is always true and the
    extra trigger would add an unrequested epoch-end validation.
    """

    def _should_check_val_fx(self, data_fetcher: Any) -> bool:
        if super()._should_check_val_fx(data_fetcher):
            return True
        if self.trainer.check_val_every_n_epoch is None:
            return False
        return bool(self.batch_progress.is_last_batch) and self._should_check_val_epoch()


def install_epoch_end_validation(trainer: L.Trainer) -> L.Trainer:
    """Replace the trainer's stock epoch loop with :class:`EpochEndValidationLoop`.

    Must run before ``trainer.fit``. Returns the trainer for chaining.
    """
    stock_loop = trainer.fit_loop.epoch_loop
    trainer.fit_loop.epoch_loop = EpochEndValidationLoop(
        trainer, stock_loop.min_steps, stock_loop.max_steps
    )
    return trainer
