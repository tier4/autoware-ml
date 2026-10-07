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

"""Abort a run that can no longer learn instead of finishing the epoch on NaN.

A mixed precision run that overflows keeps going: the gradient scaler skips every
step, the weights never change, BatchNorm running statistics turn NaN and the
epoch ends hours later with a NaN loss that ``EarlyStopping`` then reports as a
normal end of training. This callback watches every training batch on every
rank and raises as soon as the run is dead, so the job exits non-zero within
minutes and a dependent job does not start from the poisoned checkpoint.
"""

from __future__ import annotations

import logging
from typing import Any

import lightning as L
import torch

logger = logging.getLogger(__name__)


class NonFiniteTrainingError(RuntimeError):
    """Raised when training has produced non-finite losses for too long."""


class NonFiniteGuard(L.Callback):
    """Stop after consecutive non-finite losses or a collapsed gradient scaler.

    Args:
        max_consecutive: Number of consecutive training batches with a non-finite loss
            on any rank after which the run is aborted. A few isolated overflows are
            normal while the gradient scaler finds its scale.
        min_scale: Gradient scaler value below which the run is considered dead; the
            scaler halves its scale on every overflow and never recovers from zero.
    """

    def __init__(self, max_consecutive: int = 20, min_scale: float = 1.0) -> None:
        super().__init__()
        self.max_consecutive = int(max_consecutive)
        self.min_scale = float(min_scale)
        self.consecutive = 0

    @staticmethod
    def _loss_of(outputs: Any) -> torch.Tensor | None:
        loss = outputs.get("loss") if isinstance(outputs, dict) else outputs
        return loss if torch.is_tensor(loss) else None

    def on_train_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        loss = self._loss_of(outputs)
        if loss is None:
            return
        bad = torch.tensor(float(not bool(torch.isfinite(loss.detach()).all())), device=loss.device)
        # One rank overflowing poisons the all-reduced gradients of every rank, so the
        # decision has to be shared; otherwise the ranks would disagree on stopping.
        bad_any = float(trainer.strategy.reduce(bad, reduce_op="sum")) > 0
        self.consecutive = self.consecutive + 1 if bad_any else 0
        if self.consecutive >= self.max_consecutive:
            raise NonFiniteTrainingError(
                f"NonFiniteGuard: the training loss was non-finite on {self.consecutive} "
                f"consecutive batches (last batch {batch_idx}, epoch {trainer.current_epoch}); "
                "the run cannot recover. Typical causes are fp16 activation overflow or a "
                "learning rate the model cannot take."
            )
        scaler = getattr(getattr(trainer.strategy, "precision_plugin", None), "scaler", None)
        if scaler is not None and hasattr(scaler, "get_scale"):
            scale = float(scaler.get_scale())
            if scale < self.min_scale:
                raise NonFiniteTrainingError(
                    f"NonFiniteGuard: the gradient scaler collapsed to {scale:g} at batch "
                    f"{batch_idx} of epoch {trainer.current_epoch}; every optimizer step is being "
                    "skipped, so the weights no longer change. The forward pass overflows fp16."
                )
