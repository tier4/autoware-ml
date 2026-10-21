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

"""Gradient norm logging for the training loop."""

from __future__ import annotations

from enum import StrEnum

from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback
from torch.nn.utils import get_total_norm
from torch.optim import Optimizer


class GradNormType(StrEnum):
    """Order of the gradient norm, named as it appears in the logged metric key.

    Attributes:
        L1: Sum of the absolute gradient entries.
        L2: Euclidean norm, the one ``gradient_clip_algorithm: norm`` clips.
        INF: Largest absolute gradient entry.
    """

    L1 = "l1"
    L2 = "l2"
    INF = "inf"

    @property
    def order(self) -> float:
        """Norm order as ``torch.nn.utils.get_total_norm`` takes it.

        Returns:
            float: ``1.0``, ``2.0`` or ``inf``.
        """
        return {GradNormType.L1: 1.0, GradNormType.L2: 2.0, GradNormType.INF: float("inf")}[self]

    @property
    def log_key(self) -> str:
        """Metric key the norm is logged under, ``train/grad_norm_<order>``.

        Returns:
            str: Logged key.
        """
        return f"train/grad_norm_{self.value}"


class GradNormMonitor(Callback):
    """Log the total gradient norm of every optimizer step as ``train/grad_norm_<norm_type>``.

    The norm is taken over the gradients of every parameter the optimizer steps, as one
    vector, right before the optimizer step. Lightning calls the hook after the AMP scaler
    has unscaled the gradients and before ``trainer.gradient_clip_val`` clips them, so the
    logged value is the norm the clipping threshold is compared against: a run whose
    ``train/grad_norm_l2`` sits above ``gradient_clip_val`` most of the time is clipped on most
    steps. With gradient accumulation the norm is logged once per optimizer step, over the
    accumulated gradients. Non-finite gradients log ``inf`` or ``nan`` instead of raising.

    The norm is local to each rank. Under DDP the gradients are already averaged across ranks
    when the hook runs, so every rank sees the same norm. Sharded strategies hold only a
    shard of the gradients per rank and are not supported.

    The metric is a step metric: it is logged on every optimizer step, and the trainer's
    ``log_every_n_steps`` decides which steps reach the logger.
    """

    def __init__(self, norm_type: GradNormType | str = GradNormType.L2) -> None:
        """Initialize the callback.

        Args:
            norm_type: Order of the norm, a ``GradNormType`` or its string value (``"l1"``,
                ``"l2"``, ``"inf"``). The value names the logged key: ``"l2"`` logs
                ``train/grad_norm_l2``.
        """
        super().__init__()
        self.norm_type = GradNormType(norm_type)

    def on_before_optimizer_step(
        self, trainer: Trainer, pl_module: LightningModule, optimizer: Optimizer
    ) -> None:
        """Log the norm of the gradients the optimizer is about to step with."""
        grads = [
            param.grad
            for group in optimizer.param_groups
            for param in group["params"]
            if param.grad is not None
        ]
        if not grads:
            return
        total_norm = get_total_norm(grads, norm_type=self.norm_type.order, error_if_nonfinite=False)
        pl_module.log(
            self.norm_type.log_key,
            total_norm,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            batch_size=1,
        )
