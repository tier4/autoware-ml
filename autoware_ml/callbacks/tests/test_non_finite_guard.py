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

"""Unit tests for the non-finite training guard."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch

from autoware_ml.callbacks.non_finite_guard import NonFiniteGuard, NonFiniteTrainingError


class _Scaler:
    """Gradient scaler stand-in with a fixed scale."""

    def __init__(self, scale: float) -> None:
        self.scale = scale

    def get_scale(self) -> float:
        return self.scale


def _trainer(scale: float | None = 65536.0) -> SimpleNamespace:
    """Single-process trainer stand-in, with a gradient scaler unless scale is None."""
    strategy = SimpleNamespace(
        reduce=lambda tensor, reduce_op="sum": tensor,
        precision_plugin=SimpleNamespace(scaler=None if scale is None else _Scaler(scale)),
    )
    return SimpleNamespace(strategy=strategy, current_epoch=0)


class TestNonFiniteGuard(unittest.TestCase):
    """Unit tests for NonFiniteGuard."""

    def test_isolated_non_finite_losses_are_tolerated(self) -> None:
        """Test that a finite loss resets the count of non-finite ones."""
        guard = NonFiniteGuard(max_consecutive=3)
        trainer = _trainer()
        losses = [1.0, float("nan"), 0.9, float("inf"), float("nan"), 0.8]

        for batch_idx, value in enumerate(losses):
            guard.on_train_batch_end(trainer, None, {"loss": torch.tensor(value)}, None, batch_idx)

        self.assertEqual(guard.consecutive, 0)

    def test_consecutive_non_finite_losses_abort(self) -> None:
        """Test that max_consecutive non-finite losses in a row abort the run."""
        guard = NonFiniteGuard(max_consecutive=3)
        trainer = _trainer()
        guard.on_train_batch_end(trainer, None, torch.tensor(float("nan")), None, 0)
        guard.on_train_batch_end(trainer, None, torch.tensor(float("nan")), None, 1)

        with self.assertRaisesRegex(NonFiniteTrainingError, "3 consecutive"):
            guard.on_train_batch_end(trainer, None, torch.tensor(float("nan")), None, 2)

    def test_collapsed_scaler_aborts_even_with_finite_loss(self) -> None:
        """Test that a collapsed scaler aborts and a run without scaler passes."""
        guard = NonFiniteGuard()
        loss = {"loss": torch.tensor(1.0)}

        with self.assertRaisesRegex(NonFiniteTrainingError, "scaler collapsed"):
            guard.on_train_batch_end(_trainer(scale=0.0), None, loss, None, 5)
        # bf16 or fp32 runs have no scaler and are only guarded by the loss.
        guard.on_train_batch_end(_trainer(scale=None), None, loss, None, 6)

    def test_outputs_without_a_loss_are_ignored(self) -> None:
        """Test that steps returning no loss tensor are not counted."""
        guard = NonFiniteGuard(max_consecutive=1)

        for batch_idx, outputs in enumerate((None, {"other": 1})):
            with self.subTest(outputs=outputs):
                guard.on_train_batch_end(_trainer(), None, outputs, None, batch_idx)
                self.assertEqual(guard.consecutive, 0)
