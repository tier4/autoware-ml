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

"""Unit tests for the gradient norm callback: the norm over every optimizer group, parameters
without gradients, the norm order and its logged key, non-finite gradients, the step-metric
logging options, and the strategies rejected at fit start."""

from __future__ import annotations

from collections.abc import Sequence
import math
import unittest
from unittest.mock import MagicMock

from lightning.pytorch.strategies import (
    DDPStrategy,
    DeepSpeedStrategy,
    FSDPStrategy,
    ModelParallelStrategy,
    SingleDeviceStrategy,
)
from lightning.pytorch.utilities.exceptions import MisconfigurationException
import torch
from torch.optim import SGD

from autoware_ml.callbacks.grad_norm_monitor import GradNormMonitor, GradNormType


class _GradNormMonitorTestCase(unittest.TestCase):
    """Shared helpers for the GradNormMonitor test cases."""

    @staticmethod
    def _module() -> MagicMock:
        """Build a stand-in LightningModule that records ``log()`` calls."""
        module = MagicMock()
        module.log = MagicMock()
        return module

    @staticmethod
    def _param(grad: Sequence[float] | None) -> torch.nn.Parameter:
        """Build a parameter whose gradient is ``grad``, or has no gradient when ``None``."""
        size = len(grad) if grad is not None else 2
        param = torch.nn.Parameter(torch.zeros(size))
        if grad is not None:
            param.grad = torch.tensor(grad)
        return param

    def _logged_value(self, module: MagicMock, key: str = "train/grad_norm_l2") -> torch.Tensor:
        """Return the single value the callback logged under ``key``."""
        calls = [call for call in module.log.call_args_list if call.args[0] == key]
        self.assertEqual(len(calls), 1, module.log.call_args_list)
        return calls[0].args[1]


class TestTotalNorm(_GradNormMonitorTestCase):
    """The logged norm spans the gradients of every parameter the optimizer steps."""

    def test_logs_the_euclidean_norm_over_every_param_group(self) -> None:
        """Two groups with gradients (3, 4) and (12,) give the norm 13."""
        optimizer = SGD(
            [{"params": [self._param([3.0, 4.0])]}, {"params": [self._param([12.0])]}], lr=0.1
        )
        module = self._module()

        GradNormMonitor().on_before_optimizer_step(MagicMock(), module, optimizer)

        self.assertAlmostEqual(self._logged_value(module).item(), 13.0, places=5)

    def test_skips_parameters_without_gradients(self) -> None:
        """A frozen parameter contributes nothing to the norm."""
        optimizer = SGD([self._param([3.0, 4.0]), self._param(None)], lr=0.1)
        module = self._module()

        GradNormMonitor().on_before_optimizer_step(MagicMock(), module, optimizer)

        self.assertAlmostEqual(self._logged_value(module).item(), 5.0, places=5)

    def test_logs_nothing_when_no_parameter_has_a_gradient(self) -> None:
        """A step without any gradient logs no metric instead of a misleading zero."""
        optimizer = SGD([self._param(None)], lr=0.1)
        module = self._module()

        GradNormMonitor().on_before_optimizer_step(MagicMock(), module, optimizer)

        module.log.assert_not_called()

    def test_infinity_norm_logs_the_largest_gradient_entry_under_its_own_key(self) -> None:
        """With ``norm_type="inf"`` the largest absolute entry is logged as grad_norm_inf."""
        optimizer = SGD([self._param([-7.0, 4.0]), self._param([2.0])], lr=0.1)
        module = self._module()

        callback = GradNormMonitor(norm_type="inf")
        callback.on_before_optimizer_step(MagicMock(), module, optimizer)

        self.assertAlmostEqual(
            self._logged_value(module, "train/grad_norm_inf").item(), 7.0, places=5
        )

    def test_l1_norm_sums_the_absolute_entries_under_its_own_key(self) -> None:
        """With ``norm_type=GradNormType.L1`` the sum of absolute entries is logged as grad_norm_l1."""
        optimizer = SGD([self._param([-7.0, 4.0]), self._param([2.0])], lr=0.1)
        module = self._module()

        callback = GradNormMonitor(norm_type=GradNormType.L1)
        callback.on_before_optimizer_step(MagicMock(), module, optimizer)

        self.assertAlmostEqual(
            self._logged_value(module, "train/grad_norm_l1").item(), 13.0, places=5
        )

    def test_unknown_norm_type_is_rejected(self) -> None:
        """A string outside the enum fails at construction, not on the first optimizer step."""
        with self.assertRaises(ValueError):
            GradNormMonitor(norm_type="l3")

    def test_non_finite_gradients_log_without_raising(self) -> None:
        """An overflowing gradient is reported as a non-finite norm, not an exception."""
        optimizer = SGD([self._param([1.0, float("inf")])], lr=0.1)
        module = self._module()

        GradNormMonitor().on_before_optimizer_step(MagicMock(), module, optimizer)

        self.assertFalse(math.isfinite(self._logged_value(module).item()))


class TestLoggingOptions(_GradNormMonitorTestCase):
    """The norm is a per-step training metric."""

    def test_logs_a_tensor_on_step_only_with_unit_batch_size(self) -> None:
        """The value stays a tensor and is logged per step, not per epoch, weighted by one."""
        optimizer = SGD([self._param([3.0, 4.0])], lr=0.1)
        module = self._module()

        GradNormMonitor().on_before_optimizer_step(MagicMock(), module, optimizer)

        self.assertIsInstance(self._logged_value(module), torch.Tensor)
        kwargs = module.log.call_args.kwargs
        self.assertTrue(kwargs["on_step"])
        self.assertFalse(kwargs["on_epoch"])
        self.assertFalse(kwargs["prog_bar"])
        self.assertEqual(kwargs["batch_size"], 1)


class TestFitStart(_GradNormMonitorTestCase):
    """The callback refuses strategies whose ranks hold only a shard of the gradients."""

    @staticmethod
    def _trainer(strategy_cls: type) -> MagicMock:
        """Build a stand-in Trainer whose strategy is an instance of ``strategy_cls``."""
        trainer = MagicMock()
        trainer.strategy = MagicMock(spec=strategy_cls)
        return trainer

    def test_rejects_the_sharded_strategies(self) -> None:
        """FSDP, DeepSpeed and the model-parallel strategy fail fit start by name."""
        for strategy_cls in (FSDPStrategy, DeepSpeedStrategy, ModelParallelStrategy):
            with self.subTest(strategy=strategy_cls.__name__):
                with self.assertRaisesRegex(MisconfigurationException, strategy_cls.__name__):
                    GradNormMonitor().on_fit_start(self._trainer(strategy_cls), self._module())

    def test_accepts_the_replicated_strategies(self) -> None:
        """A single device and DDP hold the whole gradients on every rank, so fit proceeds."""
        for strategy_cls in (SingleDeviceStrategy, DDPStrategy):
            with self.subTest(strategy=strategy_cls.__name__):
                GradNormMonitor().on_fit_start(self._trainer(strategy_cls), self._module())


if __name__ == "__main__":
    unittest.main()
