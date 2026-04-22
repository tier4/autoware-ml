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

"""Unit tests for learning rate schedulers."""

import pytest
import torch

from autoware_ml.utils.schedulers.cosine_annealing import CosineAnnealingLR
from autoware_ml.utils.schedulers.cyclic_cosine_annealing import CyclicCosineAnnealingLR
from autoware_ml.utils.schedulers.cyclic_momentum import CyclicMomentumScheduler
from autoware_ml.utils.schedulers.linear_warmup_cosine_annealing import (
    LinearWarmupCosineAnnealingLR,
)

# Suppress PyTorch warning about calling scheduler.step() before optimizer.step()
# This is expected in unit tests where we simulate epoch progression without actual training
pytestmark = pytest.mark.filterwarnings(
    "ignore:Detected call of `lr_scheduler.step\\(\\)` before `optimizer.step\\(\\)`"
)


class TestCosineAnnealingLR:
    """Tests for CosineAnnealingLR scheduler."""

    def test_instantiation(self, optimizer: torch.optim.Optimizer) -> None:
        """Test that scheduler can be instantiated."""
        scheduler = CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-6)
        assert scheduler is not None
        assert scheduler.T_max == 10
        assert scheduler.eta_min == 1e-6

    def test_lr_decreases_during_decay(
        self, optimizer: torch.optim.Optimizer, base_lr: float
    ) -> None:
        """Test that LR decreases when eta_min < base_lr (decay mode)."""
        eta_min = 1e-6
        scheduler = CosineAnnealingLR(optimizer, T_max=10, eta_min=eta_min)

        initial_lr = optimizer.param_groups[0]["lr"]

        # Step through half the epochs
        for _ in range(5):
            scheduler.step()

        mid_lr = optimizer.param_groups[0]["lr"]

        # LR should have decreased
        assert mid_lr < initial_lr, "LR should decrease during decay"
        assert mid_lr > eta_min, "LR should not reach eta_min yet"

    def test_lr_increases_during_warmup(
        self, optimizer: torch.optim.Optimizer, base_lr: float
    ) -> None:
        """Test that LR increases when eta_min > base_lr (warmup mode)."""
        eta_min = 1e-2  # Higher than base_lr (1e-3)
        scheduler = CosineAnnealingLR(optimizer, T_max=10, eta_min=eta_min)

        initial_lr = optimizer.param_groups[0]["lr"]

        # Step through half the epochs
        for _ in range(5):
            scheduler.step()

        mid_lr = optimizer.param_groups[0]["lr"]

        # LR should have increased
        assert mid_lr > initial_lr, "LR should increase during warmup"


class TestCyclicCosineAnnealingLR:
    """Tests for CyclicCosineAnnealingLR scheduler."""

    def test_instantiation(self, optimizer: torch.optim.Optimizer) -> None:
        """Test that scheduler can be instantiated."""
        scheduler = CyclicCosineAnnealingLR(
            optimizer,
            warmup_epochs=8,
            decay_epochs=12,
            max_lr_factor=10.0,
            min_lr_factor=1e-4,
        )
        assert scheduler is not None
        assert scheduler.warmup_epochs == 8
        assert scheduler.decay_epochs == 12
        assert scheduler.total_epochs == 20

    def test_warmup_phase_increases_lr(
        self, optimizer: torch.optim.Optimizer, base_lr: float
    ) -> None:
        """Test that LR increases during warmup phase."""
        scheduler = CyclicCosineAnnealingLR(
            optimizer,
            warmup_epochs=8,
            decay_epochs=12,
            max_lr_factor=10.0,
            min_lr_factor=1e-4,
        )

        initial_lr = optimizer.param_groups[0]["lr"]

        # Step through warmup phase
        for _ in range(8):
            scheduler.step()

        peak_lr = optimizer.param_groups[0]["lr"]

        # LR should have increased to approximately base_lr * max_lr_factor
        assert peak_lr > initial_lr, "LR should increase during warmup"
        expected_peak = base_lr * 10.0
        assert abs(peak_lr - expected_peak) < expected_peak * 0.1, (
            f"Peak LR {peak_lr} should be close to {expected_peak}"
        )

    def test_decay_phase_decreases_lr(
        self, optimizer: torch.optim.Optimizer, base_lr: float
    ) -> None:
        """Test that LR decreases during decay phase."""
        scheduler = CyclicCosineAnnealingLR(
            optimizer,
            warmup_epochs=8,
            decay_epochs=12,
            max_lr_factor=10.0,
            min_lr_factor=1e-4,
        )

        # Go through warmup
        for _ in range(8):
            scheduler.step()

        peak_lr = optimizer.param_groups[0]["lr"]

        # Go through decay
        for _ in range(12):
            scheduler.step()

        final_lr = optimizer.param_groups[0]["lr"]

        # LR should have decreased
        assert final_lr < peak_lr, "LR should decrease during decay"
        expected_min = base_lr * 1e-4
        assert abs(final_lr - expected_min) < expected_min * 0.1, (
            f"Final LR {final_lr} should be close to {expected_min}"
        )


class TestCyclicMomentumScheduler:
    """Tests for CyclicMomentumScheduler."""

    def test_instantiation(self, adamw_optimizer: torch.optim.Optimizer) -> None:
        """Test that scheduler can be instantiated."""
        scheduler = CyclicMomentumScheduler(
            adamw_optimizer,
            warmup_epochs=8,
            decay_epochs=12,
            min_momentum_factor=0.85 / 0.95,
            max_momentum_factor=1.0,
        )
        assert scheduler is not None
        assert scheduler.warmup_epochs == 8
        assert scheduler.decay_epochs == 12

    def test_warmup_phase_decreases_momentum(self, adamw_optimizer: torch.optim.Optimizer) -> None:
        """Test that momentum decreases during warmup phase."""
        scheduler = CyclicMomentumScheduler(
            adamw_optimizer,
            warmup_epochs=8,
            decay_epochs=12,
            min_momentum_factor=0.85 / 0.95,
            max_momentum_factor=1.0,
        )

        initial_momentum = adamw_optimizer.param_groups[0]["betas"][0]

        # Step through warmup phase
        for _ in range(8):
            scheduler.step()

        min_momentum = adamw_optimizer.param_groups[0]["betas"][0]

        # Momentum should have decreased
        assert min_momentum < initial_momentum, "Momentum should decrease during warmup"

    def test_decay_phase_increases_momentum(self, adamw_optimizer: torch.optim.Optimizer) -> None:
        """Test that momentum increases during decay phase."""
        scheduler = CyclicMomentumScheduler(
            adamw_optimizer,
            warmup_epochs=8,
            decay_epochs=12,
            min_momentum_factor=0.85 / 0.95,
            max_momentum_factor=1.0,
        )

        # Go through warmup
        for _ in range(8):
            scheduler.step()

        min_momentum = adamw_optimizer.param_groups[0]["betas"][0]

        # Go through decay
        for _ in range(12):
            scheduler.step()

        final_momentum = adamw_optimizer.param_groups[0]["betas"][0]

        # Momentum should have increased
        assert final_momentum > min_momentum, "Momentum should increase during decay"


class TestLinearWarmupCosineAnnealingLR:
    """Tests for LinearWarmupCosineAnnealingLR scheduler."""

    def test_instantiation(self, optimizer: torch.optim.Optimizer) -> None:
        """Test that scheduler can be instantiated."""
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer,
            warmup_epochs=5,
            max_epochs=20,
            warmup_start_lr=1e-6,
            eta_min=1e-8,
        )
        assert scheduler is not None
        assert scheduler.warmup_epochs == 5
        assert scheduler.max_epochs == 20
        assert scheduler.decay_epochs == 15

    def test_linear_warmup(self, optimizer: torch.optim.Optimizer, base_lr: float) -> None:
        """Test that warmup is linear."""
        warmup_start_lr = 1e-6
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer,
            warmup_epochs=5,
            max_epochs=20,
            warmup_start_lr=warmup_start_lr,
            eta_min=1e-8,
        )

        # Collect LR values during warmup
        lr_values = []
        for _ in range(5):
            lr_values.append(optimizer.param_groups[0]["lr"])
            scheduler.step()

        # Check linear increase
        for i in range(1, len(lr_values)):
            diff = lr_values[i] - lr_values[i - 1]
            expected_diff = (base_lr - warmup_start_lr) / 5
            assert abs(diff - expected_diff) < expected_diff * 0.1, (
                "Warmup should be approximately linear"
            )

    def test_starts_at_warmup_start_lr(self, optimizer: torch.optim.Optimizer) -> None:
        """Test that LR starts at warmup_start_lr."""
        warmup_start_lr = 1e-6
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer,
            warmup_epochs=5,
            max_epochs=20,
            warmup_start_lr=warmup_start_lr,
            eta_min=1e-8,
        )

        # Get the LR that scheduler returns for epoch 0
        initial_lr = scheduler.get_lr()[0]
        assert abs(initial_lr - warmup_start_lr) < warmup_start_lr * 0.1, (
            f"Initial LR {initial_lr} should be close to warmup_start_lr {warmup_start_lr}"
        )

    def test_reaches_base_lr_after_warmup(
        self, optimizer: torch.optim.Optimizer, base_lr: float
    ) -> None:
        """Test that LR reaches base_lr at end of warmup."""
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer,
            warmup_epochs=5,
            max_epochs=20,
            warmup_start_lr=1e-6,
            eta_min=1e-8,
        )

        # Step through warmup
        for _ in range(5):
            scheduler.step()

        peak_lr = optimizer.param_groups[0]["lr"]
        assert abs(peak_lr - base_lr) < base_lr * 0.01, (
            f"Peak LR {peak_lr} should be close to base_lr {base_lr}"
        )

    def test_cosine_decay_after_warmup(
        self, optimizer: torch.optim.Optimizer, base_lr: float
    ) -> None:
        """Test that cosine decay happens after warmup."""
        eta_min = 1e-8
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer,
            warmup_epochs=5,
            max_epochs=20,
            warmup_start_lr=1e-6,
            eta_min=eta_min,
        )

        # Complete warmup
        for _ in range(5):
            scheduler.step()

        peak_lr = optimizer.param_groups[0]["lr"]

        # Complete decay
        for _ in range(15):
            scheduler.step()

        final_lr = optimizer.param_groups[0]["lr"]

        assert final_lr < peak_lr, "LR should decrease during decay"
        assert abs(final_lr - eta_min) < eta_min * 0.1, (
            f"Final LR {final_lr} should be close to eta_min {eta_min}"
        )
