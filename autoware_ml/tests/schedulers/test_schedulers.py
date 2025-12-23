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

"""
Unit tests for learning rate schedulers with visualization.

The tests themselves should be simple. The generated plots shall be used to visualize the schedulers.
"""

from pathlib import Path

import pytest
import torch
import torch.nn as nn

from autoware_ml.utils.schedulers import (
    CosineAnnealingLR,
    CyclicCosineAnnealingLR,
    CyclicMomentumScheduler,
    LinearWarmupCosineAnnealingLR,
)
from tests.schedulers.conftest import (
    collect_lr_values,
    collect_momentum_values,
    save_lr_plot,
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

    def test_plot_decay_curve(
        self, optimizer: torch.optim.Optimizer, plot_dir: Path, base_lr: float
    ) -> None:
        """Generate and save LR decay curve plot."""
        T_max = 20
        eta_min = 1e-8
        scheduler = CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)

        epochs, lr_values = collect_lr_values(scheduler, T_max, optimizer)

        save_lr_plot(
            epochs,
            lr_values,
            f"CosineAnnealingLR (Decay)\nT_max={T_max}, base_lr={base_lr:.0e}, eta_min={eta_min:.0e}",
            plot_dir / "cosine_annealing_decay.png",
        )

        # Verify plot was created
        assert (plot_dir / "cosine_annealing_decay.png").exists()

    def test_plot_warmup_curve(
        self, optimizer: torch.optim.Optimizer, plot_dir: Path, base_lr: float
    ) -> None:
        """Generate and save LR warmup curve plot."""
        T_max = 10
        eta_min = 1e-2  # Target higher than base_lr
        scheduler = CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)

        epochs, lr_values = collect_lr_values(scheduler, T_max, optimizer)

        save_lr_plot(
            epochs,
            lr_values,
            f"CosineAnnealingLR (Warmup)\nT_max={T_max}, base_lr={base_lr:.0e}, eta_min={eta_min:.0e}",
            plot_dir / "cosine_annealing_warmup.png",
        )

        # Verify plot was created
        assert (plot_dir / "cosine_annealing_warmup.png").exists()


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

    def test_plot_full_cycle(
        self, optimizer: torch.optim.Optimizer, plot_dir: Path, base_lr: float
    ) -> None:
        """Generate and save full warmup+decay curve plot."""
        warmup_epochs = 8
        decay_epochs = 12
        max_lr_factor = 10.0
        min_lr_factor = 1e-4

        scheduler = CyclicCosineAnnealingLR(
            optimizer,
            warmup_epochs=warmup_epochs,
            decay_epochs=decay_epochs,
            max_lr_factor=max_lr_factor,
            min_lr_factor=min_lr_factor,
        )

        total_epochs = warmup_epochs + decay_epochs
        epochs, lr_values = collect_lr_values(scheduler, total_epochs, optimizer)

        save_lr_plot(
            epochs,
            lr_values,
            f"CyclicCosineAnnealingLR\nwarmup={warmup_epochs}, decay={decay_epochs}, "
            f"max_factor={max_lr_factor}, min_factor={min_lr_factor}",
            plot_dir / "cyclic_cosine_annealing.png",
        )

        # Verify plot was created
        assert (plot_dir / "cyclic_cosine_annealing.png").exists()


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

    def test_plot_momentum_cycle(
        self, adamw_optimizer: torch.optim.Optimizer, plot_dir: Path
    ) -> None:
        """Generate and save momentum cycle plot."""
        warmup_epochs = 8
        decay_epochs = 12
        min_momentum_factor = 0.85 / 0.95
        max_momentum_factor = 1.0

        scheduler = CyclicMomentumScheduler(
            adamw_optimizer,
            warmup_epochs=warmup_epochs,
            decay_epochs=decay_epochs,
            min_momentum_factor=min_momentum_factor,
            max_momentum_factor=max_momentum_factor,
        )

        total_epochs = warmup_epochs + decay_epochs
        epochs, momentum_values = collect_momentum_values(scheduler, total_epochs, adamw_optimizer)

        # Use linear scale for momentum (values are close to 1.0)
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 6))
        plt.plot(epochs, momentum_values, "b-", linewidth=2, marker="o", markersize=3)
        plt.xlabel("Epoch", fontsize=12)
        plt.ylabel("Momentum (beta1)", fontsize=12)
        plt.title(
            f"CyclicMomentumScheduler\nwarmup={warmup_epochs}, decay={decay_epochs}, "
            f"min_factor={min_momentum_factor:.3f}, max_factor={max_momentum_factor}",
            fontsize=14,
        )
        plt.grid(True, alpha=0.3)

        # Add value annotations
        if len(momentum_values) > 0:
            plt.annotate(
                f"Start: {momentum_values[0]:.4f}",
                xy=(epochs[0], momentum_values[0]),
                xytext=(epochs[0] + 1, momentum_values[0] + 0.01),
                fontsize=9,
            )
            # Find minimum
            min_idx = momentum_values.index(min(momentum_values))
            plt.annotate(
                f"Min: {momentum_values[min_idx]:.4f}",
                xy=(epochs[min_idx], momentum_values[min_idx]),
                xytext=(epochs[min_idx] + 1, momentum_values[min_idx] - 0.02),
                fontsize=9,
            )

        plt.tight_layout()
        plt.savefig(plot_dir / "cyclic_momentum.png", dpi=150)
        plt.close()

        # Verify plot was created
        assert (plot_dir / "cyclic_momentum.png").exists()


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

    def test_plot_full_schedule(
        self, optimizer: torch.optim.Optimizer, plot_dir: Path, base_lr: float
    ) -> None:
        """Generate and save full linear warmup + cosine decay plot."""
        warmup_epochs = 5
        max_epochs = 20
        warmup_start_lr = 1e-6
        eta_min = 1e-8

        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer,
            warmup_epochs=warmup_epochs,
            max_epochs=max_epochs,
            warmup_start_lr=warmup_start_lr,
            eta_min=eta_min,
        )

        epochs, lr_values = collect_lr_values(scheduler, max_epochs, optimizer)

        save_lr_plot(
            epochs,
            lr_values,
            f"LinearWarmupCosineAnnealingLR\nwarmup={warmup_epochs}, total={max_epochs}, "
            f"start_lr={warmup_start_lr:.0e}, base_lr={base_lr:.0e}, eta_min={eta_min:.0e}",
            plot_dir / "linear_warmup_cosine_annealing.png",
        )

        # Verify plot was created
        assert (plot_dir / "linear_warmup_cosine_annealing.png").exists()


class TestAllSchedulersComparison:
    """Test that generates a comparison plot of all schedulers."""

    def test_comparison_plot(
        self,
        simple_model: nn.Module,
        plot_dir: Path,
        base_lr: float,
    ) -> None:
        """Generate comparison plot of all LR schedulers."""
        import matplotlib.pyplot as plt

        total_epochs = 20
        warmup_epochs = 5

        # Create separate optimizers for each scheduler
        opt1 = torch.optim.SGD(simple_model.parameters(), lr=base_lr)
        opt2 = torch.optim.SGD(simple_model.parameters(), lr=base_lr)
        opt3 = torch.optim.SGD(simple_model.parameters(), lr=base_lr)

        # Create schedulers
        schedulers = {
            "CosineAnnealingLR": CosineAnnealingLR(opt1, T_max=total_epochs, eta_min=1e-8),
            "CyclicCosineAnnealingLR": CyclicCosineAnnealingLR(
                opt2,
                warmup_epochs=warmup_epochs,
                decay_epochs=total_epochs - warmup_epochs,
                max_lr_factor=10.0,
                min_lr_factor=1e-4,
            ),
            "LinearWarmupCosineAnnealingLR": LinearWarmupCosineAnnealingLR(
                opt3,
                warmup_epochs=warmup_epochs,
                max_epochs=total_epochs,
                warmup_start_lr=1e-6,
                eta_min=1e-8,
            ),
        }

        optimizers = {
            "CosineAnnealingLR": opt1,
            "CyclicCosineAnnealingLR": opt2,
            "LinearWarmupCosineAnnealingLR": opt3,
        }

        # Collect LR values for each scheduler
        all_data = {}
        for name, scheduler in schedulers.items():
            epochs, lr_values = collect_lr_values(scheduler, total_epochs, optimizers[name])
            all_data[name] = (epochs, lr_values)

        # Create comparison plot
        plt.figure(figsize=(12, 8))
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

        for (name, (epochs, lr_values)), color in zip(all_data.items(), colors):
            plt.plot(
                epochs,
                lr_values,
                linewidth=2,
                marker="o",
                markersize=3,
                label=name,
                color=color,
            )

        plt.xlabel("Epoch", fontsize=12)
        plt.ylabel("Learning Rate", fontsize=12)
        plt.title(
            f"LR Scheduler Comparison\nbase_lr={base_lr:.0e}, total_epochs={total_epochs}",
            fontsize=14,
        )
        plt.legend(fontsize=10, loc="upper right")
        plt.grid(True, alpha=0.3)
        plt.yscale("log")

        # Add vertical line for warmup end
        plt.axvline(x=warmup_epochs, color="gray", linestyle="--", alpha=0.5)
        plt.text(
            warmup_epochs + 0.2,
            plt.ylim()[1] * 0.5,
            "Warmup ends",
            fontsize=9,
            color="gray",
        )

        plt.tight_layout()
        plt.savefig(plot_dir / "scheduler_comparison.png", dpi=150)
        plt.close()

        # Verify plot was created
        assert (plot_dir / "scheduler_comparison.png").exists()
