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

import math
from functools import partial

import pytest
import torch

from autoware_ml.utils.optimizer import build_lightning_optimizer_config
from autoware_ml.utils.schedulers.cosine_annealing import CosineAnnealingLR
from autoware_ml.utils.schedulers.cyclic_cosine_annealing import CyclicCosineAnnealingLR
from autoware_ml.utils.schedulers.cyclic_momentum import CyclicMomentumScheduler
from autoware_ml.utils.schedulers.iter_warmup_epoch_cosine import IterWarmupEpochCosineLR
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


class TestIterWarmupEpochCosineLR:
    """Tests for the iteration-warmup, epoch-cosine schedule."""

    @staticmethod
    def _build(total_steps: int, max_epochs: int = 10, warmup_iters: int = 5):
        model = torch.nn.Linear(2, 2)
        optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
        scheduler = IterWarmupEpochCosineLR(
            optimizer,
            total_steps=total_steps,
            max_epochs=max_epochs,
            warmup_iters=warmup_iters,
            warmup_start_factor=1.0 / 3.0,
            eta_min_factor=1e-4,
        )
        return optimizer, scheduler

    @staticmethod
    def _cosine_factor(epoch: int, max_epochs: int, eta_min_factor: float = 1e-4) -> float:
        return eta_min_factor + (1.0 - eta_min_factor) * 0.5 * (
            1.0 + math.cos(math.pi * epoch / max_epochs)
        )

    def test_schedule_shape(self) -> None:
        """
        Input: 100 steps in ten stages of 10 steps, with a 5-step warmup from a third of the
        learning rate.
        Expected: the learning rate starts at a third of the base one, reaches it once the
        warmup is over within the first stage, and then follows the cosine over the stages.
        Check: the learning rate before the first step, after the warmup, in the second stage
        and in the last one.
        """
        optimizer, scheduler = self._build(total_steps=100)
        assert optimizer.param_groups[0]["lr"] == pytest.approx(1.0 / 3.0)
        lrs = []
        for _ in range(100):
            optimizer.step()
            scheduler.step()
            lrs.append(optimizer.param_groups[0]["lr"])
        # After warmup and within the first epoch the factor is the epoch-0 cosine.
        assert lrs[5] == pytest.approx(1.0)
        # Epoch boundaries follow the cosine over epoch indices exactly.
        assert lrs[10] == pytest.approx(self._cosine_factor(1, 10))
        assert lrs[95] == pytest.approx(self._cosine_factor(9, 10))

    def test_rejects_fewer_steps_than_epochs(self) -> None:
        """
        Input: 5 steps for ten cosine stages.
        Expected: a stage needs at least one step, so the schedule cannot be built.
        Check: a ValueError asking for at least one step per epoch is raised.
        """
        with pytest.raises(ValueError, match="at least one step per epoch"):
            self._build(total_steps=5, max_epochs=10)

    def test_trailing_steps_stay_at_the_last_epoch_factor(self) -> None:
        """
        Input: 105 steps for ten stages, as a trainer.max_steps cap can make it: ten stages of
        10 steps and 5 trailing steps.
        Expected: the trailing steps keep the learning rate of the last stage instead of
        dropping to the eta_min floor.
        Check: the learning rate after all 105 steps is the one of the last stage.
        """
        optimizer, scheduler = self._build(total_steps=105)
        for _ in range(105):
            optimizer.step()
            scheduler.step()
        assert optimizer.param_groups[0]["lr"] == pytest.approx(self._cosine_factor(9, 10))

    def test_a_max_steps_cap_runs_the_whole_cosine_inside_the_cap(self) -> None:
        """
        Input: total_steps = 300 and max_epochs = 10, as the framework fills them when
        trainer.max_steps = 300 caps a run of ten 100-batch epochs.
        Expected: a stage is total_steps // max_epochs = 30 steps rather than a data epoch,
        so all ten stages of the cosine run inside the 300 capped steps.
        Check: the learning rate changes every 30 steps and is at the last stage from step 270.
        """
        optimizer, scheduler = self._build(total_steps=300, warmup_iters=0)
        lrs = []
        for _ in range(300):
            lrs.append(optimizer.param_groups[0]["lr"])
            optimizer.step()
            scheduler.step()

        changes = [step for step in range(1, 300) if lrs[step] != lrs[step - 1]]
        assert changes == list(range(30, 300, 30))
        assert lrs[270] == pytest.approx(self._cosine_factor(9, 10))

    def test_resume_reproduces_the_original_trajectory(self) -> None:
        """
        Input: a schedule stepped 15 times and saved, then restored into one built with
        another ``total_steps``, as the framework re-derives it on resume (a
        ``trainer.max_steps`` cap or a changed batch size can make it differ).
        Expected: the restored state overrides the constructor arguments, so the resumed
        schedule replays the original one.
        Check: the 25 learning rates after the resume against the same steps of an
        uninterrupted run.
        """
        reference_optimizer, reference = self._build(total_steps=100)
        reference_lrs = []
        for _ in range(40):
            reference_optimizer.step()
            reference.step()
            reference_lrs.append(reference_optimizer.param_groups[0]["lr"])

        optimizer_a, scheduler_a = self._build(total_steps=100)
        for _ in range(15):
            optimizer_a.step()
            scheduler_a.step()
        state = scheduler_a.state_dict()

        # Rebuild with a different total_steps estimate, then restore.
        optimizer_b, scheduler_b = self._build(total_steps=57)
        scheduler_b.load_state_dict(state)
        resumed_lrs = []
        for _ in range(25):
            optimizer_b.step()
            scheduler_b.step()
            resumed_lrs.append(optimizer_b.param_groups[0]["lr"])

        assert resumed_lrs == reference_lrs[15:]

    def test_builds_from_the_framework_with_max_epochs_bound_in_the_config(self) -> None:
        """
        Input: a scheduler factory binding ``max_epochs`` the way a config does with
        ``${trainer.max_epochs}``, built through the framework optimizer helper.
        Expected: the framework fills ``total_steps`` from the trainer estimate and leaves the
        bound ``max_epochs`` alone, so the epoch length follows both.
        Check: the built scheduler reports the bound epochs and the derived steps per epoch.
        """
        model = torch.nn.Linear(2, 2)

        optimizers = build_lightning_optimizer_config(
            model,
            optimizer_factory=lambda params: torch.optim.SGD(params, lr=1.0),
            scheduler_factory=partial(
                IterWarmupEpochCosineLR,
                max_epochs=4,
                warmup_iters=1,
                warmup_start_factor=0.5,
                eta_min_factor=1e-3,
            ),
            scheduler_config={"interval": "step"},
            estimated_stepping_batches=40,
        )

        scheduler = optimizers["lr_scheduler"]["scheduler"]
        assert (scheduler.max_epochs, scheduler.steps_per_epoch) == (4, 10)

    def test_every_group_multiplies_the_warmup_into_the_cosine_stage(self) -> None:
        """
        Input: param groups at learning rates 1.0 and 0.1, four cosine stages of 10 steps, and
        a 15-step warmup from half the learning rate that reaches into the second stage.
        Expected: every group runs at its own learning rate times the warmup factor of the
        step and the cosine factor of the step's stage. The cosine starts at step 0 instead of
        after the warmup, the two factors multiply where they overlap, and the cosine holds
        for a whole stage while the warmup advances every step.
        Check: the learning rates of all 40 steps against that product, and the one at step
        12, inside the warmup and the second stage, against its value.
        """
        weight, bias = torch.nn.Linear(2, 2).parameters()
        optimizer = torch.optim.SGD(
            [{"params": [weight], "lr": 1.0}, {"params": [bias], "lr": 0.1}], lr=1.0
        )
        scheduler = IterWarmupEpochCosineLR(
            optimizer,
            total_steps=40,
            max_epochs=4,
            warmup_iters=15,
            warmup_start_factor=0.5,
            eta_min_factor=0.0,
        )
        lrs = []
        for _ in range(40):
            lrs.append([group["lr"] for group in optimizer.param_groups])
            optimizer.step()
            scheduler.step()

        def factor(step: int) -> float:
            warmup = 0.5 + 0.5 * step / 15 if step < 15 else 1.0
            return warmup * self._cosine_factor(step // 10, 4, eta_min_factor=0.0)

        assert lrs == [pytest.approx([factor(step), 0.1 * factor(step)]) for step in range(40)]
        assert lrs[12][0] == pytest.approx(0.9 * 0.5 * (1.0 + math.cos(math.pi / 4)))

    def test_every_group_anneals_towards_the_floor_of_the_default_learning_rate(self) -> None:
        """
        Input: an optimizer whose default learning rate is 1.0, with a backbone group at 0.1,
        stepped through 100 one-step cosine stages without warmup and a floor factor of 0.01.
        Expected: the floor is one absolute learning rate, 0.01 times the default one, shared
        by both groups as the reference recipe's eta_min = lr * 1e-4 is, so every group
        anneals from its own learning rate towards that floor, and the backbone ends at the
        same floor as the default group instead of at a hundredth of its own rate.
        Check: the learning rates of both groups at every step against the floor plus the
        stage's cosine share of the gap to their own rate, and both last learning rates
        against the floor.
        """
        weight, bias = torch.nn.Linear(2, 2).parameters()
        optimizer = torch.optim.SGD([{"params": [weight]}, {"params": [bias], "lr": 0.1}], lr=1.0)
        scheduler = IterWarmupEpochCosineLR(
            optimizer,
            total_steps=100,
            max_epochs=100,
            warmup_iters=0,
            warmup_start_factor=1.0,
            eta_min_factor=0.01,
        )
        lrs = []
        for _ in range(100):
            lrs.append([group["lr"] for group in optimizer.param_groups])
            optimizer.step()
            scheduler.step()

        def annealed(base_lr: float, step: int) -> float:
            return 0.01 + (base_lr - 0.01) * 0.5 * (1.0 + math.cos(math.pi * step / 100))

        assert lrs == [
            pytest.approx([annealed(1.0, step), annealed(0.1, step)]) for step in range(100)
        ]
        assert lrs[-1] == pytest.approx([0.01, 0.01], abs=3e-4)

    def test_needs_the_default_learning_rate_of_the_optimizer(self) -> None:
        """
        Input: an optimizer whose defaults hold no learning rate.
        Expected: the cosine floor is a fraction of the optimizer's default learning rate, so
        the schedule cannot be built without one.
        Check: a ValueError naming optimizer.defaults is raised.
        """
        optimizer = torch.optim.SGD(torch.nn.Linear(2, 2).parameters(), lr=1.0)
        del optimizer.defaults["lr"]

        with pytest.raises(ValueError, match="optimizer.defaults"):
            IterWarmupEpochCosineLR(
                optimizer,
                total_steps=10,
                max_epochs=1,
                warmup_iters=0,
                warmup_start_factor=1.0,
                eta_min_factor=0.01,
            )
