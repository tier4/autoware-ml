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

"""Shared fixtures for scheduler tests."""

from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import pytest
import torch
import torch.nn as nn


@pytest.fixture
def plot_dir() -> Path:
    """Create and return the plots directory."""
    plots_path = Path(__file__).parent / "artifacts"
    plots_path.mkdir(parents=True, exist_ok=True)
    return plots_path


@pytest.fixture
def simple_model() -> nn.Module:
    """Create a simple model for optimizer testing."""
    return nn.Linear(10, 2)


@pytest.fixture
def base_lr() -> float:
    """Base learning rate for testing."""
    return 1e-3


@pytest.fixture
def optimizer(simple_model: nn.Module, base_lr: float) -> torch.optim.Optimizer:
    """Create SGD optimizer with known base_lr."""
    return torch.optim.SGD(simple_model.parameters(), lr=base_lr)


@pytest.fixture
def adamw_optimizer(simple_model: nn.Module, base_lr: float) -> torch.optim.Optimizer:
    """Create AdamW optimizer for momentum testing."""
    return torch.optim.AdamW(simple_model.parameters(), lr=base_lr, betas=(0.9, 0.999))


def save_lr_plot(
    epochs: List[int],
    lr_values: List[float],
    title: str,
    output_path: Path,
    ylabel: str = "Learning Rate",
) -> None:
    """Save learning rate plot to file.

    Args:
        epochs: List of epoch numbers.
        lr_values: List of learning rate values.
        title: Plot title.
        output_path: Path to save the plot.
        ylabel: Y-axis label.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, lr_values, "b-", linewidth=2, marker="o", markersize=3)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.yscale("log")

    # Add value annotations for key points
    if len(lr_values) > 0:
        plt.annotate(
            f"Start: {lr_values[0]:.2e}",
            xy=(epochs[0], lr_values[0]),
            xytext=(epochs[0] + 1, lr_values[0] * 1.5),
            fontsize=9,
            arrowprops=dict(arrowstyle="->", color="gray", alpha=0.5),
        )
        plt.annotate(
            f"End: {lr_values[-1]:.2e}",
            xy=(epochs[-1], lr_values[-1]),
            xytext=(epochs[-1] - 3, lr_values[-1] * 0.5),
            fontsize=9,
            arrowprops=dict(arrowstyle="->", color="gray", alpha=0.5),
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def collect_lr_values(
    scheduler,
    num_epochs: int,
    optimizer: torch.optim.Optimizer,
) -> Tuple[List[int], List[float]]:
    """Run scheduler for given epochs and collect LR values.

    Args:
        scheduler: LR scheduler instance.
        num_epochs: Number of epochs to simulate.
        optimizer: Optimizer instance.

    Returns:
        Tuple of (epochs list, lr values list).
    """
    epochs = []
    lr_values = []

    for epoch in range(num_epochs):
        current_lr = optimizer.param_groups[0]["lr"]
        epochs.append(epoch)
        lr_values.append(current_lr)
        scheduler.step()

    return epochs, lr_values


def collect_momentum_values(
    scheduler,
    num_epochs: int,
    optimizer: torch.optim.Optimizer,
) -> Tuple[List[int], List[float]]:
    """Run momentum scheduler for given epochs and collect momentum values.

    Args:
        scheduler: Momentum scheduler instance.
        num_epochs: Number of epochs to simulate.
        optimizer: Optimizer instance.

    Returns:
        Tuple of (epochs list, momentum values list).
    """
    epochs = []
    momentum_values = []

    for epoch in range(num_epochs):
        # Get current momentum (beta1 for AdamW)
        if "betas" in optimizer.param_groups[0]:
            current_momentum = optimizer.param_groups[0]["betas"][0]
        else:
            current_momentum = optimizer.param_groups[0].get("momentum", 0.9)

        epochs.append(epoch)
        momentum_values.append(current_momentum)
        scheduler.step()

    return epochs, momentum_values
