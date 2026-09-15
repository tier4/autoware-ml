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

import pytest
import torch
import torch.nn as nn


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
    """Create SGD optimizer with known base LR."""
    return torch.optim.SGD(simple_model.parameters(), lr=base_lr)


@pytest.fixture
def adamw_optimizer(simple_model: nn.Module, base_lr: float) -> torch.optim.Optimizer:
    """Create AdamW optimizer for momentum testing."""
    return torch.optim.AdamW(simple_model.parameters(), lr=base_lr, betas=(0.9, 0.999))
