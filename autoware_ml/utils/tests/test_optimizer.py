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

"""Unit tests for optimizer helper utilities."""

from __future__ import annotations

from functools import partial

import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from torch.optim.lr_scheduler import OneCycleLR

from autoware_ml.utils.optimizer import (
    build_lightning_optimizer_config,
    build_optimizer_param_groups,
    materialize_partial_kwargs,
)


class _ToyModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Linear(4, 4)
        self.block = nn.Linear(4, 4)

    def build_optimizer_groups(self) -> dict[str, list[nn.Parameter]]:
        return {
            "default": list(self.stem.parameters()),
            "block": list(self.block.parameters()),
        }


def test_build_optimizer_param_groups_assigns_named_groups() -> None:
    model = _ToyModule()

    param_groups = build_optimizer_param_groups(
        model,
        {"block": {"lr": 1e-4}},
    )

    assert isinstance(param_groups, list)
    assert len(param_groups) == 2
    assert "lr" not in param_groups[0]
    assert param_groups[1]["lr"] == 1e-4
    assert len(param_groups[0]["params"]) == 2
    assert len(param_groups[1]["params"]) == 2


def test_build_optimizer_param_groups_rejects_unknown_group_overrides() -> None:
    model = _ToyModule()

    with pytest.raises(ValueError, match="Unknown optimizer group override"):
        build_optimizer_param_groups(model, {"missing": {"lr": 1e-4}})


def test_build_optimizer_param_groups_returns_flat_params_without_overrides() -> None:
    class _DefaultOnlyModule(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = nn.Linear(4, 4)

        def build_optimizer_groups(self) -> dict[str, list[nn.Parameter]]:
            return {"default": list(self.linear.parameters())}

    model = _DefaultOnlyModule()
    params = build_optimizer_param_groups(model)

    assert isinstance(params, list)
    assert all(isinstance(parameter, nn.Parameter) for parameter in params)
    assert len(params) == 2


def test_materialize_partial_kwargs_converts_omegaconf_containers() -> None:
    value = OmegaConf.create({"max_lr": [1e-3, 1e-4], "betas": [0.9, 0.999]})

    materialized = materialize_partial_kwargs(value)

    assert materialized == {"max_lr": [1e-3, 1e-4], "betas": [0.9, 0.999]}


def test_build_lightning_optimizer_config_infers_total_steps_for_scheduler() -> None:
    model = _ToyModule()

    optimizers = build_lightning_optimizer_config(
        model,
        optimizer_factory=lambda params: torch.optim.AdamW(params, lr=1e-3),
        scheduler_factory=partial(OneCycleLR, max_lr=[1e-3, 1e-4]),
        optimizer_group_overrides={"block": {"lr": 1e-4}},
        scheduler_config={"interval": "step"},
        estimated_stepping_batches=10,
    )

    optimizer = optimizers["optimizer"]
    scheduler = optimizers["lr_scheduler"]["scheduler"]

    assert [group["max_lr"] for group in optimizer.param_groups] == [1e-3, 1e-4]
    assert optimizers["lr_scheduler"]["interval"] == "step"
    assert scheduler.total_steps == 10


def test_build_lightning_optimizer_config_materializes_partial_omegaconf_kwargs() -> None:
    model = _ToyModule()

    optimizers = build_lightning_optimizer_config(
        model,
        optimizer_factory=partial(torch.optim.AdamW, lr=1e-3),
        scheduler_factory=partial(
            OneCycleLR,
            max_lr=OmegaConf.create([1e-3, 1e-4]),
        ),
        optimizer_group_overrides={"block": {"lr": 1e-4}},
        scheduler_config={"interval": "step"},
        estimated_stepping_batches=10,
    )

    optimizer = optimizers["optimizer"]
    scheduler = optimizers["lr_scheduler"]["scheduler"]

    assert [group["max_lr"] for group in optimizer.param_groups] == [1e-3, 1e-4]
    assert scheduler.total_steps == 10


def test_build_lightning_optimizer_config_keeps_scheduler_metadata() -> None:
    model = _ToyModule()

    optimizers = build_lightning_optimizer_config(
        model,
        optimizer_factory=lambda params: torch.optim.AdamW(params, lr=1e-3),
        scheduler_factory=lambda optimizer: OneCycleLR(optimizer, max_lr=1e-3, total_steps=10),
        scheduler_config={"interval": "step", "frequency": 2, "monitor": "val/loss"},
    )

    assert optimizers["lr_scheduler"]["interval"] == "step"
    assert optimizers["lr_scheduler"]["frequency"] == 2
    assert optimizers["lr_scheduler"]["monitor"] == "val/loss"


def test_build_optimizer_param_groups_rejects_a_trainable_parameter_in_no_group() -> None:
    class _ForgetfulModule(_ToyModule):
        def build_optimizer_groups(self) -> dict[str, list[nn.Parameter]]:
            return {"default": list(self.stem.parameters())}

    with pytest.raises(ValueError, match="in no optimizer group: block.weight, block.bias"):
        build_optimizer_param_groups(_ForgetfulModule())


def test_build_optimizer_param_groups_adds_an_extra_group_for_uncovered_parameters() -> None:
    class _ModuleWithAdapter(_ToyModule):
        def __init__(self) -> None:
            super().__init__()
            self.adapter = nn.Linear(4, 4)

    model = _ModuleWithAdapter()

    param_groups = build_optimizer_param_groups(
        model, {"adapter": {"lr": 1e-5}}, extra_groups={"adapter": list(model.adapter.parameters())}
    )

    assert len(param_groups) == 3
    assert param_groups[2]["lr"] == 1e-5
    assert len(param_groups[2]["params"]) == 2


def test_build_optimizer_param_groups_drops_an_extra_group_the_model_already_covers() -> None:
    model = _ToyModule()

    param_groups = build_optimizer_param_groups(
        model, extra_groups={"block_again": list(model.block.parameters())}
    )

    assert len(param_groups) == 2


def test_build_optimizer_param_groups_rejects_an_extra_group_named_like_a_model_group() -> None:
    model = _ToyModule()
    model.adapter = nn.Linear(4, 4)

    with pytest.raises(ValueError, match="defined by the model and added as an extra group"):
        build_optimizer_param_groups(
            model, extra_groups={"block": list(model.adapter.parameters())}
        )
