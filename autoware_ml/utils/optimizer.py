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

"""Optimizer and scheduler construction helpers for Autoware-ML models."""

from __future__ import annotations

import inspect
import logging
from collections.abc import Mapping, Sequence
from functools import partial
from typing import Any

import torch.nn as nn
from omegaconf import OmegaConf
from torch.optim import Optimizer

logger = logging.getLogger(__name__)


def materialize_partial_kwargs(value: Any) -> Any:
    """Convert OmegaConf containers into plain Python values."""
    if OmegaConf.is_config(value):
        return OmegaConf.to_container(value, resolve=True)
    if isinstance(value, Mapping):
        return {key: materialize_partial_kwargs(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(materialize_partial_kwargs(item) for item in value)
    if isinstance(value, list):
        return [materialize_partial_kwargs(item) for item in value]
    return value


def _get_partial_keywords(factory: Any) -> Mapping[str, Any]:
    """Return keywords bound into a ``functools.partial`` factory."""
    return factory.keywords if isinstance(factory, partial) and factory.keywords else {}


def call_configured_factory(factory: Any, /, **runtime_kwargs: Any) -> Any:
    """Call a configured factory after materializing OmegaConf containers.

    Hydra partials can retain ``DictConfig`` or ``ListConfig`` values in their
    bound arguments. Normalize both the bound values and the runtime arguments
    before calling the underlying factory so downstream PyTorch code receives
    plain Python containers.

    Args:
        factory: Callable or ``functools.partial`` factory.
        **runtime_kwargs: Additional keyword arguments supplied at call time.

    Returns:
        The object created by the factory.
    """
    materialized_runtime_kwargs = materialize_partial_kwargs(runtime_kwargs)
    if isinstance(factory, partial):
        materialized_args = tuple(materialize_partial_kwargs(arg) for arg in factory.args)
        materialized_keywords = materialize_partial_kwargs(factory.keywords or {})
        return factory.func(
            *materialized_args,
            **materialized_keywords,
            **materialized_runtime_kwargs,
        )
    return factory(**materialized_runtime_kwargs)


def build_optimizer_param_groups(
    model: nn.Module,
    group_overrides: Mapping[str, Mapping[str, Any]] | None = None,
) -> list[nn.Parameter] | list[dict[str, Any]]:
    """Build optimizer parameter groups from model-defined named groups.

    The model defines structural parameter groups via ``build_optimizer_groups``.
    This keeps grouping logic near the model rather than encoding it with
    fragile parameter-name filters.

    Args:
        model: Model providing ``build_optimizer_groups`` and parameters.
        group_overrides: Optional per-group optimizer overrides keyed by group
            name, for example ``{"backbone": {"lr": 1e-4}}``.

    Returns:
        Either a flat list of parameters or a list of optimizer parameter group
        dictionaries compatible with PyTorch optimizers.
    """
    if hasattr(model, "build_optimizer_groups"):
        named_groups: Mapping[str, Sequence[nn.Parameter]] = model.build_optimizer_groups()
    else:
        named_groups = {"default": [p for p in model.parameters() if p.requires_grad]}
    overrides = materialize_partial_kwargs(dict(group_overrides or {}))

    unknown_overrides = set(overrides) - set(named_groups)
    if unknown_overrides:
        raise ValueError(f"Unknown optimizer group override(s): {sorted(unknown_overrides)}")

    if set(named_groups) == {"default"} and not overrides.get("default"):
        return list(named_groups["default"])

    param_groups: list[dict[str, Any]] = []
    for group_name, params in named_groups.items():
        group_override = dict(overrides.pop(group_name, {}))
        param_groups.append({"params": list(params), **group_override})
        logger.info(
            "Optimizer param group '%s' has %d tensors%s.",
            group_name,
            len(params),
            f" with overrides {group_override}" if group_override else "",
        )

    if overrides:
        raise ValueError(f"Unknown optimizer group override(s): {sorted(overrides)}")

    return param_groups


def build_lightning_optimizer_config(
    model: nn.Module,
    optimizer_factory: Any,
    scheduler_factory: Any = None,
    *,
    optimizer_group_overrides: Mapping[str, Mapping[str, Any]] | None = None,
    scheduler_config: Mapping[str, Any] | None = None,
    estimated_stepping_batches: int | None = None,
) -> Optimizer | dict[str, Any]:
    """Build the Lightning optimizer configuration for a model.

    Args:
        model: Model whose parameters are optimized.
        optimizer_factory: Configured optimizer factory.
        scheduler_factory: Optional configured scheduler factory.
        optimizer_group_overrides: Optional per-group optimizer overrides keyed
            by the names returned from ``model.build_optimizer_groups()``.
        scheduler_config: Optional Lightning scheduler metadata such as
            ``interval``, ``frequency``, or ``monitor``.
        estimated_stepping_batches: Total number of optimizer steps, used to
            auto-fill scheduler ``total_steps`` when the parameter is declared
            but not already bound in the factory.

    Returns:
        Optimizer instance or a Lightning optimizer configuration dictionary.
    """
    param_groups = build_optimizer_param_groups(model, optimizer_group_overrides)
    optimizer = call_configured_factory(optimizer_factory, params=param_groups)

    if scheduler_factory is None:
        return optimizer

    scheduler_kwargs: dict[str, Any] = {"optimizer": optimizer}
    sig = inspect.signature(scheduler_factory)
    bound_kwargs = _get_partial_keywords(scheduler_factory)
    if (
        estimated_stepping_batches is not None
        and "total_steps" in sig.parameters
        and "total_steps" not in bound_kwargs
    ):
        scheduler_kwargs["total_steps"] = estimated_stepping_batches

    scheduler = call_configured_factory(scheduler_factory, **scheduler_kwargs)
    lr_scheduler_config = {
        "scheduler": scheduler,
        **materialize_partial_kwargs(dict(scheduler_config or {})),
    }

    return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_config}
