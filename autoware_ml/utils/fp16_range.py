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

"""Largest magnitudes an export module reaches on its trace batch, checked against fp16."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import torch
from jaxtyping import Shaped
from torch._ops import OpOverload
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_leaves
from torch.utils.hooks import RemovableHandle

from autoware_ml.utils.deploy import ExportSpec

FP16_MAX = float(torch.finfo(torch.float16).max)

# The float32 extremes are empty-reduction sentinels, an fp16 engine keeps them as infinities
_FLOAT32_MAX = float(torch.finfo(torch.float32).max)

# Offenders a refused export names
_REPORTED = 10


def _peak(tensor: Shaped[torch.Tensor, "..."]) -> float:
    """Largest magnitude of a float tensor, NaN when it holds one.

    Minus infinity masks and the float32 extremes are sentinels and left out, a positive infinity
    is an overflow and counts.
    """
    value = tensor.detach().float()
    magnitude = value.abs()
    sentinel = (value == -math.inf) | (magnitude == _FLOAT32_MAX)
    return float(magnitude.masked_fill(sentinel, 0.0).amax())


class _RangeProbe(TorchDispatchMode):
    """Record the largest magnitude of every float tensor an op reads or returns."""

    def __init__(self, root: str) -> None:
        """Start the scope stack at the export module."""
        super().__init__()
        self.scopes: list[str] = [root]
        self.maxima: dict[str, float] = {}

    def enter(self, scope: str) -> Callable[..., None]:
        """A forward pre-hook that opens the scope of a module."""

        def hook(*_: Any) -> None:
            self.scopes.append(scope)

        return hook

    def leave(self, *_: Any) -> None:
        """A forward hook that closes the innermost scope, it returns nothing to keep the output."""
        self.scopes.pop()

    def record(self, name: str, value: Shaped[torch.Tensor, "..."]) -> None:
        """Keep the largest magnitude of a float tensor under ``name``, a NaN stays."""
        if value.is_floating_point() and value.numel() > 0:
            peak = _peak(value)
            current = self.maxima.get(name, 0.0)
            if math.isnan(peak) or peak > current:
                self.maxima[name] = current if math.isnan(current) else peak

    def __torch_dispatch__(
        self,
        func: OpOverload,
        types: tuple[type, ...],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        """Run the op and record its float inputs and outputs under the current module."""
        output = func(*args, **(kwargs or {}))
        name = f"{self.scopes[-1]} {func.overloadpacket.__name__}"
        for leaf in tree_leaves((args, kwargs, output)):
            if isinstance(leaf, torch.Tensor):
                self.record(name, leaf)
        return output


def fp16_range_report(spec: ExportSpec, module_name: str) -> dict[str, float]:
    """Run an export module once and report the largest magnitude of its float tensors.

    Args:
        spec: The export module and its trace batch.
        module_name: The exported module, the root of the reported names.

    Returns:
        Largest magnitude by graph input and by module path and op, NaN where one held a NaN.
    """
    probe = _RangeProbe(module_name)
    for input_name, value in zip(spec.input_names, spec.args, strict=True):
        if isinstance(value, torch.Tensor):
            probe.record(f"{module_name} input {input_name}", value)
    handles: list[RemovableHandle] = []
    for path, child in spec.module.named_modules():
        scope = f"{module_name}.{path}" if path else module_name
        handles.append(child.register_forward_pre_hook(probe.enter(scope)))
        handles.append(child.register_forward_hook(probe.leave))
    with torch.no_grad(), probe:
        spec.module(*spec.args)
    for handle in handles:
        handle.remove()
    return probe.maxima


def validate_fp16_range(spec: ExportSpec, module_name: str, share: float) -> None:
    """Refuse an export module whose trace batch leaves a share of the fp16 range.

    The deployed engines may run every layer in fp16, where a larger value overflows.

    Args:
        spec: The export module and its trace batch.
        module_name: The exported module, named in the error.
        share: Share of the fp16 maximum every float tensor has to stay within.

    Raises:
        ValueError: If ``share`` is not in ``(0, 1]``, or if a tensor exceeds the limit or holds
            a NaN.
    """
    if not 0.0 < share <= 1.0:
        raise ValueError(f"deploy.onnx.fp16_range_share must be in (0, 1], got {share}.")
    limit = share * FP16_MAX
    maxima = fp16_range_report(spec, module_name)
    offenders = sorted(
        ((name, peak) for name, peak in maxima.items() if not peak <= limit),
        key=lambda item: math.inf if math.isnan(item[1]) else item[1],
        reverse=True,
    )
    if offenders:
        listed = ", ".join(f"{name} reaches {peak:.6g}" for name, peak in offenders[:_REPORTED])
        raise ValueError(
            f"Module '{module_name}' leaves {share} of the fp16 range ({limit:.6g}) on its trace "
            f"batch, {len(offenders)} tensors: {listed}."
        )
