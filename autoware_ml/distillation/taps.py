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

"""Taps name the tensors a distillation term compares.

A tap is one of two things. An attribute path into the model outputs, such as
``segmentation3d_head_outputs.logits``, reads a tensor the model returns. A module path with the
``module:`` prefix, such as ``module:bev_neck``, reads the output of a submodule through a forward
hook, so a term can compare features the model never returns. A mapping output of a module (a
PTv3 ``Point``) yields its ``feat`` entry unless ``@<key>`` names another one; a tuple output needs
``@<index>``.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import torch
import torch.nn as nn
from torch.utils.hooks import RemovableHandle

MODULE_TAP_PREFIX = "module:"
SELECTOR_SEPARATOR = "@"
DEFAULT_MAPPING_KEY = "feat"


class Tap:
    """Where a distillation term reads a tensor from.

    Args:
        spec: Attribute path into the model outputs, or ``module:<path>[@<selector>]``.

    Raises:
        ValueError: If the spec is empty or names an empty module path.
    """

    def __init__(self, spec: str) -> None:
        if not spec:
            raise ValueError("A tap needs a non-empty spec.")
        self.spec = spec
        self.module_path: str | None = None
        self.selector: str | None = None
        self.output_path: str | None = None
        if spec.startswith(MODULE_TAP_PREFIX):
            path, _, selector = spec[len(MODULE_TAP_PREFIX) :].partition(SELECTOR_SEPARATOR)
            if not path:
                raise ValueError(f"Tap {spec!r} names no module.")
            self.module_path = path
            self.selector = selector or None
        else:
            self.output_path = spec

    @property
    def is_module(self) -> bool:
        """Whether the tap reads a module output through a hook."""
        return self.module_path is not None

    def resolve(self, outputs: Any, features: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """Read the tensor the tap names.

        Args:
            outputs: Model outputs, read when the tap is an attribute path.
            features: Captured module outputs keyed by tap spec, read when the tap is a module
                tap.

        Returns:
            The named tensor.

        Raises:
            ValueError: If a module tap captured nothing, or the attribute path leads to
                something other than a tensor.
        """
        if self.is_module:
            if self.spec not in features:
                raise ValueError(
                    f"Tap {self.spec!r} captured no output. The module did not run, or the tap "
                    "was not registered on this model."
                )
            return features[self.spec]
        value = walk_path(outputs, self.output_path or "")
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"Tap {self.spec!r} leads to {type(value).__name__}, not a tensor.")
        return value

    def __repr__(self) -> str:
        """Show the spec."""
        return f"Tap({self.spec!r})"


def walk_path(value: Any, path: str) -> Any:
    """Follow a dotted path through attributes, mapping keys and sequence indices.

    Args:
        value: Root object.
        path: Dotted path, each part an attribute, a mapping key or a sequence index.

    Returns:
        The object at the end of the path.

    Raises:
        ValueError: If a part is not found or the path crosses a ``None``.
    """
    for part in path.split("."):
        if value is None:
            raise ValueError(f"Path {path!r} crosses None at {part!r}.")
        if isinstance(value, Mapping):
            if part not in value:
                raise ValueError(f"Path {path!r}: no key {part!r} in {type(value).__name__}.")
            value = value[part]
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            if not part.isdigit() or int(part) >= len(value):
                raise ValueError(f"Path {path!r}: {part!r} is no index of {type(value).__name__}.")
            value = value[int(part)]
        elif hasattr(value, part):
            value = getattr(value, part)
        else:
            raise ValueError(f"Path {path!r}: {type(value).__name__} has no {part!r}.")
    return value


def select_tensor(output: Any, selector: str | None) -> torch.Tensor:
    """Pick the tensor a module tap captures from the output of the module.

    Args:
        output: Forward output of the module.
        selector: Key of a mapping output or index of a tuple output, None for the default.

    Returns:
        The selected tensor.

    Raises:
        ValueError: If the output holds no tensor at the selector.
    """
    if isinstance(output, torch.Tensor):
        if selector is not None:
            raise ValueError(f"Selector {selector!r} given, but the module returns a tensor.")
        return output
    if isinstance(output, Mapping):
        key = selector if selector is not None else DEFAULT_MAPPING_KEY
        value = output.get(key)
    elif isinstance(output, Sequence):
        if selector is None or not selector.isdigit():
            raise ValueError(
                f"The module returns a {type(output).__name__}, give the index as '@<index>'."
            )
        value = output[int(selector)] if int(selector) < len(output) else None
    else:
        value = getattr(output, selector, None) if selector is not None else None
    if not isinstance(value, torch.Tensor):
        raise ValueError(
            f"No tensor at {selector!r} in the {type(output).__name__} the module returns."
        )
    return value


class CaptureStore:
    """Tensors the hooks of one model captured since the last clear.

    Attributes:
        tensors: Captured tensors keyed by tap spec.
        enabled: Whether hooks store what they see. Off, the hooks are free.
    """

    def __init__(self) -> None:
        self.tensors: dict[str, torch.Tensor] = {}
        self.enabled = False

    def clear(self) -> None:
        """Drop every captured tensor."""
        self.tensors.clear()


class FeatureCapture:
    """Forward hook that stores a clone of the output of one module.

    The clone matters: PTv3 unpooling and in-place activations overwrite module outputs after
    they were returned. The clone keeps the autograd graph, so a student feature stays trainable.
    A deep copy of a hooked module (export prepares copies of the heads) keeps this very hook
    rather than copying the store it writes to.

    Args:
        store: Store receiving the tensor.
        key: Key the tensor is stored under, the spec of the tap.
        selector: Selector of the tap, see :func:`select_tensor`.
    """

    def __init__(self, store: CaptureStore, key: str, selector: str | None) -> None:
        self.store = store
        self.key = key
        self.selector = selector

    def __call__(self, module: nn.Module, inputs: tuple[Any, ...], output: Any) -> None:
        """Store the selected output tensor when the store is enabled."""
        del module, inputs
        if self.store.enabled:
            self.store.tensors[self.key] = select_tensor(output, self.selector).clone()

    def __deepcopy__(self, memo: dict[int, Any]) -> FeatureCapture:
        """Keep the hook shared instead of copying its store."""
        del memo
        return self


def register_taps(
    model: nn.Module, taps: Iterable[Tap], store: CaptureStore
) -> list[RemovableHandle]:
    """Hook the modules the module taps name, once per distinct tap.

    Args:
        model: Model whose submodules are hooked.
        taps: Taps of the terms; attribute-path taps are skipped.
        store: Store the hooks write to.

    Returns:
        Handles that remove the hooks.

    Raises:
        ValueError: If a module path names no submodule of the model.
    """
    handles: list[RemovableHandle] = []
    seen: set[str] = set()
    for tap in taps:
        if not tap.is_module or tap.spec in seen:
            continue
        seen.add(tap.spec)
        try:
            module = model.get_submodule(tap.module_path or "")
        except AttributeError as error:
            raise ValueError(
                f"Tap {tap.spec!r}: {type(model).__name__} has no submodule {tap.module_path!r}."
            ) from error
        handles.append(module.register_forward_hook(FeatureCapture(store, tap.spec, tap.selector)))
    return handles
