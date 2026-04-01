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

"""Custom indexing operators with ONNX symbolic support."""

from __future__ import annotations

from typing import Any

import torch
from torch.autograd import Function
from torch.onnx.symbolic_helper import _get_tensor_sizes


def _set_output_shape(value: torch.Tensor, sizes: list[int | None]) -> None:
    """Attach an inferred tensor shape to a symbolic ONNX value when possible."""
    if hasattr(value.type(), "with_sizes"):
        value.setType(value.type().with_sizes(sizes))


class _Unique(Function):
    """Expose ``torch.unique`` through the custom ``autoware::CustomUnique`` op."""

    @staticmethod
    def symbolic(g, x: torch.Tensor):
        outputs = g.op("autoware::CustomUnique", x, outputs=4)
        x_shape = _get_tensor_sizes(x)
        if x_shape is not None:
            _set_output_shape(outputs[0], [None])
            _set_output_shape(outputs[1], [None])
            _set_output_shape(outputs[2], [None])
            _set_output_shape(outputs[3], [1])
        return outputs

    @staticmethod
    def forward(ctx: Any, x: torch.Tensor):
        """Return unique values and indexing tensors for ONNX export."""
        del ctx
        unique_values, inverse_indices, counts = torch.unique(
            x,
            sorted=True,
            return_inverse=True,
            return_counts=True,
        )
        num_out = torch._shape_as_tensor(unique_values).to(x.device)[0]
        return unique_values, inverse_indices, counts, num_out


class _Argsort(Function):
    """Expose ``torch.sort(...).indices`` through the custom ``autoware::Argsort`` op."""

    @staticmethod
    def symbolic(g, x: torch.Tensor):
        output = g.op("autoware::Argsort", x, outputs=1)
        x_shape = _get_tensor_sizes(x)
        if x_shape is not None:
            _set_output_shape(output, list(x_shape))
        return output

    @staticmethod
    def forward(ctx: Any, x: torch.Tensor):
        """Return sort indices for serialized point ordering."""
        del ctx
        return torch.sort(x).indices


def unique(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return unique values plus inverse indices, counts, and output count.

    During ONNX export this emits the custom ``autoware::CustomUnique`` operator.
    """
    if torch.onnx.is_in_onnx_export():
        return _Unique.apply(x)

    unique_values, inverse_indices, counts = torch.unique(
        x,
        sorted=True,
        return_inverse=True,
        return_counts=True,
    )
    num_out = torch.tensor([unique_values.numel()], device=x.device, dtype=unique_values.dtype)
    return unique_values, inverse_indices, counts, num_out


def argsort(x: torch.Tensor) -> torch.Tensor:
    """Return sorted indices, emitting ``autoware::Argsort`` during ONNX export."""
    if torch.onnx.is_in_onnx_export():
        return _Argsort.apply(x)
    return torch.sort(x).indices
