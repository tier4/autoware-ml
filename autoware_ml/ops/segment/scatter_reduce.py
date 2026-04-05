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

"""ONNX export helpers for scatter-reduce operators."""

from __future__ import annotations

import torch
from torch.onnx import errors, symbolic_helper

_REGISTERED_OPSET_VERSIONS: set[int] = set()


@symbolic_helper.parse_args("v", "i", "v", "v", "s", "b")
def _symbolic_scatter_reduce(
    g,
    self: torch._C.Value,
    dim: int,
    index: torch._C.Value,
    src: torch._C.Value,
    reduce: str,
    include_self: bool,
) -> torch._C.Value:
    """Export ``scatter_reduce`` as ONNX ``ScatterElements``."""
    if reduce == "mean":
        raise errors.OnnxExporterError("ONNX does not support mean reduction for scatter_reduce")
    if not include_self:
        raise errors.OnnxExporterError(
            "ONNX does not support include_self=False for scatter_reduce"
        )

    reduction_map = {
        "sum": "add",
        "prod": "mul",
        "amin": "min",
        "amax": "max",
    }
    return g.op("ScatterElements", self, index, src, axis_i=dim, reduction_s=reduction_map[reduce])


def register_scatter_reduce_onnx_symbolic(opset_version: int) -> None:
    """Register the scatter-reduce ONNX symbolic exactly once."""
    if opset_version in _REGISTERED_OPSET_VERSIONS:
        return
    torch.onnx.register_custom_op_symbolic(
        "aten::scatter_reduce",
        _symbolic_scatter_reduce,
        opset_version,
    )
    _REGISTERED_OPSET_VERSIONS.add(opset_version)
