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

"""CSR-based segment reduction operators."""

from __future__ import annotations

import torch
from torch.autograd import Function
from torch.onnx.symbolic_helper import _get_tensor_sizes


class _SegmentCSR(Function):
    """Bridge CSR-style segment reduction into eager execution and ONNX export."""

    @staticmethod
    def symbolic(g, src: torch.Tensor, indptr: torch.Tensor, reduce: str):
        """Register the custom ONNX symbolic for segment reduction."""
        output = g.op("autoware::SegmentCSR", src, indptr, reduce_s=reduce, outputs=1)
        src_shape = _get_tensor_sizes(src)
        if src_shape is not None and hasattr(output.type(), "with_sizes"):
            output.setType(src.type().with_sizes([src_shape[0], src_shape[1]]))
        return output

    @staticmethod
    def forward(ctx, src: torch.Tensor, indptr: torch.Tensor, reduce: str) -> torch.Tensor:
        """Run segment reduction during eager execution."""
        lengths = indptr[1:] - indptr[:-1]
        return torch.segment_reduce(src, reduce=reduce, lengths=lengths)


def segment_csr(src: torch.Tensor, indptr: torch.Tensor, reduce: str) -> torch.Tensor:
    """Run CSR segment reduction.

    Args:
        src: Source tensor reduced over CSR segments.
        indptr: CSR index pointer array.
        reduce: Reduction type supported by ``torch.segment_reduce``.

    Returns:
        Reduced tensor.
    """
    lengths = indptr[1:] - indptr[:-1]
    if torch.onnx.is_in_onnx_export():
        return _SegmentCSR.apply(src, indptr, reduce)
    return torch.segment_reduce(src, reduce=reduce, lengths=lengths)
