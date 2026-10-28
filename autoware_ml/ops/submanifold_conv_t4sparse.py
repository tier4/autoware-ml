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

"""Submanifold sparse convolution on t4sparse. Importable only with ``t4sparse`` installed."""

from __future__ import annotations

import t4sparse.torch
import torch
from spconv.pytorch.core import SparseConvTensor
from t4sparse.torch.compat import to_spconv_layer
from torch import nn

from autoware_ml.ops.submanifold_conv_spconv import with_features_as

__all__ = ["T4SparseSubMConv3d"]

_T4SPARSE_DTYPES = (torch.float16, torch.bfloat16)


class T4SparseSubMConv3d(t4sparse.torch.SubMConv3d):
    """t4sparse's submanifold convolution, falling back to spconv where t4sparse has no kernel.

    fp32 features outside autocast and CPU tensors run on an spconv twin that
    shares this layer's parameters, like
    :class:`~autoware_ml.ops.submanifold_conv_spconv.SpconvSubMConv3d`. The twin
    is not a submodule, so the state dict is unchanged.

    The two libraries' ``indice_key`` caches are incompatible, so the layers
    sharing a key must run on one library within a forward pass: disable
    autocast around whole stages, never around part of one.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Bypasses nn.Module's registration: the twin shares this layer's parameters and
        # must not appear in the state dict.
        self.__dict__["_spconv_twin"] = None

    @staticmethod
    def runs_on_t4sparse(features: torch.Tensor) -> bool:
        """Return whether t4sparse has a kernel for a call on ``features``.

        Args:
            features: The input features of the call.

        Returns:
            ``True`` on CUDA under autocast or for 16-bit features.
        """
        return features.is_cuda and (
            torch.is_autocast_enabled("cuda") or features.dtype in _T4SPARSE_DTYPES
        )

    def forward(self, x: SparseConvTensor) -> SparseConvTensor:
        """Convolve ``x`` on t4sparse, or on spconv where t4sparse has no kernel.

        Args:
            x: Sparse tensor holding the input features.

        Returns:
            Sparse tensor holding the convolved features.

        Raises:
            TypeError: Raised when the ``indice_key`` slot on ``x`` holds the other
                library's index structure, see the class docstring.
        """
        if self.runs_on_t4sparse(x.features):
            return super().forward(x)
        return self._forward_spconv(x)

    def _forward_spconv(self, x: SparseConvTensor) -> SparseConvTensor:
        cached = x.indice_dict.get(self.indice_key) if self.indice_key is not None else None
        if isinstance(cached, t4sparse.torch.Rulebook):
            raise TypeError(
                f"indice_key {self.indice_key!r} holds a rulebook built under autocast, but "
                "this call runs on spconv (fp32 outside autocast); the layers that share an "
                "indice_key must all run on one library, so disable autocast around the "
                "whole stage or none of it"
            )
        with torch.autocast("cuda", enabled=False):
            return self._spconv()(with_features_as(x, self.weight.dtype))

    def _spconv(self) -> nn.Module:
        """The spconv twin, rebuilt when the layer's parameters were replaced."""
        twin = self.__dict__["_spconv_twin"]
        if twin is None or twin.weight is not self.weight or twin.bias is not self.bias:
            twin = to_spconv_layer(self)
            self.__dict__["_spconv_twin"] = twin
        return twin
