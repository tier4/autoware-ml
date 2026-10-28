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

"""Submanifold sparse convolution on spconv or t4sparse.

:class:`SubMConv3d` is spconv's layer or, when ``t4sparse.torch`` is installed,
t4sparse's. Both share the constructor, the KRSC weight layout and the parameter
names, so checkpoints are interchangeable. t4sparse runs under autocast and for
16-bit features on CUDA; spconv runs everything else, in the dtype of its weights.
"""

from __future__ import annotations

from importlib.util import find_spec

import spconv.pytorch as spconv
import torch
from spconv.pytorch.core import SparseConvTensor
from torch import nn

__all__ = ["IS_T4SPARSE_AVAILABLE", "SpconvSubMConv3d", "SubMConv3d"]

# find_spec on a dotted name imports the parent package, which raises when it is absent.
IS_T4SPARSE_AVAILABLE = (
    find_spec("t4sparse") is not None and find_spec("t4sparse.torch") is not None
)

_T4SPARSE_DTYPES = (torch.float16, torch.bfloat16)


def _with_features_as(x: SparseConvTensor, dtype: torch.dtype) -> SparseConvTensor:
    """Return ``x`` with its features in ``dtype``, ``x`` itself when they already are."""
    if x.features.dtype == dtype:
        return x
    return x.replace_feature(x.features.to(dtype))


class SpconvSubMConv3d(spconv.SubMConv3d):
    """spconv's submanifold convolution, run outside autocast in the dtype of its weights.

    spconv's kernels need features and weights in one dtype, and autocast does
    not cast for them.
    """

    def forward(self, x: SparseConvTensor) -> SparseConvTensor:
        """Convolve ``x`` on spconv in the dtype of the weights.

        Args:
            x: Sparse tensor holding the input features.

        Returns:
            Sparse tensor holding the convolved features.
        """
        with torch.autocast("cuda", enabled=False):
            return super().forward(_with_features_as(x, self.weight.dtype))


if IS_T4SPARSE_AVAILABLE:
    import t4sparse.torch
    from t4sparse.torch.compat import to_spconv_layer

    class T4SparseSubMConv3d(t4sparse.torch.SubMConv3d):
        """t4sparse's submanifold convolution, falling back to spconv where t4sparse has no kernel.

        fp32 features outside autocast and CPU tensors run on an spconv twin that
        shares this layer's parameters, like :class:`SpconvSubMConv3d`. The twin is
        not a submodule, so the state dict is unchanged.

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
                return self._spconv()(_with_features_as(x, self.weight.dtype))

        def _spconv(self) -> nn.Module:
            """The spconv twin, rebuilt when the layer's parameters were replaced."""
            twin = self.__dict__["_spconv_twin"]
            if twin is None or twin.weight is not self.weight or twin.bias is not self.bias:
                twin = to_spconv_layer(self)
                self.__dict__["_spconv_twin"] = twin
            return twin

    SubMConv3d = T4SparseSubMConv3d
    """The submanifold convolution layer: t4sparse when installed, spconv otherwise."""
else:
    SubMConv3d = SpconvSubMConv3d
    """The submanifold convolution layer: t4sparse when installed, spconv otherwise."""
