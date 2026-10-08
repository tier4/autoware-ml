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

"""Backend selection for the submanifold sparse convolutions of the PTv3 family.

Every sparse convolution of PTv3 and LitePT is a stride-1 submanifold convolution:
the conditional positional encoding at the front of each block and, optionally,
the embedding stem. Two libraries implement it behind the same ``SubMConv3d``
call, weight layout and ``indice_key`` caching:

* ``spconv`` - the historical dependency. Its kernels accumulate in fp16 and are
  not registered with autocast, so a ``16-mixed`` run either aborts in the kernel
  tuner or trains on fp16 partial sums. It stays required for the strided
  detection encoders and for ONNX export.
* ``rulebook_torch`` - the in-house replacement developed by Max Schmeller (repository
  ``spconv-replacement``); this package only adapts PTv3 to it.
  Every kernel accumulates in fp32 and the autograd function casts inside, so
  mixed precision reproduces fp32 training. It only has kernels for fp16 and
  bf16 operands; an fp32 forward runs through a slow reference path, so it is
  meant for mixed precision (``16-mixed`` or ``bf16-mixed``). Its kernels also exist
  only for a fixed set of channel widths, which
  :class:`~autoware_ml.ops.sparse_conv.rulebook.padded_conv.PaddedSubMConv3d` pads to.

Models take a ``sparse_conv_backend`` argument, a :class:`SparseConvBackend`, and
build their layers through :meth:`SparseConvBackend.submanifold_conv3d`, so the
choice is one config key and the rest of the code never imports either library at
module level. The configs default to ``spconv``, the historical behaviour; a mixed
precision recipe selects ``rulebook`` (``configs/defaults/amp_rulebook.yaml``), and
``auto`` takes rulebook whenever it is installed and CUDA is available. rulebook is
CUDA-only, so CPU paths (CPU unit tests, CPU inference) keep ``spconv``.
"""

from __future__ import annotations

from enum import StrEnum, auto

import torch
from torch import nn

from autoware_ml.ops.sparse_conv.rulebook.availability import IS_RULEBOOK_AVAILABLE
from autoware_ml.ops.spconv.availability import IS_SPCONV_AVAILABLE


class SparseConvBackend(StrEnum):
    """
    Library implementing the submanifold sparse convolutions.

    Attributes:
      AUTO: rulebook when it is installed and CUDA is available, else spconv.
      SPCONV: ``spconv``; fp16 accumulation, required for ONNX export.
      RULEBOOK: ``rulebook_torch``; fp32 accumulation under autocast, CUDA only.
    """

    AUTO = auto()
    SPCONV = auto()
    RULEBOOK = auto()

    @property
    def package_name(self) -> str:
        """Name of the Python package implementing a concrete backend.

        Raises:
            ValueError: Raised for ``AUTO``, which names no package.
        """
        match self:
            case SparseConvBackend.SPCONV:
                return "spconv"
            case SparseConvBackend.RULEBOOK:
                return "rulebook_torch"
        raise ValueError(f"{self!r} names no package; resolve it first.")

    @property
    def is_installed(self) -> bool:
        """Whether the package of a concrete backend is installed.

        Raises:
            ValueError: Raised for ``AUTO``, which names no package.
        """
        match self:
            case SparseConvBackend.SPCONV:
                return IS_SPCONV_AVAILABLE
            case SparseConvBackend.RULEBOOK:
                return IS_RULEBOOK_AVAILABLE
        raise ValueError(f"{self!r} names no package; resolve it first.")

    def resolve(self) -> SparseConvBackend:
        """Resolve the backend request against the installed libraries.

        Returns:
            ``SPCONV`` or ``RULEBOOK``.

        Raises:
            ModuleNotFoundError: Raised when the requested backend is not installed,
                or under ``AUTO`` when no backend can run.
        """
        if self is SparseConvBackend.AUTO:
            return self._resolve_auto()
        if not self.is_installed:
            raise ModuleNotFoundError(
                f"sparse_conv_backend={self.value!r} requested but {self.package_name!r} "
                "is not installed."
            )
        return self

    @staticmethod
    def _resolve_auto() -> SparseConvBackend:
        """Pick rulebook when it is installed and CUDA is available, else spconv.

        Returns:
            ``SPCONV`` or ``RULEBOOK``.

        Raises:
            ModuleNotFoundError: Raised when rulebook cannot run and spconv is not
                installed.
        """
        if IS_RULEBOOK_AVAILABLE and torch.cuda.is_available():
            return SparseConvBackend.RULEBOOK
        if IS_SPCONV_AVAILABLE:
            return SparseConvBackend.SPCONV
        raise ModuleNotFoundError(
            "sparse_conv_backend='auto' needs 'spconv' unless 'rulebook_torch' is installed "
            "and CUDA is available; neither holds."
        )

    def submanifold_conv3d(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        *,
        bias: bool,
        indice_key: str,
    ) -> nn.Module:
        """Build a stride-1 submanifold convolution on this backend.

        Both backends share the KRSC weight layout ``[out, k, k, k, in]`` and the
        parameter names ``weight``/``bias``, so checkpoints trained on one load on the
        other unchanged.

        Args:
            in_channels: Input feature dimension.
            out_channels: Output feature dimension.
            kernel_size: Cubic kernel size.
            bias: Whether the layer carries a bias.
            indice_key: Key under which the layer caches its index pairs (spconv) or
                rulebook on the sparse tensor, shared by the layers of one stage.

        Returns:
            The convolution module.

        Raises:
            ModuleNotFoundError: Raised when the backend is not installed, see
                :meth:`resolve`.
        """
        if self.resolve() is SparseConvBackend.RULEBOOK:
            from autoware_ml.ops.sparse_conv.rulebook.padded_conv import PaddedSubMConv3d

            return PaddedSubMConv3d(
                in_channels, out_channels, kernel_size=kernel_size, bias=bias, indice_key=indice_key
            )
        import spconv.pytorch as spconv

        return spconv.SubMConv3d(
            in_channels, out_channels, kernel_size=kernel_size, bias=bias, indice_key=indice_key
        )

    @classmethod
    def of_module(
        cls, module: nn.Module, submanifold_only: bool = False
    ) -> SparseConvBackend | None:
        """Return the backend a sparse convolution module belongs to.

        Args:
            module: Module inspected.
            submanifold_only: Match only the ``SubMConv3d`` classes the PTv3 family
                builds. Otherwise every spconv module counts, containers and strided
                layers included.

        Returns:
            ``SPCONV``, ``RULEBOOK``, or ``None`` when the module is not a sparse
            convolution of either backend.
        """
        if cls._is_rulebook_module(module):
            return cls.RULEBOOK
        if cls._is_spconv_module(module, submanifold_only):
            return cls.SPCONV
        return None

    @classmethod
    def is_module(cls, module: nn.Module) -> bool:
        """Return whether a module consumes sparse convolution tensors.

        Args:
            module: Module inspected.

        Returns:
            ``True`` when the module is a sparse convolution module of either backend.
        """
        return cls.of_module(module) is not None

    @staticmethod
    def is_tensor(value: object) -> bool:
        """Return whether a value is the sparse tensor container of either backend.

        Args:
            value: Value inspected.

        Returns:
            ``True`` for ``spconv.SparseConvTensor`` and ``rulebook_torch.SparseTensor``.
        """
        tensor_types: list[type] = []
        if IS_SPCONV_AVAILABLE:
            import spconv.pytorch as spconv

            tensor_types.append(spconv.SparseConvTensor)
        if IS_RULEBOOK_AVAILABLE:
            import rulebook_torch

            tensor_types.append(rulebook_torch.SparseTensor)
        return isinstance(value, tuple(tensor_types))

    @staticmethod
    def _is_rulebook_module(module: nn.Module) -> bool:
        """Return whether a module is a rulebook submanifold convolution.

        Args:
            module: Module inspected.

        Returns:
            ``True`` for ``rulebook_torch.SubMConv3d`` and its subclasses.
        """
        if not IS_RULEBOOK_AVAILABLE:
            return False
        import rulebook_torch

        return isinstance(module, rulebook_torch.SubMConv3d)

    @staticmethod
    def _is_spconv_module(module: nn.Module, submanifold_only: bool) -> bool:
        """Return whether a module is an spconv module.

        Args:
            module: Module inspected.
            submanifold_only: Match only ``spconv.SubMConv3d``.

        Returns:
            ``True`` when the module matches.
        """
        if not IS_SPCONV_AVAILABLE:
            return False
        import spconv.pytorch as spconv

        if submanifold_only:
            return isinstance(module, spconv.SubMConv3d)
        return spconv.modules.is_spconv_module(module)
