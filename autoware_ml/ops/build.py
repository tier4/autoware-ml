# Copyright 2025 TIER IV, Inc.
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

"""Build helpers for Autoware-ML custom operations.

This module contains small helpers used by native extension build scripts for
custom CUDA and C++ operators.
"""

import logging
import os
from collections.abc import Mapping, Sequence
from pathlib import Path

import torch
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension

logger = logging.getLogger(__name__)


def make_cuda_ext(
    name: str,
    module: str,
    sources: Sequence[str],
    sources_cuda: Sequence[str] | None = None,
    extra_args: Sequence[str] | None = None,
    extra_include_path: Sequence[str] | None = None,
):
    """Create a C++ or CUDA extension for a module.

    Args:
        name: Extension module name.
        module: Python package that owns the extension.
        sources: C++ source files relative to ``module``.
        sources_cuda: Optional CUDA source files relative to ``module``.
        extra_args: Additional compiler arguments.
        extra_include_path: Additional include directories.

    Returns:
        Configured PyTorch extension object.
    """
    sources = list(sources)
    sources_cuda = list(sources_cuda or [])
    extra_args = list(extra_args or [])
    extra_include_path = list(extra_include_path or [])

    define_macros = []
    extra_compile_args = {"cxx": [] + extra_args}
    extension = CppExtension

    if torch.cuda.is_available() or os.getenv("FORCE_CUDA", "0") == "1":
        define_macros.append(("WITH_CUDA", None))
        extension = CUDAExtension
        extra_compile_args["nvcc"] = extra_args + [
            "-D__CUDA_NO_HALF_OPERATORS__",
            "-D__CUDA_NO_HALF_CONVERSIONS__",
            "-D__CUDA_NO_HALF2_OPERATORS__",
            "-gencode=arch=compute_80,code=sm_80",
            "-gencode=arch=compute_86,code=sm_86",
            "-gencode=arch=compute_89,code=sm_89",
            "-gencode=arch=compute_90,code=sm_90",
            "-gencode=arch=compute_120,code=sm_120",
        ]
        sources += sources_cuda
    else:
        logger.info(f"Compiling {name} without CUDA")

    return extension(
        name=f"{module}.{name}",
        sources=[str(Path(*module.split("."), path)) for path in sources],
        include_dirs=extra_include_path,
        define_macros=define_macros,
        extra_compile_args=extra_compile_args,
    )


def get_ext_modules() -> list[CppExtension | CUDAExtension]:
    """Return extension modules shipped with Autoware-ML ops.

    Returns:
        List of PyTorch extension module definitions.
    """
    return [
        make_cuda_ext(
            name="bev_pool_ext",
            module="autoware_ml.ops.bev_pool",
            sources=[
                "src/bev_pool.cpp",
                "src/bev_pool_cuda.cu",
            ],
        ),
    ]


def get_cmdclass() -> Mapping[str, type[BuildExtension]]:
    """Return ``setuptools`` command classes for Torch extensions.

    Returns:
        Mapping of custom setuptools command classes.
    """
    return {"build_ext": BuildExtension}
