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

"""Build helpers for Autoware-ML custom operations."""

import logging
import os
from pathlib import Path

import torch
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension

logger = logging.getLogger(__name__)


def make_cuda_ext(
    name: str,
    module: str,
    sources: list[str],
    sources_cuda: list[str] | None = None,
    extra_args: list[str] | None = None,
    extra_include_path: list[str] | None = None,
):
    """Create a CUDA/CPP extension for the given module."""
    if sources_cuda is None:
        sources_cuda = []
    if extra_args is None:
        extra_args = []
    if extra_include_path is None:
        extra_include_path = []

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


def get_ext_modules():
    """Return extension modules shipped with Autoware-ML ops."""
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


def get_cmdclass():
    """Return setuptools cmdclass for Torch extensions."""
    return {"build_ext": BuildExtension}
