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

"""Submanifold sparse convolution: t4sparse when installed, spconv otherwise.

Both layers share the constructor, the KRSC weight layout and the parameter
names, so checkpoints are interchangeable. t4sparse runs under autocast and for
16-bit features on CUDA; spconv runs everything else, in the dtype of its weights.
"""

from __future__ import annotations

from importlib.util import find_spec

from autoware_ml.ops.submanifold_conv_spconv import SpconvSubMConv3d

__all__ = ["IS_T4SPARSE_AVAILABLE", "SpconvSubMConv3d", "SubMConv3d"]

# find_spec on a dotted name imports the parent package, which raises when it is absent.
IS_T4SPARSE_AVAILABLE = (
    find_spec("t4sparse") is not None and find_spec("t4sparse.torch") is not None
)

if IS_T4SPARSE_AVAILABLE:
    from autoware_ml.ops.submanifold_conv_t4sparse import T4SparseSubMConv3d

    SubMConv3d = T4SparseSubMConv3d
else:
    SubMConv3d = SpconvSubMConv3d
