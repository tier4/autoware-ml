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

"""Inference backend names.

A *backend* is whatever runs the model's forward: the PyTorch modules themselves,
an ONNX Runtime session, or a TensorRT engine. PyTorch is a backend like any other
so that ``trainer.test`` and deployment evaluation report under the same metric
keys (``{split}/{backend}/{metric}``).
"""

from __future__ import annotations

from enum import StrEnum


class Backend(StrEnum):
    """Inference backend identifiers used in configs, metric keys, and artifact lookups.

    A ``StrEnum`` so that ``f"{backend}"`` yields the config spelling (``onnx``), which
    is what the metric keys are built from. Construct with ``Backend(name)``; an unknown
    name raises a ``ValueError`` listing the supported ones.
    """

    PYTORCH = "pytorch"
    ONNX = "onnx"
    TENSORRT = "tensorrt"

    @classmethod
    def _missing_(cls, value: object) -> None:
        raise ValueError(f"Unknown backend {value!r}. Supported: {[b.value for b in cls]}.")
