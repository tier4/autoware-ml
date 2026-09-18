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

from enum import Enum


class Backend(str, Enum):
    """Inference backend identifiers used in configs, metric keys, and artifact lookups."""

    PYTORCH = "pytorch"
    ONNX = "onnx"
    TENSORRT = "tensorrt"

    @classmethod
    def parse(cls, value: str | Backend) -> Backend:
        """Parse a backend name, raising a readable error on an unknown one."""
        if isinstance(value, cls):
            return value

        try:
            return cls(value)
        except ValueError as error:
            raise ValueError(
                f"Unknown backend {value!r}. Supported: {[b.value for b in cls]}."
            ) from error

    @property
    def artifact_suffix(self) -> str:
        """File suffix of the exported artifact this backend runs.

        Raises:
            ValueError: For ``PYTORCH``, which runs the model itself and has no artifact —
                the same answer :func:`~autoware_ml.deployment.stages.artifact_path` gives
                to the same question.
        """
        if self is Backend.PYTORCH:
            raise ValueError("The pytorch backend has no exported artifact, so no suffix.")
        return {Backend.ONNX: ".onnx", Backend.TENSORRT: ".engine"}[self]
