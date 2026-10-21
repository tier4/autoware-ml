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

"""Interface of the middle encoders that turn voxel features into a dense BEV map."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch.nn as nn


class BEVMiddleEncoder(nn.Module, ABC):
    """A middle encoder the TransFusion and BEVFusion lidar branches can run and export.

    The branches read the dense BEV shape to size the maps they fuse and swap the encoder for
    its ONNX-exportable variant at deployment, so both are part of the contract rather than
    attributes an encoder happens to have.
    """

    @property
    @abstractmethod
    def bev_output_shape(self) -> tuple[int, int]:
        """Return the ``(height, width)`` of the dense BEV map the encoder produces."""

    @abstractmethod
    def prepare_for_export(self) -> BEVMiddleEncoder:
        """Return an ONNX-exportable variant of the encoder, the encoder itself when it is one."""
