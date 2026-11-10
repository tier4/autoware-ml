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

"""
Modules to save raw outputs from a segmentation3d head.
"""

from __future__ import annotations

from jaxtyping import Float
from pydantic import BaseModel, ConfigDict, InstanceOf

import torch


class Segmentation3DHeadOutputs(BaseModel):
    """
    Dataclass to save the raw outputs of a 3D segmentation head.

    Attributes:
      logits: Class logits of every point or voxel the head predicts for.
      auxiliary_features: Feature maps the auxiliary heads of the model read, empty for a model
        without auxiliary heads.
    """

    model_config = ConfigDict(frozen=True, strict=True, arbitrary_types_allowed=True)

    logits: Float[torch.Tensor, "num_points num_classes"]
    auxiliary_features: tuple[InstanceOf[torch.Tensor], ...] = ()
