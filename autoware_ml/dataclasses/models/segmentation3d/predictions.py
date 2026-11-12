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
Modules to save decoded predictions from a segmentation3d model.
"""

from __future__ import annotations

from jaxtyping import Float, Int64
from pydantic import BaseModel, ConfigDict, model_validator

import torch


class Segmentation3DPredictions(BaseModel):
    """
    Dataclass to save the decoded predictions of a 3D segmentation model for every point.

    Attributes:
      pred_labels: Predicted class of every point.
      pred_probs: Class probabilities of every point.
    """

    model_config = ConfigDict(frozen=True, strict=True, arbitrary_types_allowed=True)

    pred_labels: Int64[torch.Tensor, " num_points"]
    pred_probs: Float[torch.Tensor, "num_points num_classes"]

    @model_validator(mode="after")
    def _check_num_points(self) -> Segmentation3DPredictions:
        """Check that the labels and the probabilities describe the same points."""
        if self.pred_labels.shape[0] != self.pred_probs.shape[0]:
            raise ValueError(
                "pred_labels and pred_probs must describe the same points, got "
                f"{self.pred_labels.shape[0]} and {self.pred_probs.shape[0]}."
            )
        return self
