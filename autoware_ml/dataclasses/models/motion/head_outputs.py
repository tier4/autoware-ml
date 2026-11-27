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
Modules to save raw outputs from a motion head.
"""

from __future__ import annotations

from jaxtyping import Bool, Float
from pydantic import BaseModel, ConfigDict, model_validator

import torch


class MotionHeadOutputs(BaseModel):
    """
    Dataclass to save the raw outputs of a motion head.

    Agent trajectories are BEV positions at every future step in the frame of the (augmented)
    point cloud, relative to the box center of their detection query.

    Attributes:
      agent_mode_logits: Score logit of every mode of every detection query.
      agent_trajectories: Trajectory of every mode of every detection query.
    """

    model_config = ConfigDict(frozen=True, strict=True, arbitrary_types_allowed=True)

    agent_mode_logits: Float[torch.Tensor, "batch_size num_queries agent_modes"]
    agent_trajectories: Float[torch.Tensor, "batch_size num_queries agent_modes agent_steps 2"]

    @model_validator(mode="after")
    def _check_agent_shapes(self) -> MotionHeadOutputs:
        """Check that the scores and the trajectories describe the same modes."""
        if tuple(self.agent_trajectories.shape[:3]) != tuple(self.agent_mode_logits.shape):
            raise ValueError(
                "agent_trajectories must lead with the shape of agent_mode_logits "
                f"{tuple(self.agent_mode_logits.shape)}, got "
                f"{tuple(self.agent_trajectories.shape)}."
            )
        return self

    def select_samples(self, mask: Bool[torch.Tensor, " batch_size"]) -> MotionHeadOutputs:
        """
        Keep the outputs of the samples a mask selects.

        Args:
            mask: True for every sample to keep.

        Returns:
            The outputs of the selected samples.
        """
        return MotionHeadOutputs(
            agent_mode_logits=self.agent_mode_logits[mask],
            agent_trajectories=self.agent_trajectories[mask],
        )
