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

"""Agent futures of the 3D box slots, the targets of a motion head that forecasts detections."""

from __future__ import annotations

from typing import NamedTuple

import torch
from jaxtyping import Bool, Float32, Int64
from torch import Tensor

from autoware_ml.dataclasses.batch.motion import AgentTracksBatch
from autoware_ml.geometry.bbox_3d.instance_ids import hash_instance_ids


class AgentFutureTargets(NamedTuple):
    """
    Future of the agent of every padded 3D box slot of a batch, in the frames of the augmented
    point clouds.

    A motion head that forecasts its detection queries learns the future of the agent its query
    was matched to, so the agent tracks are aligned with the box slots of the detection ground
    truth by instance identity: the hashed track id of an agent equals the instance id of its box.
    """

    # (B, M, T, 2), BEV center of the agent of every box slot at every future step, zero where
    # the step is invalid
    agent_future_positions: Float32[Tensor, "batch_size max_num_gt_bboxes agent_steps 2"]
    # (B, M, T), whether the slot holds an agent observed at the future step. False for padding
    # slots and for boxes the tracks do not know.
    agent_future_valid: Bool[Tensor, "batch_size max_num_gt_bboxes agent_steps"]

    @staticmethod
    def from_agent_tracks(
        agents: AgentTracksBatch,
        gt_bboxes_instance_ids: Int64[Tensor, "batch_size max_num_gt_bboxes"],
    ) -> AgentFutureTargets:
        """
        Gather the future steps of the agent of every box slot.

        Args:
          agents: Agent tracks of the batch, in the frames of the augmented point clouds.
          gt_bboxes_instance_ids: Instance id of every padded box slot, -1 for padding.

        Returns:
          AgentFutureTargets: The futures of the box slots.

        Raises:
          ValueError: If the tracks and the box slots disagree on the batch size.
        """

        batch_size, num_slots = gt_bboxes_instance_ids.shape
        if agents.batch_size != batch_size:
            raise ValueError(
                f"Got the tracks of {agents.batch_size} samples and the boxes of {batch_size}."
            )
        future = slice(agents.current_step_index + 1, None)
        positions = agents.positions[:, :, future, :2]
        valid = agents.valid[:, :, future]
        num_steps = positions.shape[2]
        device = gt_bboxes_instance_ids.device
        track_ids = torch.full(agents.mask.shape, -1, dtype=torch.int64)
        for index, sample_track_ids in enumerate(agents.track_ids):
            track_ids[index, : len(sample_track_ids)] = torch.from_numpy(
                hash_instance_ids(sample_track_ids)
            )
        # (B, M, A), a padded slot or agent never matches
        matches = (gt_bboxes_instance_ids[:, :, None] >= 0) & (
            gt_bboxes_instance_ids[:, :, None] == track_ids.to(device)[:, None, :]
        )
        if matches.shape[-1] == 0:
            return AgentFutureTargets(
                agent_future_positions=positions.new_zeros(batch_size, num_slots, num_steps, 2),
                agent_future_valid=valid.new_zeros(batch_size, num_slots, num_steps),
            )
        agent = matches.float().argmax(dim=-1)
        slot_valid = (
            valid.gather(1, agent[..., None].expand(-1, -1, num_steps))
            & matches.any(dim=-1)[..., None]
        )
        slot_positions = positions.gather(1, agent[..., None, None].expand(-1, -1, num_steps, 2))
        return AgentFutureTargets(
            agent_future_positions=torch.where(slot_valid[..., None], slot_positions, 0.0),
            agent_future_valid=slot_valid,
        )

    def select_samples(self, mask: Bool[Tensor, " batch_size"]) -> AgentFutureTargets:
        """
        Keep the futures of the samples a mask selects.

        Args:
          mask: True for every sample to keep.

        Returns:
          AgentFutureTargets: The futures of the selected samples.
        """

        return AgentFutureTargets(*(field[mask] for field in self))
